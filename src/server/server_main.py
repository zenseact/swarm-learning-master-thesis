from flwr.common.typing import Dict, Optional, Tuple, List
from logging import INFO
from _thread import start_new_thread
import threading
import numpy as np
import pickle
import socket
import time
from torch.utils.data import DataLoader
from typing import Any
import os
import glob

from common.static_params import global_configs, TASK, PartitionStrategy
from common.utilities import (
    test,
    net_instance,
    get_parameters,
    set_parameters,
)
from common.logger import fleet_log
from common.datasets import ZODImporter
from common.socket_utils import recieve_large_message, send_large_message
from server.data_partitioner import partition_train_data
from server.fleet_aggregators import BaseStrategy


class GpuServer:
    def __init__(self, server_ip="0.0.0.0", server_port=65432):
        self.SERVER_IP = server_ip
        self.PORT = server_port
        self.ADDR = (self.SERVER_IP, self.PORT)
        self.global_model = net_instance("server")
        self.connected_clients = []
        self.number_of_clients = global_configs.NUM_CLIENTS
        self.zod_importer = ZODImporter(
            subset_factor=global_configs.SUBSET_FACTOR,
            img_size=global_configs.IMG_SIZE,
            batch_size=global_configs.BATCH_SIZE,
            tb_path=global_configs.TB_PATH,
            stored_gt_path=global_configs.STORED_GROUND_TRUTH_PATH,
        )
        self.data_partitions = partition_train_data(
            strat=PartitionStrategy.RANDOM,
            no_clients=int(  # todo fix
                global_configs.NUM_CLIENTS * (1 / global_configs.PERCENTAGE_OF_DATA)
            ),
            zod_importer=self.zod_importer,
        )
        self.strategy = BaseStrategy()
        self.current_round_results = []

    def _open_socket(self):
        """Open a socket to client connections and start listen."""
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.s.bind(self.ADDR)
        self.s.listen()
        print("[SERVER] Server started")
        print(f"[SERVER] Listening on {self.SERVER_IP} port {self.PORT}\n")

    def _wait_for_clients_to_connect(self):
        """Allow a specified number of clients to connect to server.

        When a client has connected, add the client to a buffer.
        """
        while len(self.connected_clients) < self.number_of_clients:
            conn, addr = self.s.accept()
            new_client = (conn, addr)
            message_from_client = self._receive_message_from_client(new_client)
            if message_from_client["message"] == "HELLO":
                print(f"[SERVER] Connected to client: {addr[0]} port {addr[1]}")
                self.connected_clients.append(new_client)

    def _create_dataloader(self) -> DataLoader:
        """Create a server side test dataloader for ZOD.

        Returns:
            DataLoader: _description_
        """
        self.testloader = self.zod_importer.load_test_dataset()
        fleet_log(INFO, f"len testloader batches: {len(self.testloader)}")
        return self.testloader

    def _send_model_and_partitions(self):
        """Send ML model and data partitions to clients."""
        for client_id, connected_client in enumerate(self.connected_clients):
            message_to_client = {
                "message": "START",
                "data": {
                    "client_id": client_id,
                    "model": get_parameters(self.global_model),
                    "partitions": self.data_partitions[str(client_id)],
                    "configs": None,
                },
            }

            self.print_lock.acquire()
            start_new_thread(
                self._send_message_to_client,
                (
                    connected_client,
                    "Scheduling task",
                    message_to_client,
                    client_id,
                ),
            )
            self.print_lock.release()

    def _receive_fit_results(self) -> List[Tuple[Any, Any]]:
        """Receive results from clients.

        Returns:
            Tuple[Any, Any]: Results and failures
        """
        # wait for client results in threads
        for connected_client in self.connected_clients:
            self.print_lock.acquire()
            start_new_thread(
                self._receive_message_from_client,
                (connected_client,),
                {"save_as_round_results": True},
            )
            self.print_lock.release()

        # wait until all results are received and saved
        while len(self.current_round_results) < self.number_of_clients:
            pass

        # collect results in list
        results = []
        for message_from_client in self.current_round_results:
            results.append(
                (
                    message_from_client["data"]["results"],
                    message_from_client["data"]["failures"],
                )
            )
            self.current_round_results = []
        return results

    def _evaluate_model(self, model, testloader):
        fleet_log(INFO, "testing model on server side test set")
        loss, accuracy = test(model, testloader)

        if global_configs.ML_TASK == TASK.CLASSIFICATION:
            fleet_log(
                INFO,
                f"Server-side evaluation loss {float(loss)} / accuracy {float(accuracy)}",
            )
            return float(loss), {"accuracy": float(accuracy)}
        else:
            fleet_log(INFO, f"Server-side evaluation loss {float(loss)}")
            return float(loss), {}

    def _send_stop_message(self):
        """Send stop message to clients."""
        for client_id, connected_client in enumerate(self.connected_clients):
            message_to_client = {
                "message": "STOP",
                "data": {
                    "client_id": client_id,
                },
            }

            self.print_lock.acquire()
            start_new_thread(
                self._send_message_to_client,
                (
                    connected_client,
                    "Sending stop message",
                    message_to_client,
                    client_id,
                ),
            )
            self.print_lock.release()

    def _send_message_to_client(
        self,
        connected_client: tuple,
        info: str,
        message_to_send: dict,
        client_id: str = "",
    ) -> None:
        """Send a message to a client.

        Args:
            connected_client (tuple): Connected FL client
            message (dict): Message to send
        """
        conn, _ = connected_client
        print(f"[SERVER] {info} on virtual vehicle {client_id}.")
        send_large_message(conn, message_to_send)
        print(f"[SERVER] msg data transmitted to virtual vehicle {client_id}")

    def _receive_message_from_client(
        self, connected_client: tuple, save_as_round_results=False
    ) -> Any:
        """Receive a message from a client.

        If save_as_round_results is set, also append the results to
        current_round_results list.

        Args:
            connected_client (tuple): Connected FL client

        Returns:
            message (dict): Message form client
        """
        conn, _ = connected_client
        message = recieve_large_message(conn)

        if save_as_round_results:
            self.current_round_results.append(message)

        return message

    def run_server(self) -> None:
        """Server main script."""
        # remove any leftover files from previous runs that might be broken :)
        files = glob.glob("tmp/fit_results/*.pickle")
        for f in files:
            os.remove(f)
        self.testloader = self._create_dataloader()

        while True:
            self.print_lock = threading.Lock()
            self._open_socket()
            try:
                self._wait_for_clients_to_connect()

                global_round = 1
                while global_round <= global_configs.NUM_GLOBAL_ROUNDS:
                    self._send_model_and_partitions()
                    results = self._receive_fit_results()

                    agg_weights_state_dict, _ = self.strategy.aggregate_fit_fedavg(
                        results
                    )

                    set_parameters(self.global_model, agg_weights_state_dict)

                    loss, _ = self._evaluate_model(self.global_model, self.testloader)

                    print(
                        f"[SERVER] Finished global round {global_round} "
                        + f"out of {global_configs.NUM_GLOBAL_ROUNDS}\n"
                    )
                    global_round += 1

                self._send_stop_message()

                # save parameters
                timestr = time.strftime("%Y%m%d-%H%M%S")
                model_params = get_parameters(self.global_model)
                model_params = np.array(model_params, dtype=object)
                filename = f"tmp/global_params/{timestr}.npz"
                np.savez(filename, model_params)

                # Finish FL
                self.s.close()
                self.connected_clients = []
                print("[SERVER] FL session finished.\n")

            except KeyboardInterrupt as e:
                print("[SERVER] Keyboard interrupt")
                self.s.close()
                print("[SERVER] Socket closed.")
                break


def main():
    server = GpuServer()
    server.run_server()


if __name__ == "__main__":
    main()
