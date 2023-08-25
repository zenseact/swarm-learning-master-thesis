import socket
from common.socket_utils import recieve_large_message, send_large_message
import time


class VirtualVehicle:
    def __init__(
        self,
        server_ip="172.25.16.71",
        server_port=65432,
        agx_ip="agx6.nodes.lab.ai.se",
        agx_port=59999,
    ):
        self.SERVER_IP = server_ip
        self.SERVER_PORT = server_port
        self.SERVER_ADDR = (self.SERVER_IP, self.SERVER_PORT)

        self.AGX_IP = agx_ip
        self.AGX_PORT = agx_port
        self.AGX_ADDR = (self.AGX_IP, self.AGX_PORT)

        self.socket_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket_agx = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def _train_on_AGX(self, message_from_server):
        message_from_agx = None
        client_id = message_from_server["data"]["client_id"]
        while True:
            try:
                self.socket_agx.connect(self.AGX_ADDR)
                break
            except:
                print("[CLIENT] agx not ready, retrying")
                time.sleep(10)
        # Should update the train queue at the AGX
        message_to_agx = {"message": "HELLO", "data": {"client_id": client_id}}
        send_large_message(self.socket_agx, message_to_agx)

        while True:
            message_from_agx = recieve_large_message(self.socket_agx)

            if message_from_agx["message"] == "TASK_SCHEDULED":
                print("[CLIENT] Scheduling training task to AGX...")
                send_large_message(self.socket_agx, message_from_server)

            elif message_from_agx["message"] == "RESULTS":
                print("[CLIENT] Received training results from AGX")
                print("[CLIENT] Terminating AGX session...")
                break

        self.socket_agx.close()
        self.socket_agx = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print("[CLIENT] AGX socket closed.")
        return message_from_agx

    def _send_results_to_server(self, results: dict):
        print("[CLIENT] Sends results to server...")
        send_large_message(self.socket_server, results)
        print("[CLIENT] Done.\n")

    def run_client(self):
        while True:
            try:
                self.socket_server.connect(self.SERVER_ADDR)
                break
            except:
                print("[CLIENT] Server not ready, retrying..")
                time.sleep(10)

        message_to_server = {
            "message": "HELLO",
            "data": {},
        }
        send_large_message(self.socket_server, message_to_server)
        print("[CLIENT] Notified availablility to server.\n")

        while True:
            # Should receive model, partitions and configs here
            message_from_server = recieve_large_message(self.socket_server)

            if message_from_server["message"] == "START":
                print("[CLIENT] Received start msg from server")
                results = self._train_on_AGX(message_from_server)
                self._send_results_to_server(results)

            elif message_from_server["message"] == "STOP":
                print("[CLIENT] Terminating server session...")
                break

        self.socket_server.close()
        print("[CLIENT] Server socket closed.\n")


def main():
    client = VirtualVehicle()
    client.run_client()


if __name__ == "__main__":
    main()
