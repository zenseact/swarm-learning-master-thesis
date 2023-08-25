import socket
import threading
import select
from common.socket_utils import recieve_large_message, send_large_message
from common.utilities import net_instance, get_parameters
from zod import ZodFrames
from scheduler.train import train_model


class AGX:
    def __init__(self, agx_ip="0.0.0.0", agx_port=59999):
        self.AGX_IP = agx_ip
        self.PORT = agx_port
        self.ADDR = (self.AGX_IP, self.PORT)
        self.BUFFER = []
        self.MAX_BUFFER_SIZE = 100
        self.TIMEOUT = 1
        self.model = net_instance("agx")
        self.zod_frames = ZodFrames(dataset_root="/mnt/ZOD2", version="full")

    def run_scheduler(self):
        self.print_lock = threading.Lock()
        self._open_socket()
        try:
            while True:
                self._update_queue()
                self._process_queue()

        except KeyboardInterrupt as e:
            print("Keyboard interrupt")
            self.socket.close()
            print("Socket closed.")

    def _open_socket(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(self.ADDR)
        self.socket.listen(self.MAX_BUFFER_SIZE)
        print("[AGX] AGX started")
        print(f"[AGX] Listening on {self.AGX_IP} port {self.PORT}\n")

    def _update_queue(self):
        while True:
            # use select to determine readable sockets
            readable_sockets, _, _ = select.select([self.socket], [], [], self.TIMEOUT)

            if readable_sockets:
                for readable_socket in readable_sockets:
                    client_socket, client_address = readable_socket.accept()

                    # add the client socket to the list of connected virtual vehicles
                    self.BUFFER.append((client_socket, client_address))
                    print(f"[AGX] Adding client {client_address} to queue")
            else:
                break  # if timeout

    def _process_queue(self):
        if self.BUFFER:
            print(f"\n[AGX] Number of connected vehicles in queue: {len(self.BUFFER)}")

            client_socket, client_address = self.BUFFER.pop(0)
            new_virtual_vehicle = (client_socket, client_address)
            print(f"[AGX] Processing virtual vehicle: {client_address}")
            self._process_virtual_vehicle(new_virtual_vehicle)

    def _process_virtual_vehicle(self, virtual_vehicle):
        conn, addr = virtual_vehicle

        while True:
            message_from_virtual_vehicle = recieve_large_message(conn)

            if message_from_virtual_vehicle["message"] == "HELLO":
                client_id = message_from_virtual_vehicle["data"]["client_id"]
                print("[AGX] Received msg from virtual vehicle: HELLO")
                message_to_virtual_vehicle = {
                    "message": "TASK_SCHEDULED",
                    "data": {"client_id": client_id},
                }
                send_large_message(conn, message_to_virtual_vehicle)

            elif message_from_virtual_vehicle["message"] == "START":
                print("[AGX] Received msg from virtual vehicle: START")
                print("[AGX] Training started")

                train_model(message_from_virtual_vehicle, self.model, self.zod_frames)

                message_to_virtual_vehicle = {
                    "message": "RESULTS",
                    "data": {
                        "client_id": message_from_virtual_vehicle["data"]["client_id"],
                        "results": get_parameters(self.model),
                        "failures": "todo",
                    },
                }

                print("[AGX] Training completed")
                print(f"[AGX] Ending session with vehicle {addr}.")
                send_large_message(conn, message_to_virtual_vehicle)
                break

            else:
                print("[AGX] Invalid messaged received from vv.")
                print(
                    f"[AGX] Received message: {message_from_virtual_vehicle}."
                    + "Ignoring message."
                )
                break

        print("[AGX] Exit connection with client.\n")
        conn.close()


def main():
    scheduler = AGX()
    scheduler.run_scheduler()


if __name__ == "__main__":
    main()
