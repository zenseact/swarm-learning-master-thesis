import socket
import threading
import time
import select
import pickle


class AGX:
    def __init__(self, agx_ip="0.0.0.0", agx_port=59999):
        self.AGX_IP = agx_ip
        self.PORT = agx_port
        self.ADDR = (self.AGX_IP, self.PORT)
        self.MSG_LENGTH = 1024
        self.message_to_virtual_vehicle = ["TASK_SCHEDULED"]
        self.message_from_virtual_vehicle = ["HELLO", "TRAIN"]
        self.BUFFER = []
        self.CLIENT_ID = 0
        self.MAX_BUFFER_SIZE = 100
        self.TIMEOUT = 1

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
            print(f"[AGX] Number of connected vehicles in qeue: {len(self.BUFFER)}")

            client_socket, client_address = self.BUFFER.pop(0)
            new_virtual_vehicle = (client_socket, client_address)
            self.CLIENT_ID += 1
            print(f"[AGX] Processing virtual vehicle {self.CLIENT_ID}: {client_address}")
            self._process_virtual_vehicle(new_virtual_vehicle)

    def _process_virtual_vehicle(self, virtual_vehicle):
        conn, _ = virtual_vehicle

        while True:
            data = conn.recv(self.MSG_LENGTH)

            if not data:
                print("[AGX] Exit connection with client.\n")
                break

            stop_session = self._parse_virtual_vehicle_message(
                virtual_vehicle, data
            )

            if stop_session:
                print("[AGX] Exit connection with client.\n")
                break

        conn.close()

    def _parse_virtual_vehicle_message(self, connected_virtual_vehicle, data):
        virtual_vehicle_message = pickle.loads(data)
        conn, addr = connected_virtual_vehicle
        stop_session = False

        if virtual_vehicle_message == self.message_from_virtual_vehicle[0]:
            print(f"[AGX] Received msg from virtual vehicle: {virtual_vehicle_message}")
            conn.sendall(pickle.dumps(self.message_to_virtual_vehicle[0]))

        elif virtual_vehicle_message == self.message_from_virtual_vehicle[1]:
            print(f"[AGX] Received msg from virtual vehicle: {virtual_vehicle_message}")
            print("[AGX] Training started")
            results = self._train()
            print("[AGX] Training completed")
            print(f"[AGX] Ending session with vehicle {addr}.")
            conn.sendall(pickle.dumps(results))
            stop_session = True

        else:
            print("[AGX] Invalid messaged received from vv.")
            print(
                f"[AGX] Received message: {virtual_vehicle_message}. Ignoring message."
            )
            stop_session = True

        return stop_session

    def _train(self):
        time.sleep(4)
        results = {'weights': []}
        return results


def main():
    scheduler = AGX()
    scheduler.run_scheduler()


if __name__ == "__main__":
    main()
