import socket
import pickle


class VirtualVehicle:
    def __init__(
        self,
        server_ip="0.0.0.0",
        server_port=65432,
        agx_ip="0.0.0.0",
        agx_port=59999,
    ):
        self.SERVER_IP = server_ip
        self.SERVER_PORT = server_port
        self.SERVER_ADDR = (self.SERVER_IP, self.SERVER_PORT)
        self.message_to_server = ["HELLO"]
        self.message_from_server = ["DO_TASK", "SESSION DONE"]

        self.AGX_IP = agx_ip
        self.AGX_PORT = agx_port
        self.AGX_ADDR = (self.AGX_IP, self.AGX_PORT)
        self.message_to_agx = ["HELLO", "TRAIN"]
        self.message_from_agx = ["TASK_SCHEDULED"]

        self.MSG_LENGTH = 1024
        self.socket_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket_agx = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def run_client(self):
        self.socket_server.connect(self.SERVER_ADDR)
        self.socket_server.send(pickle.dumps(self.message_to_server[0]))

        while True:

            data = self.socket_server.recv(self.MSG_LENGTH)
            server_message = pickle.loads(data)

            if server_message == self.message_from_server[0]:
                self._train(server_message)

            elif server_message == self.message_from_server[1]:
                print(
                    f"[CLIENT] Received msg from server: {self.message_from_server[1]}"
                )
                print("[CLIENT] Terminating server session...")
                break

        self.socket_server.close()
        print("[CLIENT] Server socket closed.\n")

    def _train(self, server_message):
        print(f"[CLIENT] Received msg from server: {server_message}")
        results = self._train_on_AGX()
        self._send_results_to_cloud(results)

    def _train_on_AGX(self):
        results = None
        self.socket_agx.connect(self.AGX_ADDR)
        self.socket_agx.send(pickle.dumps(self.message_to_agx[0]))

        while True:
            data = self.socket_agx.recv(self.MSG_LENGTH)
            results = pickle.loads(data)
            if results == self.message_from_agx[0]:
                print("[CLIENT] Scheduling training task to AGX...")
                self.socket_agx.send(pickle.dumps(self.message_to_agx[1]))

            else:
                print(f"[CLIENT] Received msg from AGX: {results}")
                print("[CLIENT] Terminating AGX session...")
                break

        self.socket_agx.close()
        print("[CLIENT] AGX socket closed.")
        return results

    def _send_results_to_cloud(self, results: dict):
        print("[CLIENT] Sends results to cloud server...")
        self.socket_server.send(pickle.dumps(results))
        print("[CLIENT] Done.")


def main():
    client = VirtualVehicle()
    client.run_client()


if __name__ == "__main__":
    main()
