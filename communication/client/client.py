import socket
class VirtualVehicle:


    def __init__(self, server_ip = "0.0.0.0", server_port = 65432):

        self.SERVER_IP = server_ip
        self.PORT = server_port
        self.ADDR = (self.SERVER_IP, self.PORT)
        self.MSG_LENGTH = 1024

        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        self.message_to_server = ["HELLO", "RESULTS", "GOODBYE"]

    def run_client(self):
        self.s.connect(self.ADDR)

        self.s.send(self.message_to_server[0].encode("utf-8"))

        while True:

            data = self.s.recv(self.MSG_LENGTH)

            if str(data, "utf-8") == "TRAIN":
                self._client_do_task(str(data, "utf-8"))

            elif str(data, "utf-8") == "SESSION DONE":
                print("[CLIENT] Received msg from server: SESSION DONE")
                print("[CLIENT] Terminating session...")
                break

        self.s.close()
        print("[CLIENT] Socket closed.")

    def _client_do_task(self, server_message):
        print(f"[CLIENT] Received msg from server: {server_message}")
        self._train_on_AGX()
        self._send_results_to_cloud()

    def _train_on_AGX(self):
        print("[CLIENT] Schedules training task on AGX...")
        print("[CLIENT] Received results from AGX." )

    def _send_results_to_cloud(self):
        print("[CLIENT] Sends results to cloud server...")
        self.s.send(self.message_to_server[1].encode("utf-8"))
        print("[CLIENT] Done.")

def main():
    client = VirtualVehicle()
    client.run_client()

if __name__ == "__main__":
    main()