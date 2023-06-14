import socket
from _thread import start_new_thread
import threading

class AzureServer:

    def __init__(self, server_ip = "0.0.0.0", server_port = 65432):
        self.SERVER_IP = server_ip
        self.PORT = server_port
        self.ADDR = (self.SERVER_IP, self.PORT)
        self.MSG_LENGTH = 1024

        self.message_to_client = ["TRAIN", "SESSION DONE"]

    def run_server(self):

        self.print_lock = threading.Lock()

        self._open_socket()
        
        try:
            while True: 
                self._process_client_connections()

        except KeyboardInterrupt as e:
            print("Keyboard interrupt")
            self.s.close()
            print("Socket closed.")

    def _open_socket(self):
        self.s =  socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.s.bind(self.ADDR)
        self.s.listen()
        print(f"[SERVER] Listening on {self.SERVER_IP} port {self.PORT}")
        
    def _process_client_connections(self):

            self.stop_session = False

            conn, addr = self.s.accept()
            new_client = (conn, addr)

            print("[SERVER] Connected to: ", addr[0], "port", addr[1])

            self.process_client(new_client)

        
    def process_client(self, connected_client):

        self.print_lock.acquire()
        start_new_thread(self.server_task, (connected_client, ))
        self.print_lock.release()

    def server_task(self, connected_client):

        conn, _ = connected_client

        while True:

            data = conn.recv(self.MSG_LENGTH)

            if not data:
                print("[SERVER] Connection exit.")
                break
            
            self._parse_client_message(connected_client, data)

            if self.stop_session:
                print("[SERVER] Connection exit.")
                break

        conn.close()

    def _parse_client_message(self, connected_client, data):

        client_message = str(data, "utf-8")
        conn, addr = connected_client

        if client_message == "HELLO":
            print(f"[SERVER] Received msg from client: {client_message}")
            print("[SERVER] Scheduling task on virtual vehicle.")
            conn.sendall(self.message_to_client[0].encode("utf-8"))
            print("[SERVER] Waiting for results...")

        elif client_message == "RESULTS":
            print("[SERVER] Received results from virtual vehicle.")
            print(f"[SERVER] Ending session with client on {addr[0]}.")
            conn.sendall(self.message_to_client[1].encode("utf-8"))
            self.stop_session = True

        else:
            print("[SERVER] Invalid messaged received from client.")
            print(f"[SERVER] Received message: {client_message}. Ignoring message.")
            conn.sendall(self.message_to_client[1].encode("utf-8"))
            self.stop_session = True

def main():
    
    server = AzureServer()
    server.run_server()

if __name__ == "__main__":
    main()