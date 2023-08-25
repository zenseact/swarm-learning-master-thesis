import struct
import socket
import pickle


def recieve_large_message(conn: socket):
    header_size = struct.calcsize("!I")
    header_data = b""

    while len(header_data) < header_size:
        chunk = conn.recv(header_size - len(header_data))
        header_data += chunk

    # Unpack the message size from the header
    message_size = struct.unpack("!I", header_data)[0]

    # Receive the pickled message data
    data = b""

    while len(data) < message_size:
        chunk = conn.recv(message_size - len(data))
        data += chunk

    # Unpickle the received data to get the original object
    return pickle.loads(data)


def send_large_message(conn: socket, message_to_send):
    pickled_message = pickle.dumps(message_to_send)
    message_size = len(pickled_message)
    header = struct.pack("!I", message_size)
    conn.sendall(header)
    conn.sendall(pickled_message)
