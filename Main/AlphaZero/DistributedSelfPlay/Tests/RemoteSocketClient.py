#!/usr/bin/env python3
import socket

HOST = '127.0.0.1'  # The server's hostname or IP address
PORT = 56025        # The port used by the server

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    print(HOST, PORT)
    s.connect((HOST, PORT))
    print("Connected")
    while(True):
        msg = input("msg: ")
        s.sendall(bytes(msg, 'utf-8'))
        data = s.recv(1024)
        print("Got msg:", data.decode('utf-8'))
