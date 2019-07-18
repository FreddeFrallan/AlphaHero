#!/usr/bin/env python3

import socket, time

HOST = '127.0.0.1'  # Standard loopback interface address (localhost)
PORT = 10047  # Port to listen on (non-privileged ports are > 1023)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    print("Waiting for connection...")
    conn, addr = s.accept()
    with conn:
        print('Connected by', addr)
        while True:
            data = conn.recv(1024)
            print("Got msg:", data.decode('utf-8'))
            continue
            time.sleep(1)
            msg = "Das msg"
            s.sendall(bytes(msg, 'utf-8'))
