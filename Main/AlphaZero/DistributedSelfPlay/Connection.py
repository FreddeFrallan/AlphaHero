from Main.AlphaZero.DistributedSelfPlay import Constants as C

CONNECTION_ID = 0


class Connection:

    def __init__(self, ip=None, port=None, server=False):
        global CONNECTION_ID
        print("Ip: {}  Port: {}  BufferSize: {}  Server: {}".format(ip, port, C.BUFFER_SIZE, server))
        self.data = b''
        self.id = CONNECTION_ID
        CONNECTION_ID += 1

        if (server):
            self._startServerConnection(ip, port)
        else:
            self._startClientConnection(ip, port)

    def _startClientConnection(self, ip, port):
        import socket

        # print("Starting connection for incoming worker:", port)
        print("Connecting to server on port", str(port) + "...")
        self.conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.conn.connect((ip, port))
        # print("Worker connected!")
        print("Connected to server on port", str(port) + "!")

    def _startServerConnection(self, ip, port):
        import socket
        print("Waiting for client on port", str(port) + "...")
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind((ip, port))
        s.listen(1)
        self.conn, addr = s.accept()
        # print("Connection established to overlord")
        print("Connection established to port", str(port) + "!")

    def readMessage(self):
        import pickle

        hasFileSize = False
        fileSize = 0
        while (True):
            if (hasFileSize or len(self.data) >= C.HEADER_MSG_SIZE):

                if (hasFileSize == False):
                    fileSize = int.from_bytes(self.data[:C.HEADER_MSG_SIZE], C.HEADER_ENDIAN_TYPE)
                    hasFileSize = True

                if (len(self.data) >= fileSize + C.HEADER_MSG_SIZE):
                    # Extract the message
                    msg = self.data[C.HEADER_MSG_SIZE: C.HEADER_MSG_SIZE + fileSize]

                    # Remove the message from the stored Q
                    self.data = self.data[C.HEADER_MSG_SIZE + fileSize:]

                    return pickle.loads(msg)

                deltaSize = fileSize - len(self.data) + C.HEADER_MSG_SIZE
                newData = self.conn.recv(deltaSize)
                self.data += newData
            else:
                newData = self.conn.recv(C.BUFFER_SIZE)
                self.data += newData

    def _encodeMsg(self, content):
        sizeInBytes = len(content).to_bytes(C.HEADER_MSG_SIZE, C.HEADER_ENDIAN_TYPE)
        return sizeInBytes + content

    def sendMessage(self, messageType, msg):
        import pickle
        outMsg = pickle.dumps((messageType, msg))
        self.conn.sendall(self._encodeMsg(outMsg))

    def close(self):
        self.conn.close()
