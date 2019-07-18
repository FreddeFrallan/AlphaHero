from sshtunnel import SSHTunnelForwarder

HOST_IP_INDEX = 0
HOST_PORT_INDEX = 1
HOST_USERNAME_INDEX = 2

SELF_PORTS = [20000, 20001, 20002, 20003, 20004]

HOSTS = {
    "jorgCPU": ('130.237.37.22', 56026, "dd142x"),
    "jorgGPU": ('130.237.37.22', 56027, "dd142x"),
    "jorgCPULocal": ('172.16.222.26', 22, "dd142x"),
    "jorgGPULocal": ('172.16.222.27', 22, "dd142x")
}

WORKERS = {
    "self1": SELF_PORTS[1],
    "self2": SELF_PORTS[2],
    "self3": SELF_PORTS[3],
    "self4": SELF_PORTS[4],
    "fenrir1": 5000,
    "fenrir2": 5001,
    "GTX1070": 6000,
    "google1": 7000,
    "google2": 9000,
    "joey": 10000,
    "linus": 11000,
    "jorgCPU": 12000,
    "jorgGPU": 13000,
}

TRAINERS = {
    "self": SELF_PORTS[0],
    "fenrir": 5002,
    "jorgCPU": 13001,
    "jorgGPU": 13002
}


def createSSHTunnel(host, client):
    remoteIP, remotePort, username = host
    bindPort = client
    password = input("Enter password for " + str(username) + ": ")

    server = SSHTunnelForwarder(
        (remoteIP, remotePort),
        ssh_username=username,
        ssh_password=password,
        remote_bind_address=('127.0.0.1', bindPort),
        local_bind_address=('127.0.0.1', bindPort)
    )

    server.start()
    return server