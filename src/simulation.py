import time
from multiprocessing import Process
from src.flower.client import start_client
from src.flower.server import start_server

from config import settings as st


def run_simulation():
    """Start a FL simulation."""

    # This will hold all the processes which we are going to create
    processes = []

    # Start the server
    server_process = Process(target=start_server, args=())
    server_process.start()
    processes.append(server_process)

    # Optionally block the script here for a second or two so the server has time to start
    time.sleep(2)

    # Start all the clients
    for cl in range(st.NUM_CLIENTS):
        client_id = str(cl)
        client_process = Process(target=start_client, args=(client_id))
        client_process.start()
        print(f"Started {cl}")
        processes.append(client_process)

    # Block until all processes are finished
    for p in processes:
        p.join()


if __name__ == "__main__":
    run_simulation()
