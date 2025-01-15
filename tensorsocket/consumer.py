import logging
import sys
import threading
import uuid
from queue import Queue
from typing import Tuple, Any, Iterator

import zmq
from zmq import devices

from .payload import TensorPayload
from .heartbeat import Heart

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.DEBUG,
    stream=sys.stdout,
)

logger = logging.getLogger("tensorsocket")
logger.setLevel(logging.WARNING)
LOCALHOST = "tcp://localhost"


def unpack(data: tuple) -> tuple:
    """Convert TensorPayload objects back to tensors.

    Args:
        data: Tuple containing possible TensorPayload objects

    Returns:
        Tuple with reconstructed tensors
    """
    return tuple((t.tensor if isinstance(t, TensorPayload) else t for t in data))


class TensorConsumer:
    """Receives and processes tensor batches from remote producer.

    Handles:
    - Connection to producer
    - Batch receiving and unpacking
    - Progress tracking
    - Heartbeat monitoring
    """

    def __init__(
        self,
        port: int = 5555,
        ack_port: int = 5556,
        heart_ports: tuple[int, int] = (4444, 4445),
        unpack_fn=unpack,
        batch_size: int = 64,
    ) -> None:
        """Initialize consumer connection.

        Args:
            port: Data reception port
            ack_port: Acknowledgement sending port
            heart_ports: (in, out) ports for heartbeat
            unpack_fn: Function to reconstruct tensors
        """
        self.unpack_fn = unpack_fn
        self.batch_size = batch_size

        self.port = port
        self.ack_port = ack_port
        self.heart_ports = heart_ports

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(f"{LOCALHOST}:{self.port}")
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.consumer_id = uuid.uuid4()

        # Ack
        self.ack_socket = self.context.socket(zmq.PUB)
        self.ack_socket.connect(f"{LOCALHOST}:{self.ack_port}")

        # Logic
        self.batch_count = 0
        self.batch_max = -1
        self.epoch = 0
        # self.receiving_epoch = 0

        # Heartbeat
        self.heart = Heart(
            self,
            self.consumer_id,
            self.batch_size,
            f"{LOCALHOST}:{self.heart_ports[0]}",
            f"{LOCALHOST}:{self.heart_ports[1]}",
        )
        self.heart.daemon = True
        self.heart.start()

        # On spawn, fetch payloads on socket until we get one with the data loader length
        while True:
            data = self.socket.recv_pyobj()
            if data.get("data_loader_len"):
                self.data_loader_len = data.get("data_loader_len")
                self.max_buffer_size = data.get("max_buffer_size")
                break

        # Buffer setup
        self.buffer = Queue(maxsize=self.max_buffer_size)
        self.fetch_thread = threading.Thread(target=self._fetch_loop, daemon=True)
        self.fetch_thread.start()

    def _fetch_loop(self) -> None:
        """Background thread for receiving batches.

        Continuously:
        1. Receives tensor data
        2. Handles special messages (length, stop)
        3. Processes regular batches
        4. Sends acknowledgements
        """
        while True:
            cuda_tensor_info = self.socket.recv_pyobj()
            # print(cuda_tensor_info)

            if "data_loader_len" in cuda_tensor_info:
                continue

            if "stop_iteration" in cuda_tensor_info:
                self.buffer.put(cuda_tensor_info)
                continue

            if f"{self.batch_size}" in cuda_tensor_info:  # Flexible
                messages = cuda_tensor_info[f"{self.batch_size}"]
            elif "-1" in cuda_tensor_info and len(cuda_tensor_info) == 1:  # Static
                messages = cuda_tensor_info["-1"]
            else:  # Ignore
                messages = []

            # messages = cuda_tensor_info["-1"]

            received_new = False

            # # ignore and bounce back as others are banding
            # if messages[0]["current_batch_index"] < self.batch_max + 1:
            #     self.ack_socket.send_multipart(
            #         [
            #             bytes(str(self.consumer_id).encode("utf-8")),
            #             bytes(str(self.batch_max).encode("utf-8")),
            #             b"0",
            #         ]
            #     )

            # # dont accept batch if it is from the future
            # elif messages[0]["current_batch_index"] > self.batch_max + 1:
            #     self.ack_socket.send_multipart(
            #         [
            #             bytes(str(self.consumer_id).encode("utf-8")),
            #             bytes(str(self.batch_max).encode("utf-8")),
            #             b"0",
            #         ]
            #     )

            for message in messages:
                if message["current_batch_index"] == self.batch_max + 1:
                    # t = self.unpack_fn(payload["data"])
                    # print(
                    #     f"received-{cuda_tensor_info['current_batch_index']}-{self.buffer.qsize()}-{t[1][0]}"
                    # )
                    self.buffer.put(message)
                    self.batch_max = message["current_batch_index"]
                    received_new = True

            if received_new:
                self.ack_socket.send_multipart(
                    [
                        bytes(str(self.consumer_id).encode("utf-8")),
                        bytes(str(self.batch_max).encode("utf-8")),
                        b"1",
                    ]
                )
            else:
                self.ack_socket.send_multipart(
                    [
                        bytes(str(self.consumer_id).encode("utf-8")),
                        bytes(str(self.batch_max).encode("utf-8")),
                        b"0",
                    ]
                )

    def __iter__(self) -> Iterator:
        """Make consumer iterable for batch processing."""
        return self

    def __len__(self) -> int:
        """Get total number of batches in dataset."""
        return self.data_loader_len

    def __next__(self) -> Tuple[int, Any]:
        """Get next batch from buffer.

        Returns:
            Tuple of (batch_index, tensor_data)

        Raises:
            StopIteration: At end of epoch
        """
        while True:
            payload = self.buffer.get()  # This will block if buffer is empty

            # if payload.get("stop_iteration"):
            if "stop_iteration" in payload:
                self.epoch += 1  # payload["stop_iteration"] TODO: do it like implemented rn to make epochs client-side
                self.batch_count = 0
                self.batch_max = -1
                # self.buffer = Queue(maxsize=self.max_buffer_size)
                # print("reset")
                continue
                # raise StopIteration

            current_epoch = payload["current_epoch"]
            batch_idx = payload["current_batch_index"]

            # batch = self.unpack_fn(payload["data"])

            # if current_epoch != self.epoch:  # TODO: make epoch count flexible
            #     self.epoch = current_epoch
            #     self.batch_count = 0

            if batch_idx == self.batch_count:
                logger.info(
                    f"Epoch: {self.epoch}, batch_idx: {batch_idx}, batch count: {self.batch_count}"
                )
                self.batch_count += 1
                return batch_idx, (0, 0)
