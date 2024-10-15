import logging
import threading
import time
import zmq

from torch import Tensor, device, cuda
from tornado import ioloop
from zmq.eventloop import zmqstream

from .heartbeat import HeartBeater
from .payload import TensorPayload

logger = logging.getLogger("tensorsocket")
logger.setLevel(logging.INFO)
LOCALHOST = "tcp://*"


def process_tensor(tensor):
    if isinstance(tensor, Tensor):
        if cuda.is_available():
            tensor = tensor.to(device="cuda")
        tensor = TensorPayload(tensor)
    return tensor


def pack(data: tuple):
    return tuple((process_tensor(t) for t in data))


class TensorProducer:
    def __init__(
        self,
        data_loader: object,
        port: int = 5555,
        ack_port: int = 5556,
        heart_ports: (int, int) = (4444, 4445),
        rubber_band_pct: int = 0.02,
        pack_fn=pack,
    ):
        """
        Data loader that sends inputs and labels over tcp to training processes (consumers).

        Args:
            data_loader (object): Data loader to wrap around. Should be iterable.
            port (int, optional): Data transmission port. Defaults to 5555.
            ack_port (int, optional): Acknowledgement port. Defaults to 5556.
            heart_ports (int, int, optional): Life pulse ports. Defaults to (4444, 4445).
            rubber_band_pct (int, optional): Maximum allowed distance between consumers, in percent of training dataset size. Defaults to 0.02.
        """
        self.port = port
        self.ack_port = ack_port
        self.heart_ports = heart_ports

        self.data_loader = data_loader
        self.data_loader_len = len(self.data_loader)
        try:
            self.data_loader_iter = iter(self.data_loader)
        except TypeError:
            print("TensorSocket: DataLoader is already iterable")
            self.data_loader_iter = self.data_loader

        self.pack_fn = pack_fn

        self.index = 0
        self.consumers = {}
        self.new_consumers = []
        self.context = zmq.Context()

        # Send batches via
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(f"{LOCALHOST}:{self.port}")

        # Ack
        self.ack_socket = self.context.socket(zmq.SUB)
        self.ack_socket.bind(f"{LOCALHOST}:{self.ack_port}")
        self.ack_socket.setsockopt_string(zmq.SUBSCRIBE, "")

        # Heartbeat monitor
        self.hb = None
        self.heartbeat_thread = threading.Thread(
            target=self._start_heartbeat, args=(), daemon=True
        )
        self.heartbeat_thread.start()

        while self.hb is None:
            time.sleep(0.2)

        self._heartbeat_monitor_alive = True
        self.heartbeat_monitor_thread = threading.Thread(
            target=self._heartbeat_monitor, args=(), daemon=True
        )
        self.heartbeat_monitor_thread.start()

        # Dataset logic
        self.dataset_is_reset = True
        self.epoch = 0

        # Rubberbanding
        self.rb_buffer = list()
        self.rb_max_len = rubber_band_pct * self.data_loader_len
        self.rb_running = False
        self.buffer_idx = (
            0  # the current batch in the buffer we are sending to the consumers
        )

    def _start_heartbeat(self):
        self.loop = ioloop.IOLoop()
        context = zmq.Context()
        pub = context.socket(zmq.PUB)
        pub.bind(f"{LOCALHOST}:{self.heart_ports[0]}")
        router = context.socket(zmq.ROUTER)
        router.bind(f"{LOCALHOST}:{self.heart_ports[1]}")

        outstream = zmqstream.ZMQStream(pub, self.loop)
        instream = zmqstream.ZMQStream(router, self.loop)

        self.hb = HeartBeater(self.loop, outstream, instream)
        self.loop.start()

    def _heartbeat_monitor(self):
        while True:
            if len(self.hb.hearts) != len(self.consumers) + len(self.new_consumers):
                if len(self.hb.hearts) > len(self.consumers) + len(self.new_consumers):
                    self._set_consumer_len()
                    self.new_consumers += [(-1, 0, 0)] # TODO: change
                # self._set_consumer_count(len(self.hb.hearts))
            if not self._heartbeat_monitor_alive:
                break
            time.sleep(1)

    def join(self):
        self.loop.stop()
        self.heartbeat_thread.join()
        self._heartbeat_monitor_alive = False
        self.heartbeat_monitor_thread.join()

    # def _set_consumer_count(self, new_consumer_count):
    #     # self.consumer_count = new_consumer_count
    #     print("consumer count", self.hb.hearts)
    #     self.consumers = [[0, 0] for _ in self.hb.hearts]

    def __iter__(self):
        self.data_loader_iter = iter(self.data_loader)
        return self

    def __next__(self):
        # adjustment = len(self.consumers)
        # self.consumers += self.new_consumers
        if len(self.new_consumers):
            self.consumers[-1] = (0,0)
            self.new_consumers = []
        # adjustment = len(self.consumers) - adjustment
        # self.new_consumers = self.new_consumers[adjustment:]

        return self.get_sample()

    def get_sample(self):
        if self.index >= self.data_loader_len:
            self.index = 0
            self.epoch += 1
            self.rb_buffer = []
            raise StopIteration

        # idle when no consumers attached
        elif not len(self.consumers):
            logger.info("No consumers, waiting ...")
            time.sleep(0.5)
            return

        current_batch_index = self.index
        # if we are relatively early in the epoch, allow for new proc to catch up
        # also reset buffer index to handle if new consumers keep joining in
        if (
            (len(self.consumers) > 0)  # TODO: remove
            and (len(self.hb.hearts) > len(self.consumers))
            and (current_batch_index < self.rb_max_len)
        ):
            logger.info("Rubberbanding")
            self.rb_running = True
            self.buffer_idx = 0
        try:
            # add CPU tensors to rubberband buffer
            if not self.rb_running:
                data = next(self.data_loader_iter)
                self.rb_buffer.append((current_batch_index, data))

                # if buffer full, pop from end
                if len(self.rb_buffer) > self.rb_max_len:
                    _ = self.rb_buffer.pop(-1)

            else:
                current_batch_index, data = self.rb_buffer[self.buffer_idx]
                self.buffer_idx += 1

            while True:
                expected = {x: y for x, y in self.consumers.items()}
                self._broadcast(self.epoch, current_batch_index, data)
                ack_result = self._handle_acks(expected)
                if "FAILED" not in ack_result:
                    self.consumers = ack_result
                    break

            if not self.rb_running:
                self.index += 1
            elif len(self.rb_buffer) == self.buffer_idx:
                self.rb_running = False

            # first batch, return starting time. only relevant for dalle2
            if current_batch_index == 0:
                return time.time()
        except StopIteration:
            self.socket.send_pyobj({"stop_iteration": 1})
            raise StopIteration

    def _broadcast(
        self,
        current_epoch: int,
        current_batch_index: int,
        data: tuple,
    ):

        payload = dict(
            data=self.pack_fn(data),
            current_epoch=current_epoch,
            current_batch_index=current_batch_index,
        )

        if True:#current_batch_index % 100 == 0:
            logger.info(
                f"current_batch_index {current_batch_index}, "
                f"buffer size: {len(self.rb_buffer)}"
            )

        self.socket.send_pyobj(payload)

    def _handle_acks(self, expected: list):
        result = {}

        while True:
            # received all Acks, can go to next batch
            if not len(expected):
                return result

            if self.ack_socket.poll(10000, zmq.POLLIN):
                (
                    consumer_index,
                    batch_count,
                ) = (
                    self.ack_socket.recv_multipart()
                )  # wait for consumer acknowledgement

                batch_count = int(batch_count)

                logger.info(
                    f"Consumer: {consumer_index}, "
                    f"batch count: {batch_count}, "
                    f"total batches: {self.data_loader_len}"
                )

                if consumer_index in expected:
                    # if (0, batch_count) == expected[consumer_index]:
                    if True:#batch_count >= expected[consumer_index][1]: TODO: missing safeguard
                        print((0, batch_count), "FOUND in", expected, result)
                        expected.pop(consumer_index)
                        result[consumer_index] = (0, batch_count + 1)
                    else:
                        print(consumer_index, (0, batch_count), "not in", expected, result)
                else:
                    if -1 in expected:# and (0, batch_count) == (0,0): TODO: missing safeguard
                        print("Added new consumer ", consumer_index)
                        expected.pop(-1)
                        result[consumer_index] = (0, batch_count + 1)
                    else:
                        print(consumer_index, "consumer not in", expected, result)

            else:
                print(
                    "Timeout on Ack, assuming consumer is dead",
                    len(self.hb.hearts),
                    expected,
                    result,
                )
                return {"FAILED": 0}
                # self.consumer_count -= 1 TODO: fix

                # self._set_consumer_count(len(self.hb.hearts))
                # expected = min(expected, len(self.hb.hearts))
                # expected = {}

    def __len__(self):
        return self.data_loader_len

    def _set_consumer_len(self):  # TODO: change (name to send?)
        self.socket.send_pyobj({"data_loader_len": self.__len__()})
