import torch
import time

from tensorsocket.producer import TensorProducer

"""
A simple example on how to implement a data sender (producer) in your training script.
The TensorProducer class wraps around your data loader.
Please check out simple_consumer.py for the paired consumer script.
"""


class DummyLoader:
    def __init__(self, length=10000):
        self.length = length
        self.id = 0
        self.batch_size = 8

    def __len__(self):
        return self.length

    def __iter__(self):
        self.id = 0
        return self

    def __next__(self):
        a, b = (
            self.id * torch.ones((self.batch_size, 1000, 2000)),
            self.id * torch.ones((self.batch_size,)),
        )

        self.id += 1
        return a, b


data_loader = DummyLoader()


producer = TensorProducer(data_loader, "5556", "5557", rubber_band_pct=0.02)

for epoch in range(10):
    print("Epoch:", epoch)
    for i, _ in enumerate(producer):
        time.sleep(0.001)
        if not i % 100:
            pass
producer.join()
print("Finished")
