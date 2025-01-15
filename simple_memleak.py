import gc
import torch
import time

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
        return self

    def __next__(self):
        a, b = (
            self.id * torch.ones((self.batch_size, 1000, 2000)),
            self.id * torch.ones((self.batch_size,)),
        )

        self.id += 1
        return a, b


data_loader = DummyLoader()


for epoch in range(10):
    for i, (data, labels) in enumerate(data_loader):
        time.sleep(0.001)
        if not i % 100:
            pass

        data = data.to(device="cuda")
        labels = labels.to(device="cuda")

        f = torch.multiprocessing.reductions.reduce_tensor(data)
        # f = data.untyped_storage()._share_cuda_()
        del f

        data.detach()
        labels.detach()
        del data
        del labels

        # Clear cache and collect garbage
        torch.cuda.empty_cache()
        gc.collect()

print("finished")
