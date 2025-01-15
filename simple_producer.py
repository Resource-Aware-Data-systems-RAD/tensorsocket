import torch
import time

from tensorsocket.payload import TensorPayload

from tensorsocket.producer import TensorProducer

"""
A simple example on how to implement a data sender (producer) in your training script.
The TensorProducer class wraps around your data loader.
Please check out simple_consumer.py for the paired consumer script.
"""
torch.multiprocessing.set_sharing_strategy("file_system")


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


producer = TensorProducer(data_loader, "5556", "5557", rubber_band_pct=0.002)
import torch
import gc

for epoch in range(10):
    for i, _ in enumerate(producer):
        time.sleep(0.001)
        if not i % 100:
            pass

        c = 0
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (
                    hasattr(obj, "data") and torch.is_tensor(obj.data)
                ):
                    # print(type(obj), obj.size())
                    c += 1
                    # obj.detach().cpu()
                    obj.detach()
                    obj.grad = None
                    # # obj.storage().resize_(0)
                    del obj
            except:
                pass
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        print("Tensors:", c)
producer.join()
print("finished")
