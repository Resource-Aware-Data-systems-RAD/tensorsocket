import torch
from torchvision import datasets, transforms

from tensorsocket.producer import TensorProducer

"""
A simple example on how to implement a data sender (producer) in your training script.
The TensorProducer class wraps around your data loader.
Please check out simple_consumer.py for the paired consumer script.
"""

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)
dataset1 = datasets.MNIST("../data", train=True, download=True, transform=transform)
dataset2 = datasets.MNIST("../data", train=False, transform=transform)

train_kwargs = {"batch_size": 8}
test_kwargs = {"batch_size": 8}

use_cuda = True
if use_cuda:
    cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)
train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

producer = TensorProducer(train_loader, "5556", "5557", rubber_band_pct=0.2)

for epoch in range(10):
    for i, _ in enumerate(producer):
        if not i % 100:
            pass
    # test_loader_iter = iter(test_loader)
    # while True:
    #     try:
    #         n = next(test_loader_iter)
    #         print(test_loader_iter._num_yielded)
    #     except StopIteration:
    #         break
    # # end of epoch

producer.join()
print("finished")
