import time
import torch
import gc
from tensorsocket.consumer import TensorConsumer

"""
A simple example on how to implement a data receiver (consumer) in your training script.
The TensorConsumer class directly replaces the data loader in the training script.
Please check out simple_producer.py for the paired producer script.
"""

consumer = TensorConsumer("5556", "5557", batch_size=16)
for i, batch in enumerate(consumer):
    b, (inputs, labels) = batch
    if labels != None:
        if True:
            # print(f"I:{i:0>7} -", b, labels[0], consumer.epoch, len(labels))
            # time.sleep(0.1)
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
                    # obj.grad = None
                    # obj.storage().resize_(0)
                    obj._fix_weakref()
                    del obj
            except:
                pass
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        print("Tensors:", c)
    else:
        print("Waiting ...")

print("Finished")
