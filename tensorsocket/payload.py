from torch import Tensor
import torch
from dataclasses import asdict, dataclass
from torch.multiprocessing.reductions import rebuild_tensor, rebuild_cuda_tensor
from typing import TypedDict, cast


def torch_dtype_to_str(dtype: torch.dtype) -> str:
    """Converts a PyTorch dtype to its string representation."""

    return str(dtype).replace("torch.", "")


def str_to_torch_dtype(dtype: str) -> torch.dtype:
    """Converts a string representation of a PyTorch dtype back to its corresponding dtype object."""

    return getattr(torch, dtype)


class SerializedCudaRebuildMetadata(TypedDict):
    """TypedDict representing a serializable version of CUDA tensor rebuild metadata."""

    dtype: str
    tensor_size: tuple[int, ...]
    tensor_stride: tuple[int, ...]
    tensor_offset: int
    storage_device: int
    storage_handle: str
    storage_size_bytes: int
    storage_offset_bytes: int
    requires_grad: bool
    ref_counter_handle: str
    ref_counter_offset: int
    event_handle: str
    event_sync_required: bool


@dataclass
class CudaRebuildMetadata:
    """Data class representing the metadata required to rebuild a CUDA tensor."""

    dtype: torch.dtype
    tensor_size: torch.Size
    tensor_stride: tuple[int, ...]
    tensor_offset: int
    storage_device: int
    storage_handle: bytes
    storage_size_bytes: int
    storage_offset_bytes: int
    requires_grad: bool
    ref_counter_handle: bytes
    ref_counter_offset: int
    event_handle: bytes
    event_sync_required: bool

    @classmethod
    def from_serialized_dict(
        cls, metadata: SerializedCudaRebuildMetadata
    ) -> "CudaRebuildMetadata":
        """Creates a `CudaRebuildMetadata` instance from a serialized dictionary."""

        return cls(
            dtype=str_to_torch_dtype(metadata["dtype"]),
            tensor_size=torch.Size(metadata["tensor_size"]),
            tensor_stride=tuple(metadata["tensor_stride"]),
            tensor_offset=metadata["tensor_offset"],
            storage_device=metadata["storage_device"],
            storage_handle=bytes.fromhex(metadata["storage_handle"]),
            storage_size_bytes=metadata["storage_size_bytes"],
            storage_offset_bytes=metadata["storage_offset_bytes"],
            requires_grad=metadata["requires_grad"],
            ref_counter_handle=bytes.fromhex(metadata["ref_counter_handle"]),
            ref_counter_offset=metadata["ref_counter_offset"],
            event_handle=bytes.fromhex(metadata["event_handle"]),
            event_sync_required=metadata["event_sync_required"],
        )

    def to_serialized_dict(self) -> SerializedCudaRebuildMetadata:
        """Converts this `CudaRebuildMetadata` instance into a serializable dictionary."""

        metadata = asdict(self)
        metadata["dtype"] = torch_dtype_to_str(self.dtype)
        metadata["tensor_size"] = tuple(self.tensor_size)
        metadata["storage_handle"] = self.storage_handle.hex()
        metadata["ref_counter_handle"] = self.ref_counter_handle.hex()
        metadata["event_handle"] = self.event_handle.hex()
        return cast(SerializedCudaRebuildMetadata, metadata)


class TensorPayload:
    """Handles serialization and transmission of PyTorch tensors.

    Provides mechanisms for both CUDA and CPU tensor transmission over ZMQ sockets
    by properly handling memory sharing and rebuilding.
    """

    def __init__(self, tensor: Tensor | tuple) -> None:
        """Initialize tensor payload for transmission.

        Args:
            tensor: Either a PyTorch tensor to transmit or a tuple containing
                   serialized tensor data for reconstruction
        """
        # self._tensor = 0
        self.payload = 0
        if isinstance(tensor, Tensor):
            self._from_tensor(tensor)
            # self._tensor = tensor
        # else:
        #     # self.payload = tensor
        #     del tensor

        # if "storage_cls" in self.payload:
        #     try:
        #         self._tensor = rebuild_cuda_tensor(Tensor, **self.payload)
        #     except RuntimeError as e:
        #         self._tensor = tensor
        # else:
        #     self._tensor = rebuild_tensor(
        #         tensor["cls"], tensor["storage"], tensor["metadata"]
        # )

    def _from_tensor(self, tensor: Tensor) -> dict:
        """Convert tensor to transmissible format.

        Args:
            tensor: PyTorch tensor to convert

        Returns:
            Dictionary containing tensor metadata and shared memory information
        """
        # storage = tensor.untyped_storage()
        storage = tensor._typed_storage()

        if storage.is_cuda:
            (
                device,
                handle,
                storage_size_bytes,
                storage_offset_bytes,
                ref_counter_handle,
                ref_counter_offset,
                event_handle,
                event_sync_required,
            ) = storage._share_cuda_()

            return
            # return CudaRebuildMetadata(
            #     dtype=tensor.dtype,
            #     tensor_size=tensor.size(),
            #     tensor_stride=tensor.stride(),
            #     tensor_offset=tensor.storage_offset(),
            #     storage_device=device,
            #     storage_handle=handle,
            #     storage_size_bytes=storage_size_bytes,
            #     storage_offset_bytes=storage_offset_bytes,
            #     requires_grad=tensor.requires_grad,
            #     ref_counter_handle=ref_counter_handle,
            #     ref_counter_offset=ref_counter_offset,
            #     event_handle=event_handle,
            #     event_sync_required=event_sync_required,
            # )

        raise Exception("break")

        # storage.share_memory_()
        # metadata = (
        #     tensor.storage_offset(),
        #     tensor.size(),
        #     tensor.stride(),
        #     tensor.requires_grad,
        # )
        # return {
        #     "storage": storage,
        #     "cls": type(storage),
        #     "metadata": metadata,
        # }

    def __reduce__(self) -> tuple:
        """Enable pickle serialization of payload.

        Returns:
            Tuple containing class and initialization arguments
        """
        return (
            self.__class__,
            (self.payload,),
        )

    @property
    def tensor(self) -> Tensor:
        """Get the reconstructed tensor.

        Returns:
            Original PyTorch tensor from payload data
        """
        return self._tensor
