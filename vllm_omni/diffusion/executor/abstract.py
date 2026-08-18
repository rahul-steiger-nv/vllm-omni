from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from vllm.utils.import_utils import resolve_obj_by_qualname

from vllm_omni.diffusion.data import OmniDiffusionConfig

if TYPE_CHECKING:
    from vllm_omni.diffusion.sched.interface import DiffusionSchedulerOutput
    from vllm_omni.diffusion.worker.utils import BaseRunnerOutput


class DiffusionExecutor(ABC):
    """Abstract base class for Diffusion executors."""

    uses_multiproc: bool = False

    @staticmethod
    def get_class(od_config: OmniDiffusionConfig) -> type[DiffusionExecutor]:
        executor_class: type[DiffusionExecutor]
        distributed_executor_backend = od_config.distributed_executor_backend
        # Mirror vLLM's `world_size == 1 -> "uni"` default
        # (vllm/config/parallel.py). A single-GPU diffusion deployment has
        # nothing to distribute: spawning a worker only adds MessageQueues,
        # /dev/shm output segments, and a second model load. Explicit "mp"
        # keeps process isolation and RPC timeouts. Multi-GPU still defaults
        # to "mp".
        if distributed_executor_backend is None:
            num_gpus = od_config.num_gpus or 1
            distributed_executor_backend = "uni" if num_gpus == 1 else "mp"

        if isinstance(distributed_executor_backend, type):
            if not issubclass(distributed_executor_backend, DiffusionExecutor):
                raise TypeError(
                    "distributed_executor_backend must be a subclass of "
                    f"DiffusionExecutor. Got {distributed_executor_backend}."
                )
            executor_class = distributed_executor_backend
        elif distributed_executor_backend == "ray":
            raise NotImplementedError("ray backend is not yet supported.")
        elif distributed_executor_backend == "mp":
            from vllm_omni.diffusion.executor.multiproc_executor import MultiprocDiffusionExecutor

            executor_class = MultiprocDiffusionExecutor
        elif distributed_executor_backend == "uni":
            from vllm_omni.diffusion.executor.uniproc_executor import UniProcDiffusionExecutor

            executor_class = UniProcDiffusionExecutor
        elif distributed_executor_backend == "external_launcher":
            raise NotImplementedError("external_launcher backend is not yet supported.")
        elif isinstance(distributed_executor_backend, str):
            try:
                executor_class = resolve_obj_by_qualname(distributed_executor_backend)
            except (ImportError, ValueError) as e:
                raise ValueError(
                    f"Failed to load executor backend '{distributed_executor_backend}'. "
                    f"Ensure it is a valid python path. Error: {e}"
                ) from e

            if not issubclass(executor_class, DiffusionExecutor):
                raise TypeError(
                    f"distributed_executor_backend must be a subclass of DiffusionExecutor. Got {executor_class}."
                )
        else:
            raise ValueError(f"Unknown distributed executor backend: {distributed_executor_backend}")
        return executor_class

    def __init__(self, od_config: OmniDiffusionConfig) -> None:
        self.od_config = od_config
        self._init_executor()

    @abstractmethod
    def _init_executor(self) -> None:
        """Initialize the executor (e.g., launch workers, setup IPC)."""
        pass

    @property
    @abstractmethod
    def is_dead(self) -> bool:
        """Whether the executor is shut down or has failed fatally."""
        pass

    @abstractmethod
    def execute_request(self, scheduler_output: DiffusionSchedulerOutput) -> BaseRunnerOutput:
        """Execute request-mode work from a scheduler output."""
        pass

    @abstractmethod
    def execute_batch(self, scheduler_output: DiffusionSchedulerOutput) -> BaseRunnerOutput:
        """Execute request-mode work through the request-batch path."""
        pass

    @abstractmethod
    def execute_step(self, scheduler_output: DiffusionSchedulerOutput) -> BaseRunnerOutput:
        """Execute step-mode work from a scheduler output."""
        pass

    @abstractmethod
    def collective_rpc(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        unique_reply_rank: int | None = None,
        exec_all_ranks: bool = False,
    ) -> Any:
        """Execute a method on workers."""
        pass

    @abstractmethod
    def check_health(self) -> None:
        """Check if the executor and workers are healthy."""
        pass

    def register_failure_callback(self, callback: Callable[[], None]) -> None:
        """Register a callback invoked when the executor fatally fails.

        Executors without a background failure monitor can keep the default
        no-op implementation.
        """
        return None

    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the executor and release resources."""
        pass
