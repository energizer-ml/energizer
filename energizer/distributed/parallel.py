from __future__ import annotations

from energizer.neural_network import Module


class DataParallel(Module):
    def __init__(self, module: Module, world):
        super().__init__(device=getattr(module, "device", "cpu"))
        self.module = module
        self.world = world
        self._mark_parameters()
        self._sync_module_state()

    def _mark_parameters(self) -> None:
        for param in self.module.parameters():
            param._distributed_world = self.world

    def _sync_module_state(self) -> None:
        state = self.module.state_dict() if self.world.rank == 0 else None
        synced_state = self.world.broadcast_object(state, src=0)
        if self.world.rank != 0:
            self.module.load_state_dict(synced_state)
        self.world._emit(
            "model_state_sync",
            {"parameters": len(self.module.parameters())},
        )

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def parameters(self):
        return self.module.parameters()

    def to(self, device: str):
        self.module.to(device)
        self.device = device
        self._mark_parameters()
        return self

    def __getattr__(self, name):
        if name in {"module", "world"}:
            return super().__getattribute__(name)
        return getattr(self.module, name)
