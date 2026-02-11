from energizer.neural_network import Module


class ModuleList(Module):
    """Holds submodules in a list.

    ``ModuleList`` can be used like a regular Python list, but modules it
    contains are properly registered so that all ``Module`` methods
    (``parameters()``, ``to()``, ``state_dict()``, …) see them.

    Unlike ``Sequential``, ``ModuleList`` does **not** define a ``forward()``
    method — the user decides how to iterate over the layers.

    Args:
        modules (iterable, optional): An iterable of modules to add.

    Example::

        import energizer

        class MyModel(energizer.Module):
            def __init__(self):
                super().__init__()
                self.linears = energizer.ModuleList(
                    [energizer.Linear(10, 10) for _ in range(5)]
                )

            def forward(self, x):
                for layer in self.linears:
                    x = layer(x)
                return x
    """

    def __init__(self, modules=None):
        super().__init__()
        self._module_list = []
        if modules is not None:
            self.extend(modules)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reindex(self):
        """Re-register every module so that ``_modules`` keys stay in sync
        with list indices (required for ``state_dict`` / ``parameters``)."""
        self._modules.clear()
        for idx, module in enumerate(self._module_list):
            self.add_module(str(idx), module)

    @staticmethod
    def _check_module(module):
        if not isinstance(module, Module):
            raise TypeError(
                f"ModuleList only accepts Module instances, got {type(module).__name__}"
            )

    # ------------------------------------------------------------------
    # List-like interface
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self._module_list)

    def __iter__(self):
        return iter(self._module_list)

    def __contains__(self, module):
        return module in self._module_list

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return ModuleList(self._module_list[idx])
        return self._module_list[idx]

    def __setitem__(self, idx, module):
        self._check_module(module)
        self._module_list[idx] = module
        self._reindex()

    def __delitem__(self, idx):
        del self._module_list[idx]
        self._reindex()

    def __iadd__(self, modules):
        self.extend(modules)
        return self

    def __repr__(self):
        lines = [f"({i}): {repr(m)}" for i, m in enumerate(self._module_list)]
        if not lines:
            return "ModuleList()"
        inner = "\n".join("  " + line for line in lines)
        return f"ModuleList(\n{inner}\n)"

    # ------------------------------------------------------------------
    # Mutation methods
    # ------------------------------------------------------------------

    def append(self, module):
        """Append a module to the end of the list.

        Args:
            module (Module): Module to append.

        Returns:
            ModuleList: ``self``, for chaining.
        """
        self._check_module(module)
        idx = len(self._module_list)
        self._module_list.append(module)
        self.add_module(str(idx), module)
        return self

    def extend(self, modules):
        """Append modules from an iterable to the end of the list.

        Args:
            modules (iterable[Module]): Modules to append.

        Returns:
            ModuleList: ``self``, for chaining.
        """
        for module in modules:
            self.append(module)
        return self

    def insert(self, idx, module):
        """Insert a module at a given index.

        Args:
            idx (int): Index before which to insert.
            module (Module): Module to insert.
        """
        self._check_module(module)
        self._module_list.insert(idx, module)
        self._reindex()

    def pop(self, idx=-1):
        """Remove and return the module at the given index (default: last).

        Args:
            idx (int): Index of the module to remove.

        Returns:
            Module: The removed module.
        """
        module = self._module_list.pop(idx)
        self._reindex()
        return module

    # ------------------------------------------------------------------
    # Device transfer
    # ------------------------------------------------------------------

    def to(self, device):
        """Move every contained module to *device*.

        Args:
            device (str): ``'cpu'`` or ``'gpu'``.

        Returns:
            ModuleList: ``self``.
        """
        for module in self._module_list:
            module.to(device)
        self.device = device
        return self
