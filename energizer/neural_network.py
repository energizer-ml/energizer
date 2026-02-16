
from ast import Dict, List
from typing import Callable, Any

from energizer.tensor import Tensor
import numpy as np
try:
    import mlx.core as mx
except ImportError:
    mx = None


class Module:
    def __init__(self, device: str = 'cpu'):
        self._parameters = {}
        self._modules = {}
        self.device = device
        self.training = False

    def forward(self, x):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self._parameters[name] = value
        if isinstance(value, Module):
            self._modules[name] = value
        self.__dict__[name] = value

    def backward(self, x):
        raise NotImplementedError

    def train(self, mode: bool = True):
        self.training = mode
        for module in self._parameters:
            module.train(mode)
        return self

    def eval(self):
        return self.train(False)
    
    def children(self):
        for name, module in self._modules.items():
            yield module
    
    def modules(self):
        yield self
        for child in self.children():
            yield from child.modules()

    def add_parameter(self, name, param):
        self._parameters[name] = param
    
    def add_module(self, name, module):
        self._modules[name] = module

    def parameters(self):
        params = []
        for param in self._parameters.values():
            if param.requires_grad:
                params.append(param)

        for module in self._modules.values():
            params.extend(module.parameters())

        return params

    def reset_parameters(self):
        for param in self._parameters.values():
            if param.requires_grad:
                param.data.zero_()
        for module in self._modules.values():
            module.reset_parameters()

    def state_dict(self):
        state = {}

        def _to_numpy(arr):
            # Ensure arrays are serializable via numpy (np.savez)
            if mx is not None and isinstance(arr, mx.array):
                return np.array(arr)
            if isinstance(arr, np.ndarray):
                return arr.copy()
            return np.array(arr)

        def collect_params(module, prefix=""):
            for name, param in module._parameters.items():
                if param is not None:
                    key = f"{prefix}{name}"
                    state[key] = _to_numpy(param.data)
            for name, submodule in module._modules.items():
                collect_params(submodule, f"{prefix}{name}.")

        collect_params(self)
        return state

    def load_state_dict(self, state_dict):
        def _from_numpy(arr_np, device: str):
            # Restore to the target device
            if device == "gpu":
                if mx is None:
                    raise RuntimeError("Cannot load GPU weights because mlx is not available")
                return mx.array(arr_np)
            return np.array(arr_np)

        def load_params(module, prefix=""):
            for name, param in module._parameters.items():
                if param is not None:
                    key = f"{prefix}{name}"
                    if key in state_dict:
                        param.data = _from_numpy(state_dict[key], getattr(param, "device", "cpu"))
            
            for name, submodule in module._modules.items():
                load_params(submodule, f"{prefix}{name}.")
        
        load_params(self)

    def save(self, filepath):
        import numpy as np
        state_dict = self.state_dict()
        np.savez(filepath, **state_dict)

    def to(self, device: str):
        for name, param in self._parameters.items():
            if param is not None:
                new_param = param.to(device)
                param.data = new_param.data
                param.device = device
        for module in self._modules.values():
            module.to(device)
        self.device = device
        return self

    @classmethod
    def load(cls, filepath):
        import numpy as np
        model = cls()
        data = np.load(filepath, allow_pickle=True)
        state_dict = {key: data[key] for key in data.keys()}
        
        model.load_state_dict(state_dict)
        print(f"Model loaded from {filepath}")
        
        return model

class Optimizer:
    def __init__(self, params, defaults: Dict[str, Any]):
        self.defaults = defaults
        self.param_groups = []
        self.state = {}
        self._hook_pre_hooks: List[Callable] = []
        self._hook_post_hooks: List[Callable] = []

        if isinstance(params, (list, tuple)) and len(params) > 0:
            if isinstance(params[0], dict) and 'params' in params[0]:
                for param_group in params:
                    merged_group = {**defaults, **param_group}
                    self.add_param_group(merged_group)
            else:
                for param in params:
                    self.add_param_group({**defaults, 'params': param})
        else:
            if isinstance(params, (list, tuple)) and len(params) == 0:
                self.add_param_group({**defaults, 'params': []})
            else:
                self.add_param_group({**defaults, 'params': params})

    def add_param_group(self, param_group: dict):
        self.param_groups.append(param_group)

    def load_state_dict(self, state_dict: dict):
        self.param_groups = state_dict['param_groups']
        self.state = state_dict['state']

    def register_load_state_dict_post_hook(self, hook: Callable, preprend: bool = False):
        if preprend:
            self.load_state_dict_post_hooks.insert(0, hook)
        else:
            self.load_state_dict_post_hooks.append(hook)

    def register_load_state_dict_pre_hook(self, hook: Callable, preprend: bool = False):
        if preprend:
            self.load_state_dict_pre_hooks.insert(0, hook)
        else:
            self.load_state_dict_pre_hooks.append(hook)

    def register_state_dict_post_hook(self, hook: Callable, preprend: bool = False):
        if preprend:
            self.state_dict_post_hooks.insert(0, hook)
        else:
            self.state_dict_post_hooks.append(hook)

    def register_state_dict_pre_hook(self, hook: Callable, preprend: bool = False):
        if preprend:
            self.state_dict_pre_hooks.insert(0, hook)
        else:
            self.state_dict_pre_hooks.append(hook)

    def register_step_post_hook(self, hook: Callable):
        self.step_post_hooks.append(hook)

    def register_step_pre_hook(self, hook: Callable):
        self.step_pre_hooks.append(hook)

    def state_dict(self) -> dict:
        return {
            'state': self.state,
            'param_groups': self.param_groups
        }

    def step(self, closure: Callable = None):
        if closure is not None:
            closure()
        else:
            for param_group in self.param_groups:
                for param in param_group['params']:
                    if param.grad is not None:
                        param.data = param.data - param.grad * param_group['lr']

    def zero_grad(self):
        for param_group in self.param_groups:
            for param in param_group['params']:
                # Gradients can be numpy arrays or Tensors. Simplest: set to None
                # and let backward() recreate them fresh.
                param.grad = None

class Parameter(Tensor):
    pass