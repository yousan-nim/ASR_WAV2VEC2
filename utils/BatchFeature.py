
import torch 
import numpy as np

from typing import List, Optional, Union, Dict, Any
from collections import UserDict

from .genertic import TensorType

################# Utils #################(start)
def is_numpy_array(x): 
    return isinstance(x, np.ndarray)

def is_torch_available():
    return False

def _is_torch_dtype(x):
    import torch

    if isinstance(x, str):
        if hasattr(torch, x):
            x = getattr(torch, x)
        else:
            return False
    return isinstance(x, torch.dtype)

def _is_torch_device(x):
    import torch

    return isinstance(x, torch.device)

def is_torch_dtype(x):
    """
    Tests if `x` is a torch dtype or not. Safe to call even if torch is not installed.
    """
    return False if not is_torch_available() else _is_torch_dtype(x)

def is_torch_device(x):
    """
    Tests if `x` is a torch device or not. Safe to call even if torch is not installed.
    """
    return False if not is_torch_available() else _is_torch_device(x)




################# Utils #################(end)


class BatchFeature(UserDict): 
    def __init__(
        self, 
        data: Optional[Dict[str, Any]] = None,
        tensor_type: Union[None, str, TensorType] = None,
    ): 
        super().__init__(data)
        self.convert_to_tensors(tensor_type=tensor_type)

    def convert_to_tensors(
            self, 
            tensor_type: Optional[Union[str, TensorType]] = None
    ):
        ################# Checking tensor types #################(start)

        if tensor_type is None:
            return self 
        
        if not isinstance(tensor_type, TensorType): 
            tensor_type = TensorType(tensor_type)
            print(tensor_type)

        elif tensor_type == TensorType.PYTORCH: 
            def as_tensor(value): 
                if isinstance(value, (list, tuple)) and len(value) > 0 and isinstance(value[0], np.ndarray): 
                    value = np.ndarray(value)
                return torch.tensor(value)
            is_tensor = torch.tensor(value)
        
        else: 
            def as_tensor(value, dtype=None):
                if isinstance(value, (list, tuple)) and isinstance(value[0], (list, tuple, np.ndarray)):
                    value_lens = [len(val) for val in value]
                    if len(set(value_lens)) > 1 and dtype is None:
                        value = as_tensor([np.asarray(val) for val in value], dtype==object)
                return np.asarray(value, dtype=dtype)
                
            is_tensor = is_numpy_array

        ################# Checking tensor types #################(end)

        for key, value in self.items(): 
            try: 
                if not is_tensor(value): 
                    tensor = as_tensor(value)

                    self[key] = tensor 
            except: 
                if key == 'overflowing_values': 
                    raise ValueError("Unable to create tensor returning overflowing values of diferen length")
                raise("Unable to create Tensor, you should activate padding with 'padding=True to have batched tensors with same length.")
            
            return self
        
        def to(self, *args, **kwargs) -> "BatchFeature": 

            new_data = {}
            device = kwargs.get("device")

            if device is None and len(args) > 0: 
                arg = args[0]

            if is_torch_dtype(arg):
                pass
            elif isinstance(arg, str) or is_torch_device(arg) or isinstance(arg, int):
                device = arg
            else:
                raise ValueError(f"Attempting to cast a BatchFeature to type {str(arg)}. This is not supported.")
            
            for k, v in self.items(): 
                if torch.is_floatinf_point(v): 
                    new_data[k] = v.to(*args, **kwargs)
                elif device is not None: 
                    new_data[k] = v.to(device=device)
                else: 
                    new_data[k] = v
            self.data = new_data
            return self