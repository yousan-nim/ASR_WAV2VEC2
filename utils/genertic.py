
import torch
import numpy as np
import importlib.util
import importlib.metadata
from enum import Enum

from collections import OrderedDict, UserDict
from typing import Any, Tuple, Union



def _is_package_available(pkg_name: str, return_version: bool = False) -> Union[Tuple[bool, str], bool]: 
    package_exists = importlib.util.find_spec(pkg_name) is not None
    package_version = "N/A"
    if package_exists:
        try:
            package_version = importlib.metadata.version(pkg_name)
            package_exists = True
        except importlib.metadata.PackageNotFoundError:
            package_exists = False

    if return_version:
        return package_exists, package_version
    else:
        return package_exists


_torch_available, _torch_version = _is_package_available("torch", return_version=True)


def _is_torch(x):
    return isinstance(x, torch.Tensor)

def is_torch_available():
    return _torch_available

def is_torch_tensor(x):
    return False if not is_torch_available() else _is_torch(x)

def to_numpy(obj):
    if isinstance(obj, (dict, UserDict)):
        return {k: to_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return np.array(obj)
    elif is_torch_tensor(obj):
        return obj.detach().cpu().numpy()
    else:
        return obj
    
class ExplicitEnum(Enum):
    """
    Enum with more explicit error message for missing values.
    """

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not a valid {cls.__name__}, please select one of {list(cls._value2member_map_.keys())}"
        )


class PaddingStrategy(ExplicitEnum): 
    LONGES = "longest"
    MAX_LENGTH = "max_length"
    DO_NOT_PAD = "do_not_pad"

class TensorType(ExplicitEnum): 
    PYTORCH = "pt"
    NUMPY = "np"
    