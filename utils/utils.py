
import os
import sys
import importlib.util
import shutil
import tarfile
import numpy as np
from uuid import uuid4
from pathlib import Path
from filelock import FileLock
import unittest

from enum import Enum
from urllib.parse import urlparse
from zipfile import ZipFile, is_zipfile
from collections import OrderedDict, UserDict
from functools import partial, wraps

from utils import logging
from typing import Any, BinaryIO, Dict, List, Optional, Tuple, Union
from transformers.utils.versions import importlib_metadata


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}

torch_cache_home = os.getenv("TORCH_HOME", os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "torch"))
hf_cache_home = os.path.expanduser(os.getenv("HF_HOME", os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "huggingface"))
                                   )
old_default_cache_path = os.path.join(torch_cache_home, "transformers")
default_cache_path = os.path.join(hf_cache_home, "transformers")

if (
    os.path.isdir(old_default_cache_path)
    and not os.path.isdir(default_cache_path)
    and "PYTORCH_PRETRAINED_BERT_CACHE" not in os.environ
    and "PYTORCH_TRANSFORMERS_CACHE" not in os.environ
    and "TRANSFORMERS_CACHE" not in os.environ
):
    shutil.move(old_default_cache_path, default_cache_path)

PYTORCH_PRETRAINED_BERT_CACHE = os.getenv("PYTORCH_PRETRAINED_BERT_CACHE", default_cache_path)
PYTORCH_TRANSFORMERS_CACHE = os.getenv("PYTORCH_TRANSFORMERS_CACHE", PYTORCH_PRETRAINED_BERT_CACHE)
TRANSFORMERS_CACHE = os.getenv("TRANSFORMERS_CACHE", PYTORCH_TRANSFORMERS_CACHE)
SESSION_ID = uuid4().hex
DISABLE_TELEMETRY = os.getenv("DISABLE_TELEMETRY", False) in ENV_VARS_TRUE_VALUES

WEIGHTS_NAME = "pytorch_model.bin"
TF2_WEIGHTS_NAME = "tf_model.h5"
TF_WEIGHTS_NAME = "model.ckpt"
FLAX_WEIGHTS_NAME = "flax_model.msgpack"
CONFIG_NAME = "config.json"
FEATURE_EXTRACTOR_NAME = "preprocessor_config.json"
MODEL_CARD_NAME = "modelcard.json"


WAV_2_VEC_2_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/wav2vec2-base-960h",
    "facebook/wav2vec2-large-960h",
    "facebook/wav2vec2-large-960h-lv60",
    "facebook/wav2vec2-large-960h-lv60-self",
    # See all Wav2Vec2 models at https://huggingface.co/models?filter=wav2vec2
]

_is_offline_mode = True if os.environ.get("TRANSFORMERS_OFFLINE", "0").upper() in ENV_VARS_TRUE_VALUES else False

def is_offline_mode():
    return _is_offline_mode

def add_end_docstrings(*docstr):
    def docstring_decorator(fn):
        fn.__doc__ = fn.__doc__ + "".join(docstr)
        return fn
    return docstring_decorator

def is_remote_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")

def is_torch_available():
    _torch_available = importlib.util.find_spec("torch") is not None
    return _torch_available

def is_vision_available():
    return importlib.util.find_spec("PIL") is not None

def _is_torch(x):
    import torch
    return isinstance(x, torch.Tensor)

def _is_numpy(x):
    return isinstance(x, np.ndarray)

def torch_required(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if is_torch_available():
            return func(*args, **kwargs)
        else:
            raise ImportError(f"Method `{func.__name__}` requires PyTorch.")
    return wrapper

def to_py_obj(obj):
    if isinstance(obj, (dict, UserDict)):
        return {k: to_py_obj(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_py_obj(o) for o in obj]
    elif is_torch_available() and _is_torch(obj):
        return obj.detach().cpu().tolist()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj
    
def is_tensor(x):
    """
    Tests if ``x`` is a :obj:`torch.Tensor`, :obj:`tf.Tensor`, obj:`jaxlib.xla_extension.DeviceArray` or
    :obj:`np.ndarray`.
    """
    if is_torch_available():
        import torch

        if isinstance(x, torch.Tensor):
            return True

    return isinstance(x, np.ndarray)

def require_torch(test_case):
    """
    Decorator marking a test that requires PyTorch.

    These tests are skipped when PyTorch isn't installed.

    """
    return unittest.skipUnless(is_torch_available(), "test requires PyTorch")(test_case)


def slow(test_case):
    """
    Decorator marking a test as slow.

    Slow tests are skipped by default. Set the RUN_SLOW environment variable to a truthy value to run them.

    """
    return unittest.skipUnless(_run_slow_tests, "test is slow")(test_case)

class ExplicitEnum(Enum):
    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not a valid {cls.__name__}, please select one of {list(cls._value2member_map_.keys())}"
        )

class PaddingStrategy(ExplicitEnum):
    LONGEST = "longest"
    MAX_LENGTH = "max_length"
    DO_NOT_PAD = "do_not_pad"


class TensorType(ExplicitEnum):
    PYTORCH = "pt"
    TENSORFLOW = "tf"
    NUMPY = "np"
    JAX = "jax"

class TensorType(ExplicitEnum):
    PYTORCH = "pt"
    TENSORFLOW = "tf"
    NUMPY = "np"
    JAX = "jax"

class TruncationStrategy(ExplicitEnum):
    ONLY_FIRST = "only_first"
    ONLY_SECOND = "only_second"
    LONGEST_FIRST = "longest_first"
    DO_NOT_TRUNCATE = "do_not_truncate"



def get_list_of_files(
    path_or_repo: Union[str, os.PathLike],
    revision: Optional[str] = None,
    use_auth_token: Optional[Union[bool, str]] = None,
) -> List[str]:
    path_or_repo = str(path_or_repo)
    # If path_or_repo is a folder, we just return what is inside (subdirectories included).
    if os.path.isdir(path_or_repo):
        list_of_files = []
        for path, dir_names, file_names in os.walk(path_or_repo):
            list_of_files.extend([os.path.join(path, f) for f in file_names])
        return list_of_files
    
    if is_offline_mode():
        return []



def cached_path(
    url_or_filename,
    cache_dir=None,
    force_download=False,
    proxies=None,
    resume_download=False,
    user_agent: Union[Dict, str, None] = None,
    extract_compressed_file=False,
    force_extract=False,
    use_auth_token: Union[bool, str, None] = None,
    local_files_only=False,
) -> Optional[str]:
    if cache_dir is None:
        cache_dir = TRANSFORMERS_CACHE
    if isinstance(url_or_filename, Path):
        url_or_filename = str(url_or_filename)
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    if is_offline_mode() and not local_files_only:
        logger.info("Offline mode: forcing local_files_only=True")
        local_files_only = True

    if os.path.exists(url_or_filename):
        # File, and it exists.
        output_path = url_or_filename
    elif urlparse(url_or_filename).scheme == "":
        # File, but it doesn't exist.
        raise EnvironmentError(f"file {url_or_filename} not found")
    else:
        # Something unknown
        raise ValueError(f"unable to parse {url_or_filename} as a URL or as a local path")

    if extract_compressed_file:
        if not is_zipfile(output_path) and not tarfile.is_tarfile(output_path):
            return output_path

        # Path where we extract compressed archives
        # We avoid '.' in dir name and add "-extracted" at the end: "./model.zip" => "./model-zip-extracted/"
        output_dir, output_file = os.path.split(output_path)
        output_extract_dir_name = output_file.replace(".", "-") + "-extracted"
        output_path_extracted = os.path.join(output_dir, output_extract_dir_name)

        if os.path.isdir(output_path_extracted) and os.listdir(output_path_extracted) and not force_extract:
            return output_path_extracted

        # Prevent parallel extractions
        lock_path = output_path + ".lock"
        with FileLock(lock_path):
            shutil.rmtree(output_path_extracted, ignore_errors=True)
            os.makedirs(output_path_extracted)
            if is_zipfile(output_path):
                with ZipFile(output_path, "r") as zip_file:
                    zip_file.extractall(output_path_extracted)
                    zip_file.close()
            elif tarfile.is_tarfile(output_path):
                tar_file = tarfile.open(output_path)
                tar_file.extractall(output_path_extracted)
                tar_file.close()
            else:
                raise EnvironmentError(f"Archive format of {output_path} could not be identified")

        return output_path_extracted

    return output_path

