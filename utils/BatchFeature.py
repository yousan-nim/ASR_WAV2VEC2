
import torch 
import numpy as np

from typing import List, Optional, Union, Dict, Any, Sequence, NamedTuple
from collections import UserDict
from dataclasses import dataclass, field
from .genertic import TensorType
from .utils import _is_numpy, torch_required

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




################# Utils #################

try: 
    from tokenizers import AddedToken
    from tokenizers import Encoding as EncodingFast
except:
    @dataclass(frozen=True, eq=True)
    class AddedToken:
            content: str = field(default_factory=str)
            single_word: bool = False
            lstrip: bool = False
            rstrip: bool = False
            normalized: bool = True

            def __getstate__(self):
                return self.__dict__
    @dataclass(frozen=True, eq=True)
    class EncodingFast: 
        pass



class TokenSpan(NamedTuple):
    start: int
    end: int


class CharSpan(NamedTuple):
    start: int
    end: int


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
        

class BatchEncoding(UserDict): 
    def __init__(
        self,
        data: Optional[Dict[str, Any]] = None, 
        encoding: Optional[Union[EncodingFast, Sequence[EncodingFast]]] = None,
        tensor_type: Union[None, str, TensorType] = None, 
        prepend_batch_axis: bool = False,
        n_sequences: Optional[int] = None,
    ):
        super().__init__(data)

        if isinstance(encoding, EncodingFast):
            encoding = [encoding]

        self._encoding = encoding

        if n_sequences is None and encoding is not None and len(encoding):
            n_sequences = encoding[0].n_sequences

        self._n_sequence = n_sequences
        self.convert_to_tensors(tensor_type=tensor_type,
                                prepend_batch_axis=prepend_batch_axis) 

    def convert_to_tensors(
        self, 
        tensor_type: Optional[Union[str, TensorType]] = None,
        prepend_batch_axis: bool = False
    ):
        if tensor_type is None:
            return self

        # Convert to TensorType
        if not isinstance(tensor_type, TensorType):
            tensor_type = TensorType(tensor_type)
        
        if not is_torch_available():
            import torch
            as_tensor = torch.tensor
            is_tensor = torch.is_tensor
        else:
            as_tensor = np.asarray
            is_tensor = _is_numpy
        
        for key, value in self.items():
            try:
                if prepend_batch_axis:
                    value = [value]

                if not is_tensor(value):
                    print(value)
                    tensor = as_tensor(value)
                    self[key] = tensor
            except:  
                if key == "overflowing_tokens":
                    raise ValueError(
                        "Unable to create tensor returning overflowing tokens of different lengths. "
                        "Please see if a fast version of this tokenizer is available to have this feature available."
                    )
                raise ValueError(
                    "Unable to create tensor, you should probably activate truncation and/or padding "
                    "with 'padding=True' 'truncation=True' to have batched tensors with the same length."
                )

        return self

    @property
    def n_sequences(self) -> Optional[int]:
        return self._n_sequences
    
    @property
    def is_fast(self) -> bool:
        return self._encodings is not None
    
    def __getitem__(self, item: Union[int, str]) -> Union[Any, EncodingFast]:
        if isinstance(item, str):
            return self.data[item]
        elif self._encodings is not None:
            return self._encodings[item]
        else:
            raise KeyError(
                "Indexing with integers (to access backend Encoding for a given batch index) "
                "is not available when using Python based tokenizers"
            )

    def __getattr__(self, item: str):
        try:
            return self.data[item]
        except KeyError:
            raise AttributeError

    def __getstate__(self):
        return {"data": self.data, "encodings": self._encodings}

    def __setstate__(self, state):
        if "data" in state:
            self.data = state["data"]

        if "encodings" in state:
            self._encodings = state["encodings"]

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()
    
    @property
    def encodings(self) -> Optional[List[EncodingFast]]:
        return self._encodings
    
    def tokens(self, batch_index: int = 0) -> List[str]:
        if not self._encodings:
            raise ValueError("tokens() is not available when using Python-based tokenizers")
        return self._encodings[batch_index].tokens

    def sequence_ids(self, batch_index: int = 0) -> List[Optional[int]]:
        if not self._encodings:
            raise ValueError("sequence_ids() is not available when using Python-based tokenizers")
        return self._encodings[batch_index].sequence_ids

    def words(self, batch_index: int = 0) -> List[Optional[int]]:
        if not self._encodings:
            raise ValueError("words() is not available when using Python-based tokenizers")
        return self.word_ids(batch_index)

    def word_ids(self, batch_index: int = 0) -> List[Optional[int]]:
        if not self._encodings:
            raise ValueError("word_ids() is not available when using Python-based tokenizers")
        return self._encodings[batch_index].word_ids

    def token_to_sequence(self, batch_or_token_index: int, token_index: Optional[int] = None) -> int:
        if not self._encodings:
            raise ValueError("token_to_sequence() is not available when using Python based tokenizers")
        if token_index is not None:
            batch_index = batch_or_token_index
        else:
            batch_index = 0
            token_index = batch_or_token_index
        if batch_index < 0:
            batch_index = self._batch_size + batch_index
        if token_index < 0:
            token_index = self._seq_len + token_index
        return self._encodings[batch_index].token_to_sequence(token_index)

    def token_to_word(self, batch_or_token_index: int, token_index: Optional[int] = None) -> int:
        if not self._encodings:
            raise ValueError("token_to_word() is not available when using Python based tokenizers")
        if token_index is not None:
            batch_index = batch_or_token_index
        else:
            batch_index = 0
            token_index = batch_or_token_index
        if batch_index < 0:
            batch_index = self._batch_size + batch_index
        if token_index < 0:
            token_index = self._seq_len + token_index
        return self._encodings[batch_index].token_to_word(token_index)

    def word_to_tokens(
        self, batch_or_word_index: int, word_index: Optional[int] = None, sequence_index: int = 0
    ) -> Optional[TokenSpan]:
        if not self._encodings:
            raise ValueError("word_to_tokens() is not available when using Python based tokenizers")
        if word_index is not None:
            batch_index = batch_or_word_index
        else:
            batch_index = 0
            word_index = batch_or_word_index
        if batch_index < 0:
            batch_index = self._batch_size + batch_index
        if word_index < 0:
            word_index = self._seq_len + word_index
        span = self._encodings[batch_index].word_to_tokens(word_index, sequence_index)
        return TokenSpan(*span) if span is not None else None

    def token_to_chars(self, batch_or_token_index: int, token_index: Optional[int] = None) -> CharSpan:
        if not self._encodings:
            raise ValueError("token_to_chars() is not available when using Python based tokenizers")
        if token_index is not None:
            batch_index = batch_or_token_index
        else:
            batch_index = 0
            token_index = batch_or_token_index
        return CharSpan(*(self._encodings[batch_index].token_to_chars(token_index)))

    def char_to_token(
        self, batch_or_char_index: int, char_index: Optional[int] = None, sequence_index: int = 0
    ) -> int:
        if not self._encodings:
            raise ValueError("char_to_token() is not available when using Python based tokenizers")
        if char_index is not None:
            batch_index = batch_or_char_index
        else:
            batch_index = 0
            char_index = batch_or_char_index
        return self._encodings[batch_index].char_to_token(char_index, sequence_index)

    def word_to_chars(
        self, batch_or_word_index: int, word_index: Optional[int] = None, sequence_index: int = 0
    ) -> CharSpan:
        if not self._encodings:
            raise ValueError("word_to_chars() is not available when using Python based tokenizers")
        if word_index is not None:
            batch_index = batch_or_word_index
        else:
            batch_index = 0
            word_index = batch_or_word_index
        return CharSpan(*(self._encodings[batch_index].word_to_chars(word_index, sequence_index)))

    def char_to_word(self, batch_or_char_index: int, char_index: Optional[int] = None, sequence_index: int = 0) -> int:
        if not self._encodings:
            raise ValueError("char_to_word() is not available when using Python based tokenizers")
        if char_index is not None:
            batch_index = batch_or_char_index
        else:
            batch_index = 0
            char_index = batch_or_char_index
        return self._encodings[batch_index].char_to_word(char_index, sequence_index)

    def convert_to_tensors(
        self, tensor_type: Optional[Union[str, TensorType]] = None, prepend_batch_axis: bool = False
    ):
        if tensor_type is None:
            return self

        # Convert to TensorType
        if not isinstance(tensor_type, TensorType):
            tensor_type = TensorType(tensor_type)

        # Get a function reference for the correct framework
        if  tensor_type == TensorType.PYTORCH:
            if not is_torch_available():
                raise ImportError("Unable to convert output to PyTorch tensors format, PyTorch is not installed.")
            import torch

            as_tensor = torch.tensor
            is_tensor = torch.is_tensor
        else:
            as_tensor = np.asarray
            is_tensor = _is_numpy
        
        for key, value in self.items():
            try:
                if prepend_batch_axis:
                    value = [value]

                if not is_tensor(value):
                    tensor = as_tensor(value)
                    self[key] = tensor
            except:  # noqa E722
                if key == "overflowing_tokens":
                    raise ValueError(
                        "Unable to create tensor returning overflowing tokens of different lengths. "
                        "Please see if a fast version of this tokenizer is available to have this feature available."
                    )
                raise ValueError(
                    "Unable to create tensor, you should probably activate truncation and/or padding "
                    "with 'padding=True' 'truncation=True' to have batched tensors with the same length."
                )

        return self

    @torch_required
    def to(self, device: Union[str, "torch.device"]) -> "BatchEncoding":
        if isinstance(device, str) or _is_torch_device(device) or isinstance(device, int):
            self.data = {k: v.to(device=device) for k, v in self.data.items()}
        return self











