import copy
import json
import os
import re
import warnings
import bisect
import sys
import itertools
import torch
import numpy as np
import requests

from itertools import groupby
from collections import OrderedDict, UserDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union, overload

from packaging import version

from tokenizers import AddedToken

from utils.BatchFeature import BatchEncoding, BatchEncoding
from utils.utils_tokenizer import _is_end_of_word, _is_start_of_word
from utils.utils import (
    PaddingStrategy, 
    TensorType, 
    TruncationStrategy, 
    add_end_docstrings, 
    is_offline_mode, 
    is_remote_url,
    get_list_of_files,
    to_py_obj,
    cached_path,
)

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "tokenizer_config_file": "tokenizer_config.json",
}
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "facebook/wav2vec2-base-960h": "https://huggingface.co/facebook/wav2vec2-base-960h/resolve/main/vocab.json",
    },
    "tokenizer_config_file": {
        "facebook/wav2vec2-base-960h": "https://huggingface.co/facebook/wav2vec2-base-960h/resolve/main/tokenizer_config.json",
    },
}
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {"facebook/wav2vec2-base-960h": sys.maxsize}
ENCODE_KWARGS_DOCSTRING = r"""
            add_special_tokens (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether or not to encode the sequences with the special tokens relative to their model.
            padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`False`):
                Activates and controls padding. Accepts the following values:

                * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a
                  single sequence if provided).
                * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
                  maximum acceptable input length for the model if that argument is not provided.
                * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
                  different lengths).
            truncation (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.TruncationStrategy`, `optional`, defaults to :obj:`False`):
                Activates and controls truncation. Accepts the following values:

                * :obj:`True` or :obj:`'longest_first'`: Truncate to a maximum length specified with the argument
                  :obj:`max_length` or to the maximum acceptable input length for the model if that argument is not
                  provided. This will truncate token by token, removing a token from the longest sequence in the pair
                  if a pair of sequences (or a batch of pairs) is provided.
                * :obj:`'only_first'`: Truncate to a maximum length specified with the argument :obj:`max_length` or to
                  the maximum acceptable input length for the model if that argument is not provided. This will only
                  truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
                * :obj:`'only_second'`: Truncate to a maximum length specified with the argument :obj:`max_length` or
                  to the maximum acceptable input length for the model if that argument is not provided. This will only
                  truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
                * :obj:`False` or :obj:`'do_not_truncate'` (default): No truncation (i.e., can output batch with
                  sequence lengths greater than the model maximum admissible input size).
            max_length (:obj:`int`, `optional`):
                Controls the maximum length to use by one of the truncation/padding parameters.

                If left unset or set to :obj:`None`, this will use the predefined model maximum length if a maximum
                length is required by one of the truncation/padding parameters. If the model has no specific maximum
                input length (like XLNet) truncation/padding to a maximum length will be deactivated.
            stride (:obj:`int`, `optional`, defaults to 0):
                If set to a number along with :obj:`max_length`, the overflowing tokens returned when
                :obj:`return_overflowing_tokens=True` will contain some tokens from the end of the truncated sequence
                returned to provide some overlap between truncated and overflowing sequences. The value of this
                argument defines the number of overlapping tokens.
            is_split_into_words (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not the input is already pre-tokenized (e.g., split into words). If set to :obj:`True`, the
                tokenizer assumes the input is already split into words (for instance, by splitting it on whitespace)
                which it will tokenize. This is useful for NER or token classification.
            pad_to_multiple_of (:obj:`int`, `optional`):
                If set will pad the sequence to a multiple of the provided value. This is especially useful to enable
                the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
            return_tensors (:obj:`str` or :class:`~transformers.file_utils.TensorType`, `optional`):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                * :obj:`'tf'`: Return TensorFlow :obj:`tf.constant` objects.
                * :obj:`'pt'`: Return PyTorch :obj:`torch.Tensor` objects.
                * :obj:`'np'`: Return Numpy :obj:`np.ndarray` objects.
"""
ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING = r"""
            return_token_type_ids (:obj:`bool`, `optional`):
                Whether to return token type IDs. If left to the default, will return the token type IDs according to
                the specific tokenizer's default, defined by the :obj:`return_outputs` attribute.

                `What are token type IDs? <../glossary.html#token-type-ids>`__
            return_attention_mask (:obj:`bool`, `optional`):
                Whether to return the attention mask. If left to the default, will return the attention mask according
                to the specific tokenizer's default, defined by the :obj:`return_outputs` attribute.

                `What are attention masks? <../glossary.html#attention-mask>`__
            return_overflowing_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to return overflowing token sequences.
            return_special_tokens_mask (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to return special tokens mask information.
            return_offsets_mapping (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to return :obj:`(char_start, char_end)` for each token.

                This is only available on fast tokenizers inheriting from
                :class:`~transformers.PreTrainedTokenizerFast`, if using Python's tokenizer, this method will raise
                :obj:`NotImplementedError`.
            return_length  (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to return the lengths of the encoded inputs.
            verbose (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether or not to print more information and warnings.
            **kwargs: passed to the :obj:`self.tokenize()` method

        Return:
            :class:`~transformers.BatchEncoding`: A :class:`~transformers.BatchEncoding` with the following fields:

            - **input_ids** -- List of token ids to be fed to a model.

              `What are input IDs? <../glossary.html#input-ids>`__

            - **token_type_ids** -- List of token type ids to be fed to a model (when :obj:`return_token_type_ids=True`
              or if `"token_type_ids"` is in :obj:`self.model_input_names`).

              `What are token type IDs? <../glossary.html#token-type-ids>`__

            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              :obj:`return_attention_mask=True` or if `"attention_mask"` is in :obj:`self.model_input_names`).

              `What are attention masks? <../glossary.html#attention-mask>`__

            - **overflowing_tokens** -- List of overflowing tokens sequences (when a :obj:`max_length` is specified and
              :obj:`return_overflowing_tokens=True`).
            - **num_truncated_tokens** -- Number of tokens truncated (when a :obj:`max_length` is specified and
              :obj:`return_overflowing_tokens=True`).
            - **special_tokens_mask** -- List of 0s and 1s, with 1 specifying added special tokens and 0 specifying
              regular sequence tokens (when :obj:`add_special_tokens=True` and :obj:`return_special_tokens_mask=True`).
            - **length** -- The length of the inputs (when :obj:`return_length=True`)
"""
INIT_TOKENIZER_DOCSTRING = r"""
    Class attributes (overridden by derived classes)

        - **vocab_files_names** (:obj:`Dict[str, str]`) -- A dictionary with, as keys, the ``__init__`` keyword name of
          each vocabulary file required by the model, and as associated values, the filename for saving the associated
          file (string).
        - **pretrained_vocab_files_map** (:obj:`Dict[str, Dict[str, str]]`) -- A dictionary of dictionaries, with the
          high-level keys being the ``__init__`` keyword name of each vocabulary file required by the model, the
          low-level being the :obj:`short-cut-names` of the pretrained models with, as associated values, the
          :obj:`url` to the associated pretrained vocabulary file.
        - **max_model_input_sizes** (:obj:`Dict[str, Optinal[int]]`) -- A dictionary with, as keys, the
          :obj:`short-cut-names` of the pretrained models, and as associated values, the maximum length of the sequence
          inputs of this model, or :obj:`None` if the model has no maximum input size.
        - **pretrained_init_configuration** (:obj:`Dict[str, Dict[str, Any]]`) -- A dictionary with, as keys, the
          :obj:`short-cut-names` of the pretrained models, and as associated values, a dictionary of specific arguments
          to pass to the ``__init__`` method of the tokenizer class for this pretrained model when loading the
          tokenizer with the :meth:`~transformers.tokenization_utils_base.PreTrainedTokenizerBase.from_pretrained`
          method.
        - **model_input_names** (:obj:`List[str]`) -- A list of inputs expected in the forward pass of the model.
        - **padding_side** (:obj:`str`) -- The default value for the side on which the model should have padding
          applied. Should be :obj:`'right'` or :obj:`'left'`.

    Args:
        model_max_length (:obj:`int`, `optional`):
            The maximum length (in number of tokens) for the inputs to the transformer model. When the tokenizer is
            loaded with :meth:`~transformers.tokenization_utils_base.PreTrainedTokenizerBase.from_pretrained`, this
            will be set to the value stored for the associated model in ``max_model_input_sizes`` (see above). If no
            value is provided, will default to VERY_LARGE_INTEGER (:obj:`int(1e30)`).
        padding_side: (:obj:`str`, `optional`):
            The side on which the model should have padding applied. Should be selected between ['right', 'left'].
            Default value is picked from the class attribute of the same name.
        model_input_names (:obj:`List[string]`, `optional`):
            The list of inputs accepted by the forward pass of the model (like :obj:`"token_type_ids"` or
            :obj:`"attention_mask"`). Default value is picked from the class attribute of the same name.
        bos_token (:obj:`str` or :obj:`tokenizers.AddedToken`, `optional`):
            A special token representing the beginning of a sentence. Will be associated to ``self.bos_token`` and
            ``self.bos_token_id``.
        eos_token (:obj:`str` or :obj:`tokenizers.AddedToken`, `optional`):
            A special token representing the end of a sentence. Will be associated to ``self.eos_token`` and
            ``self.eos_token_id``.
        unk_token (:obj:`str` or :obj:`tokenizers.AddedToken`, `optional`):
            A special token representing an out-of-vocabulary token. Will be associated to ``self.unk_token`` and
            ``self.unk_token_id``.
        sep_token (:obj:`str` or :obj:`tokenizers.AddedToken`, `optional`):
            A special token separating two different sentences in the same input (used by BERT for instance). Will be
            associated to ``self.sep_token`` and ``self.sep_token_id``.
        pad_token (:obj:`str` or :obj:`tokenizers.AddedToken`, `optional`):
            A special token used to make arrays of tokens the same size for batching purpose. Will then be ignored by
            attention mechanisms or loss computation. Will be associated to ``self.pad_token`` and
            ``self.pad_token_id``.
        cls_token (:obj:`str` or :obj:`tokenizers.AddedToken`, `optional`):
            A special token representing the class of the input (used by BERT for instance). Will be associated to
            ``self.cls_token`` and ``self.cls_token_id``.
        mask_token (:obj:`str` or :obj:`tokenizers.AddedToken`, `optional`):
            A special token representing a masked token (used by masked-language modeling pretraining objectives, like
            BERT). Will be associated to ``self.mask_token`` and ``self.mask_token_id``.
        additional_special_tokens (tuple or list of :obj:`str` or :obj:`tokenizers.AddedToken`, `optional`):
            A tuple or a list of additional special tokens. Add them here to ensure they won't be split by the
            tokenization process. Will be associated to ``self.additional_special_tokens`` and
            ``self.additional_special_tokens_ids``.
"""
VERY_LARGE_INTEGER = int(1e30)  # This is used to set the max input length for a model with infinite size input
LARGE_INTEGER = int(1e20)  # This is used when we need something big but slightly smaller than VERY_LARGE_INTEGER

SPECIAL_TOKENS_MAP_FILE = "special_tokens_map.json"
ADDED_TOKENS_FILE = "added_tokens.json"
TOKENIZER_CONFIG_FILE = "tokenizer_config.json"

TextInput = str
PreTokenizedInput = List[str]
EncodedInput = List[int]
TextInputPair = Tuple[str, str]
PreTokenizedInputPair = Tuple[List[str], List[str]]
EncodedInputPair = Tuple[List[int], List[int]]

FULL_TOKENIZER_FILE = "tokenizer.json"
_re_tokenizer_file = re.compile(r"tokenizer\.(.*)\.json")

def _insert_one_token_to_ordered_list(
    token_list: List[str], 
    new_token: str
):
    insertion_idx = bisect.bisect_left(token_list, new_token)
    if insertion_idx < len(token_list) and token_list[insertion_idx] == new_token:
        return
    else:
        token_list.insert(insertion_idx, new_token)

def get_fast_tokenizer_file(
    path_or_repo: Union[str, os.PathLike],
    revision: Optional[str] = None,
    use_auth_token: Optional[Union[bool, str]] = None,
) -> str:
    all_files = get_list_of_files(path_or_repo, revision=revision, use_auth_token=use_auth_token)
    print(path_or_repo)
    tokenizer_files_map = {}
    for file_name in all_files:
        search = _re_tokenizer_file.search(file_name)
        if search is not None:
            v = search.groups()[0]
            tokenizer_files_map[v] = file_name
    available_versions = sorted(tokenizer_files_map.keys())
    tokenizer_file = FULL_TOKENIZER_FILE
    return tokenizer_file


class SpecialTokensMixin:
    SPECIAL_TOKENS_ATTRIBUTES = [
        "bos_token",
        "eos_token",
        "unk_token",
        "sep_token",
        "pad_token",
        "cls_token",
        "mask_token",
        "additional_special_tokens",
    ]

    def __init__(self, verbose=True, **kwargs):
        self._bos_token = None
        self._eos_token = None
        self._unk_token = None
        self._sep_token = None
        self._pad_token = None
        self._cls_token = None
        self._mask_token = None
        self._pad_token_type_id = 0
        self._additional_special_tokens = []
        self.verbose = verbose

        for key, value in kwargs.items():
            if value is None:
                continue
            if key in self.SPECIAL_TOKENS_ATTRIBUTES:
                if key == "additional_special_tokens":
                    assert isinstance(value, (list, tuple)), f"Value {value} is not a list or tuple"
                    assert all(
                        isinstance(t, (str, AddedToken)) for t in value
                    ), "One of the tokens is not a string or an AddedToken"
                    setattr(self, key, value)
                elif isinstance(value, (str, AddedToken)):
                    setattr(self, key, value)
                else:
                    raise TypeError(f"special token {key} has to be either str or AddedToken but got: {type(value)}")

    def sanitize_special_tokens(self) -> int:
        return self.add_tokens(self.all_special_tokens_extended, special_tokens=True)

    def add_special_tokens(self, special_tokens_dict: Dict[str, Union[str, AddedToken]]) -> int:
        if not special_tokens_dict:
            return 0

        added_tokens = 0
        for key, value in special_tokens_dict.items():
            assert key in self.SPECIAL_TOKENS_ATTRIBUTES, f"Key {key} is not a special token"

            setattr(self, key, value)

            if key == "additional_special_tokens":
                assert isinstance(value, (list, tuple)) and all(
                    isinstance(t, (str, AddedToken)) for t in value
                ), f"Tokens {value} for key {key} should all be str or AddedToken instances"
                added_tokens += self.add_tokens(value, special_tokens=True)
            else:
                assert isinstance(
                    value, (str, AddedToken)
                ), f"Token {value} for key {key} should be a str or an AddedToken instance"
                added_tokens += self.add_tokens([value], special_tokens=True)

        return added_tokens

    def add_tokens(
        self, new_tokens: Union[str, AddedToken, List[Union[str, AddedToken]]], special_tokens: bool = False
    ) -> int:
        if not new_tokens:
            return 0

        if not isinstance(new_tokens, (list, tuple)):
            new_tokens = [new_tokens]

        return self._add_tokens(new_tokens, special_tokens=special_tokens)

    def _add_tokens(self, new_tokens: Union[List[str], List[AddedToken]], special_tokens: bool = False) -> int:
        raise NotImplementedError

    @property
    def bos_token(self) -> str:
        if self._bos_token is None and self.verbose:
            return None
        return str(self._bos_token)

    @property
    def eos_token(self) -> str:
        if self._eos_token is None and self.verbose:
            return None
        return str(self._eos_token)

    @property
    def unk_token(self) -> str:
        if self._unk_token is None and self.verbose:
            return None
        return str(self._unk_token)

    @property
    def sep_token(self) -> str:
        if self._sep_token is None and self.verbose:
            return None
        return str(self._sep_token)

    @property
    def pad_token(self) -> str:
        if self._pad_token is None and self.verbose:
            return None
        return str(self._pad_token)

    @property
    def cls_token(self) -> str:
        if self._cls_token is None and self.verbose:
            return None
        return str(self._cls_token)

    @property
    def mask_token(self) -> str:
        if self._mask_token is None and self.verbose:
            return None
        return str(self._mask_token)

    @property
    def additional_special_tokens(self) -> List[str]:
        if self._additional_special_tokens is None and self.verbose:
            return None
        return [str(tok) for tok in self._additional_special_tokens]

    @bos_token.setter
    def bos_token(self, value):
        self._bos_token = value

    @eos_token.setter
    def eos_token(self, value):
        self._eos_token = value

    @unk_token.setter
    def unk_token(self, value):
        self._unk_token = value

    @sep_token.setter
    def sep_token(self, value):
        self._sep_token = value

    @pad_token.setter
    def pad_token(self, value):
        self._pad_token = value

    @cls_token.setter
    def cls_token(self, value):
        self._cls_token = value

    @mask_token.setter
    def mask_token(self, value):
        self._mask_token = value

    @additional_special_tokens.setter
    def additional_special_tokens(self, value):
        self._additional_special_tokens = value

    @property
    def bos_token_id(self) -> Optional[int]:
        if self._bos_token is None:
            return None
        return self.convert_tokens_to_ids(self.bos_token)

    @property
    def eos_token_id(self) -> Optional[int]:
        if self._eos_token is None:
            return None
        return self.convert_tokens_to_ids(self.eos_token)

    @property
    def unk_token_id(self) -> Optional[int]:
        if self._unk_token is None:
            return None
        return self.convert_tokens_to_ids(self.unk_token)

    @property
    def sep_token_id(self) -> Optional[int]:
        if self._sep_token is None:
            return None
        return self.convert_tokens_to_ids(self.sep_token)

    @property
    def pad_token_id(self) -> Optional[int]:
        if self._pad_token is None:
            return None
        return self.convert_tokens_to_ids(self.pad_token)

    @property
    def pad_token_type_id(self) -> int:
        return self._pad_token_type_id

    @property
    def cls_token_id(self) -> Optional[int]:
        if self._cls_token is None:
            return None
        return self.convert_tokens_to_ids(self.cls_token)

    @property
    def mask_token_id(self) -> Optional[int]:
        if self._mask_token is None:
            return None
        return self.convert_tokens_to_ids(self.mask_token)

    @property
    def additional_special_tokens_ids(self) -> List[int]:
        return self.convert_tokens_to_ids(self.additional_special_tokens)

    @bos_token_id.setter
    def bos_token_id(self, value):
        self._bos_token = self.convert_tokens_to_ids(value)

    @eos_token_id.setter
    def eos_token_id(self, value):
        self._eos_token = self.convert_tokens_to_ids(value)

    @unk_token_id.setter
    def unk_token_id(self, value):
        self._unk_token = self.convert_tokens_to_ids(value)

    @sep_token_id.setter
    def sep_token_id(self, value):
        self._sep_token = self.convert_tokens_to_ids(value)

    @pad_token_id.setter
    def pad_token_id(self, value):
        self._pad_token = self.convert_tokens_to_ids(value)

    @cls_token_id.setter
    def cls_token_id(self, value):
        self._cls_token = self.convert_tokens_to_ids(value)

    @mask_token_id.setter
    def mask_token_id(self, value):
        self._mask_token = self.convert_tokens_to_ids(value)

    @additional_special_tokens_ids.setter
    def additional_special_tokens_ids(self, values):
        self._additional_special_tokens = [self.convert_tokens_to_ids(value) for value in values]

    @property
    def special_tokens_map(self) -> Dict[str, Union[str, List[str]]]:
        set_attr = {}
        for attr in self.SPECIAL_TOKENS_ATTRIBUTES:
            attr_value = getattr(self, "_" + attr)
            if attr_value:
                set_attr[attr] = (
                    type(attr_value)(str(attr_value_sub) for attr_value_sub in attr_value)
                    if isinstance(attr_value, (list, tuple))
                    else str(attr_value)
                )
        return set_attr

    @property
    def special_tokens_map_extended(self) -> Dict[str, Union[str, AddedToken, List[Union[str, AddedToken]]]]:
        set_attr = {}
        for attr in self.SPECIAL_TOKENS_ATTRIBUTES:
            attr_value = getattr(self, "_" + attr)
            if attr_value:
                set_attr[attr] = attr_value
        return set_attr

    @property
    def all_special_tokens(self) -> List[str]:
        all_toks = [str(s) for s in self.all_special_tokens_extended]
        return all_toks

    @property
    def all_special_tokens_extended(self) -> List[Union[str, AddedToken]]:
        all_toks = []
        set_attr = self.special_tokens_map_extended
        for attr_value in set_attr.values():
            all_toks = all_toks + (list(attr_value) if isinstance(attr_value, (list, tuple)) else [attr_value])
        all_toks = list(OrderedDict.fromkeys(all_toks))
        return all_toks

    @property
    def all_special_ids(self) -> List[int]:
        all_toks = self.all_special_tokens
        all_ids = self.convert_tokens_to_ids(all_toks)
        return all_ids

class PreTrainedTokenizerBase(SpecialTokensMixin):
    vocab_files_names: Dict[str, str] = {}
    pretrained_vocab_files_map: Dict[str, Dict[str, str]] = {}
    pretrained_init_configuration: Dict[str, Dict[str, Any]] = {}
    max_model_input_sizes: Dict[str, Optional[int]] = {}
    model_input_names: List[str] = ["input_ids", "token_type_ids", "attention_mask"]
    padding_side: str = "right"
    slow_tokenizer_class = None

    def __init__(self, **kwargs):
        self.init_inputs = ()
        self.init_kwargs = copy.deepcopy(kwargs)
        self.name_or_path = kwargs.pop("name_or_path", "")
        model_max_length = kwargs.pop("model_max_length", kwargs.pop("max_len", None))
        self.model_max_length = model_max_length if model_max_length is not None else VERY_LARGE_INTEGER
        self.padding_side = kwargs.pop("padding_side", self.padding_side)
        assert self.padding_side in [
            "right",
            "left",
        ], f"Padding side should be selected between 'right' and 'left', current value: {self.padding_side}"
        self.model_input_names = kwargs.pop("model_input_names", self.model_input_names)
        self.deprecation_warnings = ({})  
        super().__init__(**kwargs)

    @property
    def max_len_single_sentence(self) -> int:
        return self.model_max_length - self.num_special_tokens_to_add(pair=False)

    @property
    def max_len_sentences_pair(self) -> int:
        return self.model_max_length - self.num_special_tokens_to_add(pair=True)

    @max_len_single_sentence.setter
    def max_len_single_sentence(self, value) -> int:
        if value == self.model_max_length - self.num_special_tokens_to_add(pair=False) and self.verbose:
            if not self.deprecation_warnings.get("max_len_single_sentence", False):
                pass
            self.deprecation_warnings["max_len_single_sentence"] = True
        else:
            raise ValueError(
                "Setting 'max_len_single_sentence' is now deprecated. " "This value is automatically set up."
            )

    @max_len_sentences_pair.setter
    def max_len_sentences_pair(self, value) -> int:
        # For backward compatibility, allow to try to setup 'max_len_sentences_pair'.
        if value == self.model_max_length - self.num_special_tokens_to_add(pair=True) and self.verbose:
            if not self.deprecation_warnings.get("max_len_sentences_pair", False):
                pass
            self.deprecation_warnings["max_len_sentences_pair"] = True
        else:
            raise ValueError(
                "Setting 'max_len_sentences_pair' is now deprecated. " "This value is automatically set up."
            )

    def __repr__(self) -> str:
        return (
            f"{'PreTrainedTokenizerFast' if self.is_fast else 'PreTrainedTokenizer'}(name_or_path='{self.name_or_path}', "
            f"vocab_size={self.vocab_size}, model_max_len={self.model_max_length}, is_fast={self.is_fast}, "
            f"padding_side='{self.padding_side}', special_tokens={self.special_tokens_map_extended})"
        )

    def get_vocab(self) -> Dict[str, int]:
        raise NotImplementedError()

    @classmethod
    def from_pretrained(
        cls, 
        pretrained_model_name_or_path: Union[str, os.PathLike],
        *init_inputs, 
        **kwargs
    ):
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", False)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", None)
        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)

        user_agent = {"file_type": "tokenizer", "from_auto_class": from_auto_class, "is_fast": "Fast" in cls.__name__}
        if from_pipeline is not None:
            user_agent["using_pipeline"] = from_pipeline

        if is_offline_mode() and not local_files_only:
            local_files_only = True

        pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        vocab_files = {}
        init_configuration = {}

        if os.path.isfile(pretrained_model_name_or_path) or is_remote_url(pretrained_model_name_or_path):
            if len(cls.vocab_files_names) > 1:
                raise ValueError(
                    f"Calling {cls.__name__}.from_pretrained() with the path to a single file or url is not "
                    "supported for this tokenizer. Use a model identifier or the path to a directory instead."
                )
            warnings.warn(
                f"Calling {cls.__name__}.from_pretrained() with the path to a single file or url is deprecated and "
                "won't be possible anymore in v5. Use a model identifier or the path to a directory instead.",
                FutureWarning,
            )
            file_id = list(cls.vocab_files_names.keys())[0]
            vocab_files[file_id] = pretrained_model_name_or_path
        else:
            # At this point pretrained_model_name_or_path is either a directory or a model identifier name
            fast_tokenizer_file = get_fast_tokenizer_file(
                pretrained_model_name_or_path, revision=revision, use_auth_token=use_auth_token
            )
            additional_files_names = {
                "added_tokens_file": ADDED_TOKENS_FILE,
                "special_tokens_map_file": SPECIAL_TOKENS_MAP_FILE,
                "tokenizer_config_file": TOKENIZER_CONFIG_FILE,
                "tokenizer_file": fast_tokenizer_file,
            }
            # Look for the tokenizer files
            for file_id, file_name in {**cls.vocab_files_names, **additional_files_names}.items():
                if os.path.isdir(pretrained_model_name_or_path):
                    if subfolder is not None:
                        full_file_name = os.path.join(pretrained_model_name_or_path, subfolder, file_name)
                    else:
                        full_file_name = os.path.join(pretrained_model_name_or_path, file_name)
                    if not os.path.exists(full_file_name):
                        full_file_name = None
                # else:
                #     full_file_name = hf_bucket_url(
                #         pretrained_model_name_or_path,
                #         filename=file_name,
                #         subfolder=subfolder,
                #         revision=revision,
                #         mirror=None,
                #     )

                vocab_files[file_id] = full_file_name

        # Get files from url, cache, or disk depending on the case
        resolved_vocab_files = {}
        unresolved_files = []
        for file_id, file_path in vocab_files.items():
            if file_path is None:
                resolved_vocab_files[file_id] = None
            else:
                try:
                    resolved_vocab_files[file_id] = cached_path(
                        file_path,
                        cache_dir=cache_dir,
                        force_download=force_download,
                        proxies=proxies,
                        resume_download=resume_download,
                        local_files_only=local_files_only,
                        use_auth_token=use_auth_token,
                        user_agent=user_agent,
                    )

                except FileNotFoundError as error:
                    if local_files_only:
                        unresolved_files.append(file_id)
                    else:
                        raise error

                except requests.exceptions.HTTPError as err:
                    if "404 Client Error" in str(err):
                        resolved_vocab_files[file_id] = None
                    else:
                        raise err

        if len(unresolved_files) > 0:
            pass

        if all(full_file_name is None for full_file_name in resolved_vocab_files.values()):
            msg = (
                f"Can't load tokenizer for '{pretrained_model_name_or_path}'. Make sure that:\n\n"
                f"- '{pretrained_model_name_or_path}' is a correct model identifier listed on 'https://huggingface.co/models'\n\n"
                f"- or '{pretrained_model_name_or_path}' is the correct path to a directory containing relevant tokenizer files\n\n"
            )
            raise EnvironmentError(msg)

        for file_id, file_path in vocab_files.items():
            if file_id not in resolved_vocab_files:
                continue

            if file_path == resolved_vocab_files[file_id]:
                pass
            else:
                pass

        return cls._from_pretrained(
            resolved_vocab_files, pretrained_model_name_or_path, init_configuration, *init_inputs, **kwargs
        )

    @classmethod
    def _from_pretrained(
        cls, resolved_vocab_files, pretrained_model_name_or_path, init_configuration, *init_inputs, **kwargs
    ):
        # We instantiate fast tokenizers based on a slow tokenizer if we don't have access to the tokenizer.json
        # file or if `from_slow` is set to True.
        from_slow = kwargs.get("from_slow", False)
        has_tokenizer_file = resolved_vocab_files.get("tokenizer_file", None) is not None
        if (from_slow or not has_tokenizer_file) and cls.slow_tokenizer_class is not None:
            slow_tokenizer = (cls.slow_tokenizer_class)._from_pretrained(
                copy.deepcopy(resolved_vocab_files),
                pretrained_model_name_or_path,
                copy.deepcopy(init_configuration),
                *init_inputs,
                **(copy.deepcopy(kwargs)),
            )
        else:
            slow_tokenizer = None

        # Prepare tokenizer initialization kwargs
        # Did we saved some inputs and kwargs to reload ?
        tokenizer_config_file = resolved_vocab_files.pop("tokenizer_config_file", None)
        if tokenizer_config_file is not None:
            with open(tokenizer_config_file, encoding="utf-8") as tokenizer_config_handle:
                init_kwargs = json.load(tokenizer_config_handle)
            # First attempt. We get tokenizer_class from tokenizer_config to check mismatch between tokenizers.
            config_tokenizer_class = init_kwargs.get("tokenizer_class")
            init_kwargs.pop("tokenizer_class", None)
            saved_init_inputs = init_kwargs.pop("init_inputs", ())
            if not init_inputs:
                init_inputs = saved_init_inputs
        else:
            config_tokenizer_class = None
            init_kwargs = init_configuration

        # if config_tokenizer_class is None:
        #     from .models.auto.configuration_auto import AutoConfig

        #     # Second attempt. If we have not yet found tokenizer_class, let's try to use the config.
        #     try:
        #         config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        #         config_tokenizer_class = config.tokenizer_class
        #     except (OSError, ValueError, KeyError):
        #         # skip if an error occured.
        #         config = None
        #     if config_tokenizer_class is None:
        #         # Third attempt. If we have not yet found the original type of the tokenizer,
        #         # we are loading we see if we can infer it from the type of the configuration file
        #         from .models.auto.configuration_auto import CONFIG_MAPPING
        #         from .models.auto.tokenization_auto import TOKENIZER_MAPPING

        #         if hasattr(config, "model_type"):
        #             config_class = CONFIG_MAPPING.get(config.model_type)
        #         else:
        #             # Fallback: use pattern matching on the string.
        #             config_class = None
        #             for pattern, config_class_tmp in CONFIG_MAPPING.items():
        #                 if pattern in str(pretrained_model_name_or_path):
        #                     config_class = config_class_tmp
        #                     break

        #         if config_class in TOKENIZER_MAPPING.keys():
        #             config_tokenizer_class, config_tokenizer_class_fast = TOKENIZER_MAPPING[config_class]
        #             if config_tokenizer_class is not None:
        #                 config_tokenizer_class = config_tokenizer_class.__name__
        #             else:
        #                 config_tokenizer_class = config_tokenizer_class_fast.__name__

        if config_tokenizer_class is not None:
            if cls.__name__.replace("Fast", "") != config_tokenizer_class.replace("Fast", ""):
                pass
        # Update with newly provided kwargs
        init_kwargs.update(kwargs)

        # Convert AddedTokens serialized as dict to class instances
        def convert_added_tokens(obj: Union[AddedToken, Any]):
            if isinstance(obj, dict) and "__type" in obj and obj["__type"] == "AddedToken":
                obj.pop("__type")
                return AddedToken(**obj)
            elif isinstance(obj, (list, tuple)):
                return list(convert_added_tokens(o) for o in obj)
            elif isinstance(obj, dict):
                return {k: convert_added_tokens(v) for k, v in obj.items()}
            return obj

        init_kwargs = convert_added_tokens(init_kwargs)

        # Set max length if needed
        if pretrained_model_name_or_path in cls.max_model_input_sizes:
            # if we're using a pretrained model, ensure the tokenizer
            # wont index sequences longer than the number of positional embeddings
            model_max_length = cls.max_model_input_sizes[pretrained_model_name_or_path]
            if model_max_length is not None and isinstance(model_max_length, (int, float)):
                init_kwargs["model_max_length"] = min(init_kwargs.get("model_max_length", int(1e30)), model_max_length)

        # Merge resolved_vocab_files arguments in init_kwargs.
        added_tokens_file = resolved_vocab_files.pop("added_tokens_file", None)
        for args_name, file_path in resolved_vocab_files.items():
            if args_name not in init_kwargs:
                init_kwargs[args_name] = file_path

        if slow_tokenizer is not None:
            init_kwargs["__slow_tokenizer"] = slow_tokenizer

        init_kwargs["name_or_path"] = pretrained_model_name_or_path

        # Instantiate tokenizer.
        try:
            tokenizer = cls(*init_inputs, **init_kwargs)
        except OSError:
            raise OSError(
                "Unable to load vocabulary from file. "
                "Please check that the provided vocabulary is accessible and not corrupted."
            )

        # Save inputs and kwargs for saving and re-loading with ``save_pretrained``
        # Removed: Now done at the base class level
        # tokenizer.init_inputs = init_inputs
        # tokenizer.init_kwargs = init_kwargs

        # If there is a complementary special token map, load it
        special_tokens_map_file = resolved_vocab_files.pop("special_tokens_map_file", None)
        if special_tokens_map_file is not None:
            with open(special_tokens_map_file, encoding="utf-8") as special_tokens_map_handle:
                special_tokens_map = json.load(special_tokens_map_handle)
            for key, value in special_tokens_map.items():
                if isinstance(value, dict):
                    value = AddedToken(**value)
                elif isinstance(value, list):
                    value = [AddedToken(**token) if isinstance(token, dict) else token for token in value]
                setattr(tokenizer, key, value)

        # Add supplementary tokens.
        special_tokens = tokenizer.all_special_tokens
        if added_tokens_file is not None:
            with open(added_tokens_file, encoding="utf-8") as added_tokens_handle:
                added_tok_encoder = json.load(added_tokens_handle)

            # Sort added tokens by index
            added_tok_encoder_sorted = list(sorted(added_tok_encoder.items(), key=lambda x: x[1]))

            for token, index in added_tok_encoder_sorted:
                if has_tokenizer_file and index != len(tokenizer) and tokenizer.convert_tokens_to_ids(token) != index:
                    # Tokenizer fast: added token needs to either be in the vocabulary with the proper index or the
                    # index is the current length of the tokenizer (not in vocabulary)
                    raise ValueError(
                        f"Wrong index found for {token}: should be {tokenizer.convert_tokens_to_ids(token)} but found "
                        f"{index}."
                    )
                elif not has_tokenizer_file and index != len(tokenizer):
                    # Tokenizer slow: added token cannot already be in the vocabulary so its index needs to be the
                    # current length of the tokenizer.
                    raise ValueError(
                        f"Non-consecutive added token '{token}' found. "
                        f"Should have index {len(tokenizer)} but has index {index} in saved vocabulary."
                    )

                # Safe to call on a tokenizer fast even if token already there.
                tokenizer.add_tokens(token, special_tokens=bool(token in special_tokens))

        # Check all our special tokens are registered as "no split" token (we don't cut them) and are in the vocab
        added_tokens = tokenizer.sanitize_special_tokens()
        if added_tokens:
            pass

        return tokenizer

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        legacy_format: Optional[bool] = None,
        filename_prefix: Optional[str] = None,
        push_to_hub: bool = False,
        **kwargs,
    ) -> Tuple[str]:
        if os.path.isfile(save_directory):
            pass
            return

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            repo = self._create_or_get_repo(save_directory, **kwargs)

        os.makedirs(save_directory, exist_ok=True)

        special_tokens_map_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + SPECIAL_TOKENS_MAP_FILE
        )
        tokenizer_config_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + TOKENIZER_CONFIG_FILE
        )

        tokenizer_config = copy.deepcopy(self.init_kwargs)
        if len(self.init_inputs) > 0:
            tokenizer_config["init_inputs"] = copy.deepcopy(self.init_inputs)
        for file_id in self.vocab_files_names.keys():
            tokenizer_config.pop(file_id, None)

        # Sanitize AddedTokens
        def convert_added_tokens(obj: Union[AddedToken, Any], add_type_field=True):
            if isinstance(obj, AddedToken):
                out = obj.__getstate__()
                if add_type_field:
                    out["__type"] = "AddedToken"
                return out
            elif isinstance(obj, (list, tuple)):
                return list(convert_added_tokens(o, add_type_field=add_type_field) for o in obj)
            elif isinstance(obj, dict):
                return {k: convert_added_tokens(v, add_type_field=add_type_field) for k, v in obj.items()}
            return obj

        # add_type_field=True to allow dicts in the kwargs / differentiate from AddedToken serialization
        tokenizer_config = convert_added_tokens(tokenizer_config, add_type_field=True)

        # Add tokenizer class to the tokenizer config to be able to reload it with from_pretrained
        tokenizer_class = self.__class__.__name__
        # Remove the Fast at the end unless we have a special `PreTrainedTokenizerFast`
        if tokenizer_class.endswith("Fast") and tokenizer_class != "PreTrainedTokenizerFast":
            tokenizer_class = tokenizer_class[:-4]
        tokenizer_config["tokenizer_class"] = tokenizer_class

        with open(tokenizer_config_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(tokenizer_config, ensure_ascii=False))
        

        # Sanitize AddedTokens in special_tokens_map
        write_dict = convert_added_tokens(self.special_tokens_map_extended, add_type_field=False)
        with open(special_tokens_map_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(write_dict, ensure_ascii=False))
        

        file_names = (tokenizer_config_file, special_tokens_map_file)

        save_files = self._save_pretrained(
            save_directory=save_directory,
            file_names=file_names,
            legacy_format=legacy_format,
            filename_prefix=filename_prefix,
        )

        if push_to_hub:
            url = self._push_to_hub(repo, commit_message=commit_message)
            

        return save_files

    def _save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        file_names: Tuple[str],
        legacy_format: Optional[bool] = None,
        filename_prefix: Optional[str] = None,
    ) -> Tuple[str]:
        if legacy_format is False:
            raise ValueError(
                "Only fast tokenizers (instances of PreTrainedTokenizerFast) can be saved in non legacy format."
            )

        save_directory = str(save_directory)

        added_tokens_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + ADDED_TOKENS_FILE
        )
        added_vocab = self.get_added_vocab()
        if added_vocab:
            with open(added_tokens_file, "w", encoding="utf-8") as f:
                out_str = json.dumps(added_vocab, ensure_ascii=False)
                f.write(out_str)
               

        vocab_files = self.save_vocabulary(save_directory, filename_prefix=filename_prefix)

        return file_names + vocab_files + (added_tokens_file,)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        raise NotImplementedError

    def tokenize(self, text: str, pair: Optional[str] = None, add_special_tokens: bool = False, **kwargs) -> List[str]:
        raise NotImplementedError

    def encode(
        self,
        text: Union[TextInput, PreTokenizedInput, EncodedInput],
        text_pair: Optional[Union[TextInput, PreTokenizedInput, EncodedInput]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = False,
        max_length: Optional[int] = None,
        stride: int = 0,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs
    ) -> List[int]:

        encoded_inputs = self.encode_plus(
            text,
            text_pair=text_pair,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            stride=stride,
            return_tensors=return_tensors,
            **kwargs,
        )

        return encoded_inputs["input_ids"]

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        raise NotImplementedError

    def _get_padding_truncation_strategies(
        self, padding=False, truncation=False, max_length=None, pad_to_multiple_of=None, verbose=True, **kwargs
    ):
        """
        Find the correct padding/truncation strategy with backward compatibility for old arguments (truncation_strategy
        and pad_to_max_length) and behaviors.
        """
        old_truncation_strategy = kwargs.pop("truncation_strategy", "do_not_truncate")
        old_pad_to_max_length = kwargs.pop("pad_to_max_length", False)

        # Backward compatibility for previous behavior, maybe we should deprecate it:
        # If you only set max_length, it activates truncation for max_length
        if max_length is not None and padding is False and truncation is False:
            if verbose:
                if not self.deprecation_warnings.get("Truncation-not-explicitly-activated", False):
                    pass
                self.deprecation_warnings["Truncation-not-explicitly-activated"] = True
            truncation = "longest_first"

        # Get padding strategy
        if padding is False and old_pad_to_max_length:
            if verbose:
                warnings.warn(
                    "The `pad_to_max_length` argument is deprecated and will be removed in a future version, "
                    "use `padding=True` or `padding='longest'` to pad to the longest sequence in the batch, or "
                    "use `padding='max_length'` to pad to a max length. In this case, you can give a specific "
                    "length with `max_length` (e.g. `max_length=45`) or leave max_length to None to pad to the "
                    "maximal input size of the model (e.g. 512 for Bert).",
                    FutureWarning,
                )
            if max_length is None:
                padding_strategy = PaddingStrategy.LONGEST
            else:
                padding_strategy = PaddingStrategy.MAX_LENGTH
        elif padding is not False:
            if padding is True:
                padding_strategy = PaddingStrategy.LONGEST  # Default to pad to the longest sequence in the batch
            elif not isinstance(padding, PaddingStrategy):
                padding_strategy = PaddingStrategy(padding)
            elif isinstance(padding, PaddingStrategy):
                padding_strategy = padding
        else:
            padding_strategy = PaddingStrategy.DO_NOT_PAD

        # Get truncation strategy
        if truncation is False and old_truncation_strategy != "do_not_truncate":
            if verbose:
                warnings.warn(
                    "The `truncation_strategy` argument is deprecated and will be removed in a future version, "
                    "use `truncation=True` to truncate examples to a max length. You can give a specific "
                    "length with `max_length` (e.g. `max_length=45`) or leave max_length to None to truncate to the "
                    "maximal input size of the model (e.g. 512 for Bert). "
                    " If you have pairs of inputs, you can give a specific truncation strategy selected among "
                    "`truncation='only_first'` (will only truncate the first sentence in the pairs) "
                    "`truncation='only_second'` (will only truncate the second sentence in the pairs) "
                    "or `truncation='longest_first'` (will iteratively remove tokens from the longest sentence in the pairs).",
                    FutureWarning,
                )
            truncation_strategy = TruncationStrategy(old_truncation_strategy)
        elif truncation is not False:
            if truncation is True:
                truncation_strategy = (
                    TruncationStrategy.LONGEST_FIRST
                )  # Default to truncate the longest sequences in pairs of inputs
            elif not isinstance(truncation, TruncationStrategy):
                truncation_strategy = TruncationStrategy(truncation)
            elif isinstance(truncation, TruncationStrategy):
                truncation_strategy = truncation
        else:
            truncation_strategy = TruncationStrategy.DO_NOT_TRUNCATE

        # Set max length if needed
        if max_length is None:
            if padding_strategy == PaddingStrategy.MAX_LENGTH:
                if self.model_max_length > LARGE_INTEGER:
                    if verbose:
                        if not self.deprecation_warnings.get("Asking-to-pad-to-max_length", False):
                            pass
                        self.deprecation_warnings["Asking-to-pad-to-max_length"] = True
                    padding_strategy = PaddingStrategy.DO_NOT_PAD
                else:
                    max_length = self.model_max_length

            if truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE:
                if self.model_max_length > LARGE_INTEGER:
                    if verbose:
                        if not self.deprecation_warnings.get("Asking-to-truncate-to-max_length", False):
                            pass
                        self.deprecation_warnings["Asking-to-truncate-to-max_length"] = True
                    truncation_strategy = TruncationStrategy.DO_NOT_TRUNCATE
                else:
                    max_length = self.model_max_length

        # Test if we have a padding token
        if padding_strategy != PaddingStrategy.DO_NOT_PAD and (not self.pad_token or self.pad_token_id < 0):
            raise ValueError(
                "Asking to pad but the tokenizer does not have a padding token. "
                "Please select a token to use as `pad_token` `(tokenizer.pad_token = tokenizer.eos_token e.g.)` "
                "or add a new pad token via `tokenizer.add_special_tokens({'pad_token': '[PAD]'})`."
            )

        # Check that we will truncate to a multiple of pad_to_multiple_of if both are provided
        if (
            truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE
            and padding_strategy != PaddingStrategy.DO_NOT_PAD
            and pad_to_multiple_of is not None
            and max_length is not None
            and (max_length % pad_to_multiple_of != 0)
        ):
            raise ValueError(
                f"Truncation and padding are both activated but "
                f"truncation length ({max_length}) is not a multiple of pad_to_multiple_of ({pad_to_multiple_of})."
            )

        return padding_strategy, truncation_strategy, max_length, kwargs

    # @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]],
        text_pair: Optional[Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = False,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs
    ) -> BatchEncoding:
        
        def _is_valid_text_input(t):
            if isinstance(t, str):
                # Strings are fine
                return True
            elif isinstance(t, (list, tuple)):
                # List are fine as long as they are...
                if len(t) == 0:
                    # ... empty
                    return True
                elif isinstance(t[0], str):
                    # ... list of strings
                    return True
                elif isinstance(t[0], (list, tuple)):
                    # ... list with an empty list or with a list of strings
                    return len(t[0]) == 0 or isinstance(t[0][0], str)
                else:
                    return False
            else:
                return False

        if not _is_valid_text_input(text):
            raise ValueError(
                "text input must of type `str` (single example), `List[str]` (batch or single pretokenized example) "
                "or `List[List[str]]` (batch of pretokenized examples)."
            )

        if text_pair is not None and not _is_valid_text_input(text_pair):
            raise ValueError(
                "text input must of type `str` (single example), `List[str]` (batch or single pretokenized example) "
                "or `List[List[str]]` (batch of pretokenized examples)."
            )

        if is_split_into_words:
            is_batched = isinstance(text, (list, tuple)) and text and isinstance(text[0], (list, tuple))
        else:
            is_batched = isinstance(text, (list, tuple))

        if is_batched:
            if isinstance(text_pair, str):
                raise TypeError(
                    "when tokenizing batches of text, `text_pair` must be a list or tuple with the same length as `text`."
                )
            if text_pair is not None and len(text) != len(text_pair):
                raise ValueError(
                    f"batch length of `text`: {len(text)} does not match batch length of `text_pair`: {len(text_pair)}."
                )
            batch_text_or_text_pairs = list(zip(text, text_pair)) if text_pair is not None else text
            return self.batch_encode_plus(
                batch_text_or_text_pairs=batch_text_or_text_pairs,
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                stride=stride,
                is_split_into_words=is_split_into_words,
                pad_to_multiple_of=pad_to_multiple_of,
                return_tensors=return_tensors,
                return_token_type_ids=return_token_type_ids,
                return_attention_mask=return_attention_mask,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_offsets_mapping=return_offsets_mapping,
                return_length=return_length,
                verbose=verbose,
                **kwargs,
            )
        else:
            return self.encode_plus(
                text=text,
                text_pair=text_pair,
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                stride=stride,
                is_split_into_words=is_split_into_words,
                pad_to_multiple_of=pad_to_multiple_of,
                return_tensors=return_tensors,
                return_token_type_ids=return_token_type_ids,
                return_attention_mask=return_attention_mask,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_offsets_mapping=return_offsets_mapping,
                return_length=return_length,
                verbose=verbose,
                **kwargs,
            )

    # @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    def encode_plus(
        self,
        text: Union[TextInput, PreTokenizedInput, EncodedInput],
        text_pair: Optional[Union[TextInput, PreTokenizedInput, EncodedInput]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = False,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs
    ) -> BatchEncoding:

        # Backward compatibility for 'truncation_strategy', 'pad_to_max_length'
        padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            verbose=verbose,
            **kwargs,
        )

        return self._encode_plus(
            text=text,
            text_pair=text_pair,
            add_special_tokens=add_special_tokens,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            stride=stride,
            is_split_into_words=is_split_into_words,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
            return_token_type_ids=return_token_type_ids,
            return_attention_mask=return_attention_mask,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_offsets_mapping=return_offsets_mapping,
            return_length=return_length,
            verbose=verbose,
            **kwargs,
        )

    def _encode_plus(
        self,
        text: Union[TextInput, PreTokenizedInput, EncodedInput],
        text_pair: Optional[Union[TextInput, PreTokenizedInput, EncodedInput]] = None,
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs
    ) -> BatchEncoding:
        raise NotImplementedError

    # @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    def batch_encode_plus(
        self,
        batch_text_or_text_pairs: Union[
            List[TextInput],
            List[TextInputPair],
            List[PreTokenizedInput],
            List[PreTokenizedInputPair],
            List[EncodedInput],
            List[EncodedInputPair],
        ],
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = False,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs
    ) -> BatchEncoding:

        # Backward compatibility for 'truncation_strategy', 'pad_to_max_length'
        padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            verbose=verbose,
            **kwargs,
        )

        return self._batch_encode_plus(
            batch_text_or_text_pairs=batch_text_or_text_pairs,
            add_special_tokens=add_special_tokens,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            stride=stride,
            is_split_into_words=is_split_into_words,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
            return_token_type_ids=return_token_type_ids,
            return_attention_mask=return_attention_mask,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_offsets_mapping=return_offsets_mapping,
            return_length=return_length,
            verbose=verbose,
            **kwargs,
        )

    def _batch_encode_plus(
        self,
        batch_text_or_text_pairs: Union[
            List[TextInput],
            List[TextInputPair],
            List[PreTokenizedInput],
            List[PreTokenizedInputPair],
            List[EncodedInput],
            List[EncodedInputPair],
        ],
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs
    ) -> BatchEncoding:
        raise NotImplementedError

    def pad(
        self,
        encoded_inputs: Union[
            BatchEncoding,
            List[BatchEncoding],
            Dict[str, EncodedInput],
            Dict[str, List[EncodedInput]],
            List[Dict[str, EncodedInput]],
        ],
        padding: Union[bool, str, PaddingStrategy] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        verbose: bool = True,
    ) -> BatchEncoding:
        if isinstance(encoded_inputs, (list, tuple)) and isinstance(encoded_inputs[0], (dict, BatchEncoding)):
            encoded_inputs = {key: [example[key] for example in encoded_inputs] for key in encoded_inputs[0].keys()}

        # The model's main input name, usually `input_ids`, has be passed for padding
        if self.model_input_names[0] not in encoded_inputs:
            raise ValueError(
                "You should supply an encoding or a list of encodings to this method "
                f"that includes {self.model_input_names[0]}, but you provided {list(encoded_inputs.keys())}"
            )

        required_input = encoded_inputs[self.model_input_names[0]]

        if not required_input:
            if return_attention_mask:
                encoded_inputs["attention_mask"] = []
            return encoded_inputs

        first_element = required_input[0]
        if isinstance(first_element, (list, tuple)):
            # first_element might be an empty list/tuple in some edge cases so we grab the first non empty element.
            index = 0
            while len(required_input[index]) == 0:
                index += 1
            if index < len(required_input):
                first_element = required_input[index][0]
        # At this state, if `first_element` is still a list/tuple, it's an empty one so there is nothing to do.
        if not isinstance(first_element, (int, list, tuple)):
            return_tensors = "pt" if return_tensors is None else return_tensors

            for key, value in encoded_inputs.items():
                encoded_inputs[key] = to_py_obj(value)

        # Convert padding_strategy in PaddingStrategy
        padding_strategy, _, max_length, _ = self._get_padding_truncation_strategies(
            padding=padding, max_length=max_length, verbose=verbose
        )

        required_input = encoded_inputs[self.model_input_names[0]]
        if required_input and not isinstance(required_input[0], (list, tuple)):
            encoded_inputs = self._pad(
                encoded_inputs,
                max_length=max_length,
                padding_strategy=padding_strategy,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
            )
            return BatchEncoding(encoded_inputs, tensor_type=return_tensors)

        batch_size = len(required_input)
        assert all(
            len(v) == batch_size for v in encoded_inputs.values()
        ), "Some items in the output dictionary have a different batch size than others."

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = max(len(inputs) for inputs in required_input)
            padding_strategy = PaddingStrategy.MAX_LENGTH

        batch_outputs = {}
        for i in range(batch_size):
            inputs = dict((k, v[i]) for k, v in encoded_inputs.items())
            outputs = self._pad(
                inputs,
                max_length=max_length,
                padding_strategy=padding_strategy,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
            )

            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)

        return BatchEncoding(batch_outputs, tensor_type=return_tensors)

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:

        if token_ids_1 is None:
            return len(token_ids_0) * [0]
        return [0] * len(token_ids_0) + [1] * len(token_ids_1)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:

        if token_ids_1 is None:
            return token_ids_0
        return token_ids_0 + token_ids_1

    # @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    def prepare_for_model(
        self,
        ids: List[int],
        pair_ids: Optional[List[int]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = False,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        prepend_batch_axis: bool = False,
        **kwargs
    ) -> BatchEncoding:


        # Backward compatibility for 'truncation_strategy', 'pad_to_max_length'
        padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            verbose=verbose,
            **kwargs,
        )

        pair = bool(pair_ids is not None)
        len_ids = len(ids)
        len_pair_ids = len(pair_ids) if pair else 0

        if return_token_type_ids and not add_special_tokens:
            raise ValueError(
                "Asking to return token_type_ids while setting add_special_tokens to False "
                "results in an undefined behavior. Please set add_special_tokens to True or "
                "set return_token_type_ids to None."
            )

        # Load from model defaults
        if return_token_type_ids is None:
            return_token_type_ids = "token_type_ids" in self.model_input_names
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names

        encoded_inputs = {}

        # Compute the total size of the returned encodings
        total_len = len_ids + len_pair_ids + (self.num_special_tokens_to_add(pair=pair) if add_special_tokens else 0)

        # Truncation: Handle max sequence length
        overflowing_tokens = []
        if truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE and max_length and total_len > max_length:
            ids, pair_ids, overflowing_tokens = self.truncate_sequences(
                ids,
                pair_ids=pair_ids,
                num_tokens_to_remove=total_len - max_length,
                truncation_strategy=truncation_strategy,
                stride=stride,
            )

        if return_overflowing_tokens:
            encoded_inputs["overflowing_tokens"] = overflowing_tokens
            encoded_inputs["num_truncated_tokens"] = total_len - max_length

        # Add special tokens
        if add_special_tokens:
            sequence = self.build_inputs_with_special_tokens(ids, pair_ids)
            token_type_ids = self.create_token_type_ids_from_sequences(ids, pair_ids)
        else:
            sequence = ids + pair_ids if pair else ids
            token_type_ids = [0] * len(ids) + ([0] * len(pair_ids) if pair else [])

        # Build output dictionary
        encoded_inputs["input_ids"] = sequence
        if return_token_type_ids:
            encoded_inputs["token_type_ids"] = token_type_ids
        if return_special_tokens_mask:
            if add_special_tokens:
                encoded_inputs["special_tokens_mask"] = self.get_special_tokens_mask(ids, pair_ids)
            else:
                encoded_inputs["special_tokens_mask"] = [0] * len(sequence)

        # Check lengths
        self._eventual_warn_about_too_long_sequence(encoded_inputs["input_ids"], max_length, verbose)

        # Padding
        if padding_strategy != PaddingStrategy.DO_NOT_PAD or return_attention_mask:
            encoded_inputs = self.pad(
                encoded_inputs,
                max_length=max_length,
                padding=padding_strategy.value,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
            )

        if return_length:
            encoded_inputs["length"] = len(encoded_inputs["input_ids"])

        batch_outputs = BatchEncoding(
            encoded_inputs, tensor_type=return_tensors, prepend_batch_axis=prepend_batch_axis
        )

        return batch_outputs

    def truncate_sequences(
        self,
        ids: List[int],
        pair_ids: Optional[List[int]] = None,
        num_tokens_to_remove: int = 0,
        truncation_strategy: Union[str, TruncationStrategy] = "longest_first",
        stride: int = 0,
    ) -> Tuple[List[int], List[int], List[int]]:

        if num_tokens_to_remove <= 0:
            return ids, pair_ids, []

        if not isinstance(truncation_strategy, TruncationStrategy):
            truncation_strategy = TruncationStrategy(truncation_strategy)

        overflowing_tokens = []
        if truncation_strategy == TruncationStrategy.LONGEST_FIRST:
            for _ in range(num_tokens_to_remove):
                if pair_ids is None or len(ids) > len(pair_ids):
                    if not overflowing_tokens:
                        window_len = min(len(ids), stride + 1)
                    else:
                        window_len = 1
                    overflowing_tokens.extend(ids[-window_len:])
                    ids = ids[:-1]
                else:
                    if not overflowing_tokens:
                        window_len = min(len(pair_ids), stride + 1)
                    else:
                        window_len = 1
                    overflowing_tokens.extend(pair_ids[-window_len:])
                    pair_ids = pair_ids[:-1]
        elif truncation_strategy == TruncationStrategy.ONLY_FIRST:
            if len(ids) > num_tokens_to_remove:
                window_len = min(len(ids), stride + num_tokens_to_remove)
                overflowing_tokens = ids[-window_len:]
                ids = ids[:-num_tokens_to_remove]
            else:
                pass
        elif truncation_strategy == TruncationStrategy.ONLY_SECOND and pair_ids is not None:
            if len(pair_ids) > num_tokens_to_remove:
                window_len = min(len(pair_ids), stride + num_tokens_to_remove)
                overflowing_tokens = pair_ids[-window_len:]
                pair_ids = pair_ids[:-num_tokens_to_remove]
            else:
                pass

        return (ids, pair_ids, overflowing_tokens)

    def _pad(
        self,
        encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
        max_length: Optional[int] = None,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
    ) -> dict:
        # Load from model defaults
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names

        required_input = encoded_inputs[self.model_input_names[0]]

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = len(required_input)

        if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        needs_to_be_padded = padding_strategy != PaddingStrategy.DO_NOT_PAD and len(required_input) != max_length

        if needs_to_be_padded:
            difference = max_length - len(required_input)
            if self.padding_side == "right":
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = [1] * len(required_input) + [0] * difference
                if "token_type_ids" in encoded_inputs:
                    encoded_inputs["token_type_ids"] = (
                        encoded_inputs["token_type_ids"] + [self.pad_token_type_id] * difference
                    )
                if "special_tokens_mask" in encoded_inputs:
                    encoded_inputs["special_tokens_mask"] = encoded_inputs["special_tokens_mask"] + [1] * difference
                encoded_inputs[self.model_input_names[0]] = required_input + [self.pad_token_id] * difference
            elif self.padding_side == "left":
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = [0] * difference + [1] * len(required_input)
                if "token_type_ids" in encoded_inputs:
                    encoded_inputs["token_type_ids"] = [self.pad_token_type_id] * difference + encoded_inputs[
                        "token_type_ids"
                    ]
                if "special_tokens_mask" in encoded_inputs:
                    encoded_inputs["special_tokens_mask"] = [1] * difference + encoded_inputs["special_tokens_mask"]
                encoded_inputs[self.model_input_names[0]] = [self.pad_token_id] * difference + required_input
            else:
                raise ValueError("Invalid padding strategy:" + str(self.padding_side))
        elif return_attention_mask and "attention_mask" not in encoded_inputs:
            encoded_inputs["attention_mask"] = [1] * len(required_input)

        return encoded_inputs

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        raise NotImplementedError

    def batch_decode(
        self,
        sequences: Union[List[int], List[List[int]], "np.ndarray", "torch.Tensor"],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = True,
        **kwargs
    ) -> List[str]:
        return [
            self.decode(
                seq,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                **kwargs,
            )
            for seq in sequences
        ]

    def decode(
        self,
        token_ids: Union[int, List[int], "np.ndarray", "torch.Tensor"],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = True,
        **kwargs
    ) -> str:
        # Convert inputs to python lists
        token_ids = to_py_obj(token_ids)

        return self._decode(
            token_ids=token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

    def _decode(
        self,
        token_ids: Union[int, List[int]],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = True,
        **kwargs
    ) -> str:
        raise NotImplementedError

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        assert already_has_special_tokens and token_ids_1 is None, (
            "You cannot use ``already_has_special_tokens=False`` with this tokenizer. "
            "Please use a slow (full python) tokenizer to activate this argument."
            "Or set `return_special_tokens_mask=True` when calling the encoding method "
            "to get the special tokens mask in any tokenizer. "
        )

        all_special_ids = self.all_special_ids  # cache the property
        special_tokens_mask = [1 if token in all_special_ids else 0 for token in token_ids_0]
        return special_tokens_mask

    @staticmethod
    def clean_up_tokenization(out_string: str) -> str:
        out_string = (
            out_string.replace(" .", ".")
            .replace(" ?", "?")
            .replace(" !", "!")
            .replace(" ,", ",")
            .replace(" ' ", "'")
            .replace(" n't", "n't")
            .replace(" 'm", "'m")
            .replace(" 's", "'s")
            .replace(" 've", "'ve")
            .replace(" 're", "'re")
        )
        return out_string

    def _eventual_warn_about_too_long_sequence(self, ids: List[int], max_length: Optional[int], verbose: bool):
        if max_length is None and len(ids) > self.model_max_length and verbose:
            self.deprecation_warnings["sequence-length-is-longer-than-the-specified-maximum"] = True

    @contextmanager
    def as_target_tokenizer(self):
        yield

    def prepare_seq2seq_batch(
        self,
        src_texts: List[str],
        tgt_texts: Optional[List[str]] = None,
        max_length: Optional[int] = None,
        max_target_length: Optional[int] = None,
        padding: str = "longest",
        return_tensors: str = None,
        truncation: bool = True,
        **kwargs,
    ) -> BatchEncoding:
        
        kwargs.pop("src_lang", None)
        kwargs.pop("tgt_lang", None)
        if max_length is None:
            max_length = self.model_max_length
        model_inputs = self(
            src_texts,
            add_special_tokens=True,
            return_tensors=return_tensors,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            **kwargs,
        )
        if tgt_texts is None:
            return model_inputs
        if max_target_length is None:
            max_target_length = max_length
        with self.as_target_tokenizer():
            labels = self(
                tgt_texts,
                add_special_tokens=True,
                return_tensors=return_tensors,
                padding=padding,
                max_length=max_target_length,
                truncation=truncation,
                **kwargs,
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

class PreTrainedTokenizer(PreTrainedTokenizerBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.added_tokens_encoder: Dict[str, int] = {}
        self.added_tokens_decoder: Dict[int, str] = {}
        self.unique_no_split_tokens: List[str] = []
        self._decode_use_source_tokenizer = False

    @property
    def is_fast(self) -> bool:
        return False

    @property
    def vocab_size(self) -> int:
        raise NotImplementedError

    def get_added_vocab(self) -> Dict[str, int]:
        return self.added_tokens_encoder

    def __len__(self):
        return self.vocab_size + len(self.added_tokens_encoder)

    def _add_tokens(
        self, 
        new_tokens: Union[List[str], 
        List[AddedToken]], 
        special_tokens: bool = False
    ) -> int:
        new_tokens = [str(tok) for tok in new_tokens]
        tokens_to_add = []
        for token in new_tokens:
            assert isinstance(token, str)
            if not special_tokens and hasattr(self, "do_lower_case") and self.do_lower_case:
                token = token.lower()
            if (
                token != self.unk_token
                and self.convert_tokens_to_ids(token) == self.convert_tokens_to_ids(self.unk_token)
                and token not in tokens_to_add
            ):
                tokens_to_add.append(token)

        added_tok_encoder = dict((tok, len(self) + i) for i, tok in enumerate(tokens_to_add))
        added_tok_decoder = {v: k for k, v in added_tok_encoder.items()}
        self.added_tokens_encoder.update(added_tok_encoder)
        self.added_tokens_decoder.update(added_tok_decoder)

        if special_tokens:
            if len(new_tokens) == 1:
                _insert_one_token_to_ordered_list(self.unique_no_split_tokens, new_tokens[0])
            else:
                self.unique_no_split_tokens = sorted(set(self.unique_no_split_tokens).union(set(new_tokens)))
        else:
            if len(tokens_to_add) == 1:
                _insert_one_token_to_ordered_list(self.unique_no_split_tokens, tokens_to_add[0])
            else:
                self.unique_no_split_tokens = sorted(set(self.unique_no_split_tokens).union(set(tokens_to_add)))
        return len(tokens_to_add)

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        token_ids_0 = []
        token_ids_1 = []
        return len(self.build_inputs_with_special_tokens(token_ids_0, token_ids_1 if pair else None))

    def tokenize(
        self, 
        text: TextInput, 
        **kwargs
    ) -> List[str]:
        all_special_tokens_extended = dict(
            (str(t), t) for t in self.all_special_tokens_extended if isinstance(t, AddedToken)
        )
        text, kwargs = self.prepare_for_tokenization(text, **kwargs)

        if hasattr(self, "do_lower_case") and self.do_lower_case:
            escaped_special_toks = [re.escape(s_tok) for s_tok in self.all_special_tokens]
            pattern = r"(" + r"|".join(escaped_special_toks) + r")|" + r"(.+?)"
            text = re.sub(pattern, lambda m: m.groups()[0] or m.groups()[1].lower(), text)

        def split_on_token(tok, text):
            result = []
            tok_extended = all_special_tokens_extended.get(tok, None)
            split_text = text.split(tok)
            full_word = ""
            for i, sub_text in enumerate(split_text):
                if isinstance(tok_extended, AddedToken):
                    if tok_extended.single_word:
                        if (
                            i < len(split_text) - 1
                            and not _is_end_of_word(sub_text)
                            and not _is_start_of_word(split_text[i + 1])
                        ):
                            full_word += sub_text + tok
                        elif full_word:
                            full_word += sub_text
                            result.append(full_word)
                            full_word = ""
                            continue

                    if tok_extended.rstrip and i > 0:
                        sub_text = sub_text.lstrip()
                    if tok_extended.lstrip and i < len(split_text) - 1:
                        sub_text = sub_text.rstrip()  
                else:
                    if i < len(split_text) - 1:
                        sub_text = sub_text.rstrip()
                    if i > 0:
                        sub_text = sub_text.lstrip()

                if i == 0 and not sub_text:
                    result.append(tok)
                elif i == len(split_text) - 1:
                    if sub_text:
                        result.append(sub_text)
                    else:
                        pass
                else:
                    if sub_text:
                        result.append(sub_text)
                    result.append(tok)
            return result

        def split_on_tokens(tok_list, text):
            if not text.strip():
                return []
            if not tok_list:
                return self._tokenize(text)

            tokenized_text = []
            text_list = [text]
            for tok in tok_list:
                tokenized_text = []
                for sub_text in text_list:
                    if sub_text not in self.unique_no_split_tokens:
                        tokenized_text.extend(split_on_token(tok, sub_text))
                    else:
                        tokenized_text.append(sub_text)
                text_list = tokenized_text

            return list(
                itertools.chain.from_iterable(
                    (
                        self._tokenize(token) if token not in self.unique_no_split_tokens else [token]
                        for token in tokenized_text
                    )
                )
            )

        no_split_token = self.unique_no_split_tokens
        tokenized_text = split_on_tokens(no_split_token, text)
        return tokenized_text

    def _tokenize(self, text, **kwargs):
        raise NotImplementedError

    def convert_tokens_to_ids(
            self, 
            tokens: Union[str, List[str]]
        ) -> Union[int, List[int]]:

        if tokens is None:
            return None
        if isinstance(tokens, str):
            return self._convert_token_to_id_with_added_voc(tokens)
        ids = []
        for token in tokens:
            ids.append(self._convert_token_to_id_with_added_voc(token))
        return ids

    def _convert_token_to_id_with_added_voc(self, token):
        if token is None:
            return None
        if token in self.added_tokens_encoder:
            return self.added_tokens_encoder[token]
        return self._convert_token_to_id(token)

    def _convert_token_to_id(self, token):
        raise NotImplementedError

    def _encode_plus(
        self,
        text: Union[TextInput, PreTokenizedInput, EncodedInput],
        text_pair: Optional[Union[TextInput, PreTokenizedInput, EncodedInput]] = None,
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs
    ) -> BatchEncoding:
        def get_input_ids(text):
            if isinstance(text, str):
                tokens = self.tokenize(text, **kwargs)
                return self.convert_tokens_to_ids(tokens)
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], str):
                if is_split_into_words:
                    tokens = list(
                        itertools.chain(*(self.tokenize(t, is_split_into_words=True, **kwargs) for t in text))
                    )
                    return self.convert_tokens_to_ids(tokens)
                else:
                    return self.convert_tokens_to_ids(text)
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], int):
                return text
            else:
                if is_split_into_words:
                    raise ValueError(
                        f"Input {text} is not valid. Should be a string or a list/tuple of strings when `is_split_into_words=True`."
                    )
                else:
                    raise ValueError(
                        f"Input {text} is not valid. Should be a string, a list/tuple of strings or a list/tuple of integers."
                    )

        first_ids = get_input_ids(text)
        second_ids = get_input_ids(text_pair) if text_pair is not None else None

        return self.prepare_for_model(
            first_ids,
            pair_ids=second_ids,
            add_special_tokens=add_special_tokens,
            padding=padding_strategy.value,
            truncation=truncation_strategy.value,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
            prepend_batch_axis=True,
            return_attention_mask=return_attention_mask,
            return_token_type_ids=return_token_type_ids,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_length=return_length,
            verbose=verbose,
        )

    def _batch_encode_plus(
        self,
        batch_text_or_text_pairs: Union[
            List[TextInput],
            List[TextInputPair],
            List[PreTokenizedInput],
            List[PreTokenizedInputPair],
            List[EncodedInput],
            List[EncodedInputPair],
        ],
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs
    ) -> BatchEncoding:
        def get_input_ids(text):
            if isinstance(text, str):
                tokens = self.tokenize(text, **kwargs)
                return self.convert_tokens_to_ids(tokens)
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], str):
                if is_split_into_words:
                    tokens = list(
                        itertools.chain(*(self.tokenize(t, is_split_into_words=True, **kwargs) for t in text))
                    )
                    return self.convert_tokens_to_ids(tokens)
                else:
                    return self.convert_tokens_to_ids(text)
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], int):
                return text
            else:
                raise ValueError(
                    "Input is not valid. Should be a string, a list/tuple of strings or a list/tuple of integers."
                )

        input_ids = []
        for ids_or_pair_ids in batch_text_or_text_pairs:
            if not isinstance(ids_or_pair_ids, (list, tuple)):
                ids, pair_ids = ids_or_pair_ids, None
            elif is_split_into_words and not isinstance(ids_or_pair_ids[0], (list, tuple)):
                ids, pair_ids = ids_or_pair_ids, None
            else:
                ids, pair_ids = ids_or_pair_ids

            first_ids = get_input_ids(ids)
            second_ids = get_input_ids(pair_ids) if pair_ids is not None else None
            input_ids.append((first_ids, second_ids))

        batch_outputs = self._batch_prepare_for_model(
            input_ids,
            add_special_tokens=add_special_tokens,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
            return_token_type_ids=return_token_type_ids,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_length=return_length,
            return_tensors=return_tensors,
            verbose=verbose,
        )
        return BatchEncoding(batch_outputs)

    # @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    def _batch_prepare_for_model(
        self,
        batch_ids_pairs: List[Union[PreTokenizedInputPair, Tuple[List[int], None]]],
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[str] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_length: bool = False,
        verbose: bool = True,
    ) -> BatchEncoding:

        batch_outputs = {}
        for first_ids, second_ids in batch_ids_pairs:
            outputs = self.prepare_for_model(
                first_ids,
                second_ids,
                add_special_tokens=add_special_tokens,
                padding=PaddingStrategy.DO_NOT_PAD.value,  # we pad in batch afterward
                truncation=truncation_strategy.value,
                max_length=max_length,
                stride=stride,
                pad_to_multiple_of=None,  # we pad in batch afterward
                return_attention_mask=False,  # we pad in batch afterward
                return_token_type_ids=return_token_type_ids,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_length=return_length,
                return_tensors=None,  # We convert the whole batch to tensors at the end
                prepend_batch_axis=False,
                verbose=verbose,
            )

            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)

        batch_outputs = self.pad(
            batch_outputs,
            padding=padding_strategy.value,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
        )
        batch_outputs = BatchEncoding(batch_outputs, tensor_type=return_tensors)
        return batch_outputs

    def prepare_for_tokenization(
        self, text: str, is_split_into_words: bool = False, **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        return (text, kwargs)

    def get_special_tokens_mask(
        self, token_ids_0: List, token_ids_1: Optional[List] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formatted with special tokens for the model."
                )

            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )
        return [0] * ((len(token_ids_1) if token_ids_1 else 0) + len(token_ids_0))

    @overload
    def convert_ids_to_tokens(self, ids: int, skip_special_tokens: bool = False) -> str:
        ...

    @overload
    def convert_ids_to_tokens(self, ids: List[int], skip_special_tokens: bool = False) -> List[str]:
        ...

    def convert_ids_to_tokens(
        self, ids: Union[int, List[int]], skip_special_tokens: bool = False
    ) -> Union[str, List[str]]:
        if isinstance(ids, int):
            if ids in self.added_tokens_decoder:
                return self.added_tokens_decoder[ids]
            else:
                return self._convert_id_to_token(ids)
        tokens = []
        for index in ids:
            index = int(index)
            if skip_special_tokens and index in self.all_special_ids:
                continue
            if index in self.added_tokens_decoder:
                tokens.append(self.added_tokens_decoder[index])
            else:
                tokens.append(self._convert_id_to_token(index))
        return tokens

    def _convert_id_to_token(self, index: int) -> str:
        raise NotImplementedError

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return " ".join(tokens)

    def _decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = True,
        spaces_between_special_tokens: bool = True,
        **kwargs
    ) -> str:
        self._decode_use_source_tokenizer = kwargs.pop("use_source_tokenizer", False)
        filtered_tokens = self.convert_ids_to_tokens(token_ids, skip_special_tokens=skip_special_tokens)
        sub_texts = []
        current_sub_text = []
        for token in filtered_tokens:
            if skip_special_tokens and token in self.all_special_ids:
                continue
            if token in self.added_tokens_encoder:
                if current_sub_text:
                    sub_texts.append(self.convert_tokens_to_string(current_sub_text))
                    current_sub_text = []
                sub_texts.append(token)
            else:
                current_sub_text.append(token)
        if current_sub_text:
            sub_texts.append(self.convert_tokens_to_string(current_sub_text))

        if spaces_between_special_tokens:
            text = " ".join(sub_texts)
        else:
            text = "".join(sub_texts)

        if clean_up_tokenization_spaces:
            clean_text = self.clean_up_tokenization(text)
            return clean_text
        else:
            return text

class Wav2Vec2CTCTokenizer(PreTrainedTokenizer):
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        word_delimiter_token="|",
        do_lower_case=False,
        **kwargs
    ):
        super().__init__(
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            do_lower_case=do_lower_case,
            word_delimiter_token=word_delimiter_token,
            **kwargs,
        )

        self._word_delimiter_token = word_delimiter_token
        self.do_lower_case = do_lower_case

        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        self.decoder = {v: k for k, v in self.encoder.items()}
        for token in self.encoder.keys():
            if len(token) > 1:
                self.unique_no_split_tokens.append(token)

    @property
    def word_delimiter_token(self) -> str:
        if self._word_delimiter_token is None and self.verbose:
            return None
        return str(self._word_delimiter_token)

    @property
    def word_delimiter_token_id(self) -> Optional[int]:
        if self._word_delimiter_token is None:
            return None
        return self.convert_tokens_to_ids(self.word_delimiter_token)

    @word_delimiter_token.setter
    def word_delimiter_token(self, value):
        self._word_delimiter_token = value

    @word_delimiter_token_id.setter
    def word_delimiter_token_id(self, value):
        self._word_delimiter_token = self.convert_tokens_to_ids(value)

    @property
    def vocab_size(self) -> int:
        return len(self.decoder)

    def get_vocab(self) -> Dict:
        return dict(self.encoder, **self.added_tokens_encoder)

    def _tokenize(self, text, **kwargs):
        if self.do_lower_case:
            text = text.upper()
        return list(text.replace(" ", self.word_delimiter_token))

    def _convert_token_to_id(self, token: str) -> int:
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index: int) -> str:
        result = self.decoder.get(index, self.unk_token)
        return result

    def convert_tokens_to_string(
        self, 
        tokens: List[str], 
        group_tokens: bool = True, 
        spaces_between_special_tokens: bool = False
    ) -> str:
        
        if group_tokens:
            tokens = [token_group[0] for token_group in groupby(tokens)]
        filtered_tokens = list(filter(lambda token: token != self.pad_token, tokens))
        if spaces_between_special_tokens:
            join_token = " "
        else:
            join_token = ""

        string = join_token.join(
            [" " if token == self.word_delimiter_token else token for token in filtered_tokens]
        ).strip()

        if self.do_lower_case:
            string = string.lower()
        return string

    def prepare_for_tokenization(
        self, 
        text, 
        is_split_into_words=False,
        **kwargs
    ):
        if is_split_into_words:
            text = " " + text
        return (text, kwargs)

    def _decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = True,
        group_tokens: bool = True,
        spaces_between_special_tokens: bool = False,
    ) -> str:
        filtered_tokens = self.convert_ids_to_tokens(token_ids, skip_special_tokens=skip_special_tokens)

        result = []
        for token in filtered_tokens:
            if skip_special_tokens and token in self.all_special_ids:
                continue
            result.append(token)

        text = self.convert_tokens_to_string(
            result, 
            group_tokens=group_tokens, 
            spaces_between_special_tokens=spaces_between_special_tokens
        )
        if clean_up_tokenization_spaces:
            clean_text = self.clean_up_tokenization(text)
            return clean_text
        else:
            return text

    def save_vocabulary(
        self, 
        save_directory: str,
        filename_prefix: Optional[str] = None
    ) -> Tuple[str]:
        if not os.path.isdir(save_directory):
            return
        vocab_file = os.path.join(save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"])
        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, ensure_ascii=False))
        return (vocab_file,)

    def _add_tokens(self, new_tokens: Union[List[str], List[AddedToken]], special_tokens: bool = False) -> int:
        new_tokens = [str(tok) for tok in new_tokens]
        tokens_to_add = []
        for token in new_tokens:
            assert isinstance(token, str)
            if not special_tokens and hasattr(self, "do_lower_case") and self.do_lower_case:
                token = token.lower()
            if (
                token != self.unk_token
                and self.convert_tokens_to_ids(token) == self.convert_tokens_to_ids(self.unk_token)
                and token not in tokens_to_add
            ):
                tokens_to_add.append(token)

        added_tok_encoder = dict((tok, len(self) + i) for i, tok in enumerate(tokens_to_add))
        added_tok_decoder = {v: k for k, v in added_tok_encoder.items()}
        self.added_tokens_encoder.update(added_tok_encoder)
        self.added_tokens_decoder.update(added_tok_decoder)

        for token in tokens_to_add:
            if len(token) > 1:
                self._additional_special_tokens.append(AddedToken(token))
                _insert_one_token_to_ordered_list(self.unique_no_split_tokens, token)

        return len(tokens_to_add)
