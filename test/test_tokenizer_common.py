import os
import shutil
import tempfile
import inspect
import re

from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Union
# from transformers import PreTrainedTokenizerBase, PreTrainedTokenizer, PreTrainedTokenizerFast

from wav2vec2CTCTokenizer import PreTrainedTokenizerBase, PreTrainedTokenizer#, PreTrainedTokenizerFast


def get_tests_dir(append_path=None):
    caller__file__ = inspect.stack()[1][1]
    tests_dir = os.path.abspath(os.path.dirname(caller__file__))
    if append_path:
        return os.path.join(tests_dir, append_path)
    else:
        return tests_dir


class TokenizerTesterMixin:
    tokenizer_class = None
    rust_tokenizer_class = None
    test_slow_tokenizer = True
    test_rust_tokenizer = False
    space_between_special_tokens = False
    from_pretrained_kwargs = None
    from_pretrained_filter = None
    from_pretrained_vocab_key = "vocab_file"
    test_seq2seq = True
    test_sentencepiece = False
    test_sentencepiece_ignore_case = False

    def setUp(self) -> None:
        if self.test_rust_tokenizer:
            tokenizers_list = [
                (
                    self.rust_tokenizer_class,
                    pretrained_name,
                    self.from_pretrained_kwargs if self.from_pretrained_kwargs is not None else {},
                )
                for pretrained_name in self.rust_tokenizer_class.pretrained_vocab_files_map[
                    self.from_pretrained_vocab_key
                ].keys()
                if self.from_pretrained_filter is None
                or (self.from_pretrained_filter is not None and self.from_pretrained_filter(pretrained_name))
            ]
            self.tokenizers_list = tokenizers_list[:1]  # Let's just test the first pretrained vocab for speed
        else:
            self.tokenizers_list = []

        with open(f"{get_tests_dir()}/fixtures/sample_text.txt", encoding="utf-8") as f_data:
            self._data = f_data.read().replace("\n\n", "\n").strip()

        self.tmpdirname = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    def get_input_output_texts(self, tokenizer):
        input_txt = self.get_clean_sequence(tokenizer)[0]
        return input_txt, input_txt

    def get_clean_sequence(self, tokenizer, with_prefix_space=False, max_length=20, min_length=5) -> Tuple[str, list]:
        toks = [(i, tokenizer.decode([i], clean_up_tokenization_spaces=False)) for i in range(len(tokenizer))]
        toks = list(filter(lambda t: re.match(r"^[ a-zA-Z]+$", t[1]), toks))
        toks = list(filter(lambda t: [t[0]] == tokenizer.encode(t[1], add_special_tokens=False), toks))
        if max_length is not None and len(toks) > max_length:
            toks = toks[:max_length]
        if min_length is not None and len(toks) < min_length and len(toks) > 0:
            while len(toks) < min_length:
                toks = toks + toks
        # toks_str = [t[1] for t in toks]
        toks_ids = [t[0] for t in toks]

        # Ensure consistency
        output_txt = tokenizer.decode(toks_ids, clean_up_tokenization_spaces=False)
        if " " not in output_txt and len(toks_ids) > 1:
            output_txt = (
                tokenizer.decode([toks_ids[0]], clean_up_tokenization_spaces=False)
                + " "
                + tokenizer.decode(toks_ids[1:], clean_up_tokenization_spaces=False)
            )
        if with_prefix_space:
            output_txt = " " + output_txt
        output_ids = tokenizer.encode(output_txt, add_special_tokens=False)
        return output_txt, output_ids

    def get_tokenizers(self, fast=True, **kwargs) -> List[PreTrainedTokenizerBase]:
        if fast and self.test_rust_tokenizer and self.test_slow_tokenizer:
            return [self.get_tokenizer(**kwargs), self.get_rust_tokenizer(**kwargs)]
        elif fast and self.test_rust_tokenizer:
            return [self.get_rust_tokenizer(**kwargs)]
        elif self.test_slow_tokenizer:
            return [self.get_tokenizer(**kwargs)]
        else:
            raise ValueError("This tokenizer class has no tokenizer to be tested.")

    def get_tokenizer(self, **kwargs) -> PreTrainedTokenizer:
        return self.tokenizer_class.from_pretrained(self.tmpdirname, **kwargs)

    # def get_rust_tokenizer(self, **kwargs) -> PreTrainedTokenizerFast:
    #     return self.rust_tokenizer_class.from_pretrained(self.tmpdirname, **kwargs)