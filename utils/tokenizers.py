# Generated content DO NOT EDIT
class AddedToken:
    def __init__(self, content, single_word=False, lstrip=False, rstrip=False, normalized=True):
        pass

    @property
    def content(self):
        pass

    @property
    def lstrip(self):
        pass

    @property
    def normalized(self):
        pass

    @property
    def rstrip(self):
        pass
    
    @property
    def single_word(self):
        pass

class Encoding:
    @property
    def attention_mask(self):
        pass

    def char_to_token(self, char_pos, sequence_index=0):
        pass

    def char_to_word(self, char_pos, sequence_index=0):
        pass

    @property
    def ids(self):
        pass

    @staticmethod
    def merge(encodings, growing_offsets=True):
        pass

    @property
    def n_sequences(self):
        pass

    @property
    def offsets(self):
        pass

    @property
    def overflowing(self):
        pass

    def pad(self, length, direction="right", pad_id=0, pad_type_id=0, pad_token="[PAD]"):
        pass

    @property
    def sequence_ids(self):
        pass

    def set_sequence_id(self, sequence_id):
        pass

    @property
    def special_tokens_mask(self):
        pass

    def token_to_chars(self, token_index):
        pass

    def token_to_sequence(self, token_index):
        pass

    def token_to_word(self, token_index):
        pass

    @property
    def tokens(self):
        pass

    def truncate(self, max_length, stride=0, direction="right"):
        pass

    @property
    def type_ids(self):
        pass

    @property
    def word_ids(self):
        pass

    def word_to_chars(self, word_index, sequence_index=0):
        pass

    def word_to_tokens(self, word_index, sequence_index=0):
        pass

    @property
    def words(self):
        pass

class NormalizedString:
    def append(self, s):
        pass

    def clear(self):
        pass

    def filter(self, func):

        pass
    def for_each(self, func):

        pass
    def lowercase(self):
        pass

    def lstrip(self):
        pass
    
    def map(self, func):
        pass

    def nfc(self):
        pass

    def nfd(self):
        pass

    def nfkc(self):
        pass

    def nfkd(self):
        pass

    @property
    def normalized(self):
        pass

    def prepend(self, s):
        pass

    def replace(self, pattern, content):
        pass

    def rstrip(self):
        pass

    def slice(self, range):
        pass

    def split(self, pattern, behavior):
        pass

    def strip(self):
        pass

    def uppercase(self):
        pass

class PreTokenizedString:
    def __init__(self, sequence):
        pass

    def get_splits(self, offset_referential="original", offset_type="char"):
        pass

    def normalize(self, func):
        pass

    def split(self, func):
        pass

    def to_encoding(self, type_id=0, word_idx=None):
        pass

    def tokenize(self, func):
        pass

class Regex:
    def __init__(self, pattern):
        pass

class Token:
    pass

class Tokenizer:
    def __init__(self, model):
        pass

    def add_special_tokens(self, tokens):
        pass

    def add_tokens(self, tokens):
        pass
    def decode(self, ids, skip_special_tokens=True):
        pass

    def decode_batch(self, sequences, skip_special_tokens=True):
        pass

    @property
    def decoder(self):
        pass

    def enable_padding(
        self, direction="right", pad_id=0, pad_type_id=0, pad_token="[PAD]", length=None, pad_to_multiple_of=None
    ):
        pass

    def enable_truncation(self, max_length, stride=0, strategy="longest_first", direction="right"):
        pass

    def encode(self, sequence, pair=None, is_pretokenized=False, add_special_tokens=True):
        pass

    def encode_batch(self, input, is_pretokenized=False, add_special_tokens=True):
        pass

    @staticmethod
    def from_buffer(buffer):
        pass

    @staticmethod
    def from_file(path):
        pass

    @staticmethod
    def from_pretrained(identifier, revision="main", auth_token=None):
        pass

    @staticmethod
    def from_str(json):
        pass

    def get_vocab(self, with_added_tokens=True):
        pass

    def get_vocab_size(self, with_added_tokens=True):
        pass

    def id_to_token(self, id):
        pass

    @property
    def model(self):
        pass

    def no_padding(self):
        pass

    def no_truncation(self):
        pass

    @property
    def normalizer(self):
        pass

    def num_special_tokens_to_add(self, is_pair):
        pass

    @property
    def padding(self):
        pass

    def post_process(self, encoding, pair=None, add_special_tokens=True):
        pass

    @property
    def post_processor(self):
        pass

    @property
    def pre_tokenizer(self):
        pass

    def save(self, path, pretty=True):
        pass

    def to_str(self, pretty=False):
        pass

    def token_to_id(self, token):
        pass

    def train(self, files, trainer=None):
        pass

    def train_from_iterator(self, iterator, trainer=None, length=None):
        pass

    @property
    def truncation(self):
        pass
