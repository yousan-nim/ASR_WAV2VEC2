import os 
import json
import tempfile
import unittest

from wav2vec2CTCTokenizer import Wav2Vec2CTCTokenizer, VOCAB_FILES_NAMES






class Wav2Vec2CTCTokenizerTest(unittest.TestCase): 
    tokenizer_class = Wav2Vec2CTCTokenizer

    def setUp(self): 
        super().setUp() 

        vocab = "<pad> <s> </s> <unk> | E T A O N I H S R D L U M W C F G Y P B V K ' X J Q Z".split(" ")
        vocab_tokens = dict(zip(vocab, range(len(vocab))))

        self.special_tokens_map = {"pad_token": "<pad>", "unk_token": "<unk>", "bos_token": "<s>", "eos_token": "</s>"}

        self.tmpdirname = tempfile.mkdtemp()
        self.vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["vocab_file"])

        with open(self.vocab_file, "w", encoding="utf-8") as fp: 
            fp.write(json.dump(vocab_tokens) + "\n")

    def get_tokenizer(self, **kwargs): 
        kwargs.update(self.special_tokens_map)
        return Wav2Vec2CTCTokenizer.from_pretrained(self.tmpdirname, **kwargs)
















tokens = Wav2Vec2CTCTokenizer.from_pretrained('./tokens')
print(tokens)














