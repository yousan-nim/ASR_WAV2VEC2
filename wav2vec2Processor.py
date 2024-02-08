import torch 
from torch import nn
from contextlib import contextmanager

from wav2vec2FeatureExtractor import Wav2Vec2FeatureExtractor 
from wav2vec2CTCTokenizer import Wav2Vec2CTCTokenizer

class Wav2Vec2Processor:
    def __init__(
            self, 
            feature_extractor, 
            tokenizer
        ):
        if not isinstance(feature_extractor, Wav2Vec2FeatureExtractor):
            raise ValueError(
                f"`feature_extractor` has to be of type {Wav2Vec2FeatureExtractor.__class__}, but is {type(feature_extractor)}"
            )
        if not isinstance(tokenizer, Wav2Vec2CTCTokenizer):
            raise ValueError(
                f"`tokenizer` has to be of type {Wav2Vec2CTCTokenizer.__class__}, but is {type(tokenizer)}"
            )

        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.current_processor = self.feature_extractor

    def save_pretrained(self, save_directory):
        self.feature_extractor.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(pretrained_model_name_or_path, **kwargs)
        tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)

        return cls(feature_extractor=feature_extractor, tokenizer=tokenizer)

    def __call__(self, *args, **kwargs):
        
        return self.current_processor(*args, **kwargs)

    def pad(self, *args, **kwargs):
       
        return self.current_processor.pad(*args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        
        return self.tokenizer.decode(*args, **kwargs)

    @contextmanager
    def as_target_processor(self):
        
        self.current_processor = self.tokenizer
        yield
        self.current_processor = self.feature_extractor

    feature_extractor_class = "wav2Vec2FeatureExxtractor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, feature_etractor, tokenizer):
        super().__init__(feature_etractor, tokenizer)
        self.currebt_processor = self.feature_etractor
        self._in_target_context_managet = False

    @classmethod
    def from_pretrain():
        pass

    def __call__(self, *args, **kwargs):
        if self._in_target_context_managet: 
            return self.currebt_processor(*args, **kwargs)
        
        audio = kwargs.pop("audio", None)
        sampling_rate = kwargs.pop("sampling_rate", None)
        text = kwargs.pop("text", None)
        
        if len(args) > 0:
            audio = args[0]
            args = args[1:]
        
        if audio is not None:
            inputs = self.feature_extractor(audio, *args, sampling_rate=sampling_rate, **kwargs)
        if text is not None:
            encodings = self.tokenizer(text, **kwargs)

        if text is None:
            return inputs
        elif audio is None:
            return encodings
        else:
            inputs["labels"] = encodings["input_ids"]
            return inputs