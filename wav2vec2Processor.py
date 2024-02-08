
from contextlib import contextmanager

from wav2vec2FeatureExtractor import Wav2Vec2FeatureExtractor
from wav2vec2CTCTokenizer import Wav2Vec2CTCTokenizer


class Wav2Vec2Processor:
    def __init__(self, feature_extractor, tokenizer):
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
