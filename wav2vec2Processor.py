import torch 
from torch import nn

class Wav2Vec2Processor(nn.Module):
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