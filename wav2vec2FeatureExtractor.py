import torch 
import numpy as np

from torch import nn 
from typing import List, Optional, Union, Dict
from utils import PaddingStrategy, TensorType, BatchFeature, to_numpy


class FeatureExtractionMixin(): 
    _auto_class = None
    def __init__(
            self, 
            **kwargs
        ): 
        self._processor_class = kwargs.pop("processor_class", None)
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                raise err
            
    def _set_processor_class(self, processor_class: str): 
        self._processor_class = processor_class
    


class SequenceFeatureExtractor(FeatureExtractionMixin):
    def __init__(
        self,
        feature_size: int,
        sampling_rate: int,
        padding_value: float, 
        **kwargs,
    ):
        self.feature_size = feature_size
        self.sampling_rate = sampling_rate
        self.padding_value = padding_value
        self.padding_side = kwargs.pop("padding_side", "right")
        self.return_attion_mask = kwargs.pop("return_attention_mask", True)

        super().__init__(**kwargs)

    def pad(
        self,
        processed_features = Union[
            BatchFeature,
            List[BatchFeature],
            Dict[str, BatchFeature],
            Dict[str, List[BaseException]],
            List[Dict[str, BaseException]],
        ],
        padding: Union[bool, str, PaddingStrategy] = True,
        max_length: Optional[int] = None, 
        truncation: bool = False,
        pad_to_multiple_of: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
    ) -> BatchFeature:
        
        if isinstance(processed_features, (list, tuple)) and isinstance(processed_features[0], (dict, BatchFeature)):
            processed_features = { 
                key: [example[key] for example in processed_features] for key in processed_features[0].keys()
            }

        # if self.model_input_names[0] not in processesd_features[self.model_input_names[0]]:
        #     raise ValueError("you should supply an instance of transformers.BatchFeature")
        
        required_input = processed_features[self.model_input_names[0]]
        return_attention_mask = (
            return_attention_mask if return_attention_mask is not None else self.return_attion_mask
        )
        
        if len(required_input) == 0:
            if return_attention_mask:
                processed_features["attention_mask"] = []
            return processed_features

        first_element = required_input[0]
        if isinstance(first_element, (list, tuple)):
            index = 0 
            while len(required_input[index]) == 0: 
                index +=1 
            if index < len(required_input):
                first_element = required_input[index][0]

        # if return_tensors is None:
        #     if is_tf_tensor(first_element):
        #         return_tensors = "tf"
        #     elif is_torch_tensor(first_element):
        #         return_tensors = "pt"
        #     elif isinstance(first_element, (int, float, list, tuple, np.ndarray)):
        #         return_tensors = "np"
        #     else:
        #         raise ValueError(
        #             f"type of {first_element} unknown: {type(first_element)}. "
        #             "Should be one of a python, numpy, pytorch or tensorflow object."
        #         )

        for key, value in processed_features.items():
            if isinstance(value[0], (int, float)):
                processed_features[key] = to_numpy(value)
            else:
                processed_features[key] = [to_numpy(v) for v in value]

class Wav2Vec2FeatureExtractor(SequenceFeatureExtractor):
    
    model_input_names = ["input_values", "attention_mask"]

    def __init__(
        self,
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        return_attention_mask=False,
        do_normalize=True,
        **kwargs,
    ):
        super().__init__(
                feature_size=feature_size,
                sampling_rate=sampling_rate,
                padding_value=padding_value,
                **kwargs,
        )
        self.return_attention_mask = return_attention_mask
        self.do_normalize = do_normalize


        @staticmethod
        def zero_mean_unit_var_norm(
            input_values: List[np.ndarray], 
            attention_mask: List[np.ndarray], 
            padding_value: float = 0.0,
        ) -> List[np.ndarray]:

            if attention_mask is not None: 
                attention_mask = np.array(attention_mask, np.int32)
                normed_input_values = [] 
                for vector, length in zip(input_values, attention_mask.sum(-1)): 
                    normed_slice = (vector - vector[:length].mean()) / np.sqrt(vector[:length].var() + 1e-7)
                    if length < normed_slice.shape[0]: 
                        normed_slice[length:] = padding_value
                    normed_input_values.append(normed_slice)
            else: 
                normed_input_values = [(x - x.mean()) / np.sqrt(x.var() + 1e-7) for x in input_values]
            return normed_input_values
                    
    def __call__(
        self, 
        raw_speech: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]], 
        padding: Union[bool, str, PaddingStrategy] = False,
        max_length: Optional[int] = None, 
        trnction: bool = False, 
        pad_to_multiple_of: Optional[int] = None, 
        return_attention_mask: Optional[bool] = None, 
        return_tensors: Optional[Union[str, TensorType]] = None, 
        sampling_rate: Optional[int] = None, 
        **kwargs,
    ) -> BatchFeature: 
        
        if sampling_rate is not None: 
            if sampling_rate != self.sampling_rate: 
                raise ValueError(" Sampling rate is not equal 16000")
            
        else: 
            pass 

        is_batched_numpy = isinstance(raw_speech, np.ndarray) and len(raw_speech.shape) > 1 
        if is_batched_numpy and len(raw_speech.shape) > 2: 
            raise ValueError("Only mono-channel audio is supported for in put")
        is_batched = is_batched_numpy or (isinstance(raw_speech, (list, tuple)) and (isinstance(raw_speech[0]), (np.ndarray, tuple, list)))

        if not is_batched: 
            raw_speech = [raw_speech]
        
        encoder_inputs = BatchFeature({"input_values": raw_speech})

        print(encoder_inputs)
        # padded_values = self.pad()