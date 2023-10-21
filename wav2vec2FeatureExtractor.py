import os
import torch 
import json
import numpy as np

from torch import nn 
from typing import List, Optional, Union, Dict, Tuple, Any
from utils import (FEATURE_EXTRACTOR_NAME,
                   PaddingStrategy, 
                   TensorType, 
                   BatchFeature, 
                   _is_torch, 
                   to_numpy, 
                   is_torch_available, 
                   to_py_obj, 
                   cached_path, 
                   is_offline_mode, 
                   is_remote_url
                   )

class FeatureExtractionMixin(): 
    _auto_class = None
    def __init__(
        self, 
        **kwargs
    ): 
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                raise err
            
    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ):
        feature_extractor_dict, kwargs = cls.get_feature_extractor_dict(pretrained_model_name_or_path, **kwargs)

        return cls.from_dict(feature_extractor_dict, **kwargs)
    
    @classmethod
    def get_feature_extractor_dict(
        cls, 
        pretrained_model_name_or_path: Union[str, os.PathLike], 
        **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        use_auth_token = kwargs.pop("use_auth_token", None)
        local_files_only = kwargs.pop("local_files_only", False)
        revision = kwargs.pop("revision", None)

        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)

        user_agent = {"file_type": "feature extractor", "from_auto_class": from_auto_class}
        if from_pipeline is not None:
            user_agent["using_pipeline"] = from_pipeline

        if is_offline_mode() and not local_files_only:
            local_files_only = True

        pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        if os.path.isdir(pretrained_model_name_or_path):
            feature_extractor_file = os.path.join(pretrained_model_name_or_path, FEATURE_EXTRACTOR_NAME)
        elif os.path.isfile(pretrained_model_name_or_path) or is_remote_url(pretrained_model_name_or_path):
            feature_extractor_file = pretrained_model_name_or_path

        try:
            # Load from URL or cache if already cached
            resolved_feature_extractor_file = cached_path(
                feature_extractor_file,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                user_agent=user_agent,
            )
            # Load feature_extractor dict
            with open(resolved_feature_extractor_file, "r", encoding="utf-8") as reader:
                text = reader.read()
            feature_extractor_dict = json.loads(text)

        except json.JSONDecodeError:
            msg = (
                f"Couldn't reach server at '{feature_extractor_file}' to download feature extractor configuration file or "
                "feature extractor configuration file is not a valid JSON file. "
                f"Please check network or file content here: {resolved_feature_extractor_file}."
            )
            raise EnvironmentError(msg)

        return feature_extractor_dict, kwargs


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
        
        if not required_input: 
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

        if not isinstance(first_element, (float, int, list, tuple)):
            if is_torch_available() and _is_torch(first_element):
                return_tensors = "pt" if return_tensors is None else return_tensors
            elif isinstance(first_element, np.ndarray):
                return_tensors = "np" if return_tensors is None else return_tensors

            for key, value in processed_features.items():
                    processed_features[key] = to_py_obj(value)


        padding_strategy, max_length, _ = self._get_padding_strategies(padding=padding, max_length=max_length)

        required_input = processed_features[self.model_input_names[0]]

        # for key, value in processed_features.items():
        #     if isinstance(value[0], (int, float)):
        #         processed_features[key] = to_numpy(value)
        #     else:
        #         processed_features[key] = [to_numpy(v) for v in value]




    def _get_padding_strategies(
        self,
        padding=False, 
        max_length=None,
        padd_to_multiple=None,
        **kwargs
    ):
        if padding is not False: 
            if padding is True:
                padding_strategy = PaddingStrategy.LONGEST
            elif not isinstance(padding, PaddingStrategy): 
                padding_strategy = PaddingStrategy(padding)
            elif isinstance(padding, PaddingStrategy):
                padding_strategy = padding
        else:
            padding_strategy = PaddingStrategy.DO_NOT_PAD

        if max_length is None: 
            if padding_strategy == PaddingStrategy.MAX_LENGTH:
                raise ValueError(
                    f"When setting ``padding={PaddingStrategy.MAX_LENGTH}``, make sure that" f" max_length is defined"
                )
                
        if padding_strategy != PaddingStrategy.DO_NOT_PAD and (self.padding_value is None):
            raise ValueError(
            "Asking to pad but the feature_extractor does not have a padding value. "
            "Please select a value to use as `padding_value`. For example: `feature_extractor.padding_value = 0.0`."
        )

        return padding_strategy, max_length, kwargs

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
    ) -> List[np.ndarray]:
        return [(x - np.mean(x)) / np.sqrt(np.var(x) + 1e-5) for x in input_values]
                    
    def __call__(
        self, 
        raw_speech: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]], 
        padding: Union[bool, str, PaddingStrategy] = False,
        max_length: Optional[int] = None, 
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

        is_batched = bool(
            isinstance(raw_speech, (list, tuple))
            and (isinstance(raw_speech[0], np.ndarray) or isinstance(raw_speech[0], (tuple, list)))
        )
        if is_batched and not isinstance(raw_speech[0], np.ndarray):
            raw_speech = [np.asarray(speech) for speech in raw_speech]
        elif not is_batched and not isinstance(raw_speech, np.ndarray):
            raw_speech = np.asarray(raw_speech)

        if not is_batched: 
            raw_speech = [raw_speech]

        if self.do_normalize:
            raw_speech = self.zero_mean_unit_var_norm(raw_speech)
        
        encoded_inputs = BatchFeature({"input_values": raw_speech})

        padded_inputs = self.pad(
            encoded_inputs,
            padding=padding,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
            return_tensors=return_tensors,
        )

        return padded_inputs