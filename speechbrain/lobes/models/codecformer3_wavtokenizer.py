import dac
print("descript audio codec v",dac.__version__)
from audiotools import AudioSignal
import torchaudio.transforms as T
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import collections
import warnings
import pyloudnorm
import random
import numpy as np
from torch.nn.utils import weight_norm
from dac.nn.layers import Snake1d
from speechbrain.lobes.models.transformer.Transformer import TransformerDecoder
from speechbrain.lobes.models.transformer.Transformer import PositionalEncoding
from speechbrain.lobes.models.wavtokenizer.decoder.pretrained import WavTokenizer

class WavTokenizerWrapper:
    '''
    Wrapper model for WavTokenizer
    '''
    def __init__(self, input_sample_rate=8000, model_config_path=None, model_ckpt_path=None, tokenizer_sample_rate=24000, Freeze=True):
        '''
        input_sample_rate: defaults to 8000 as expected file input
        model_config_path: Path to the config file for WavTokenizer
        model_ckpt_path: Path to the checkpoint file for WavTokenizer
        tokenizer_sample_rate: defaults to 24000. Specify if using a model with a different sample rate.
        '''
        super(WavTokenizerWrapper, self).__init__()
        self.input_sample_rate = input_sample_rate
        self.tokenizer_sample_rate = tokenizer_sample_rate

        if model_config_path is None or model_ckpt_path is None:
            raise ValueError("Please provide both the model config and checkpoint paths.")

        self.model = WavTokenizer.from_pretrained0802(model_config_path, model_ckpt_path)

        self.dac_sampler = T.Resample(input_sample_rate, tokenizer_sample_rate)
        self.org_sampler = T.Resample(tokenizer_sample_rate, input_sample_rate)

        def count_all_parameters(model): return sum(p.numel() for p in model.parameters())
        def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)

        if Freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            
            print(f'Model frozen with {count_parameters(self.model)/1000000:.2f}M trainable parameters remaining')
            print(f'Model has {count_all_parameters(self.model)/1000000:.2f}M parameters in total')
        else:
            print(f'Model with {count_all_parameters(self.model)/1000000:.2f}M trainable parameters loaded')

    def resample_audio(self, x, condition):
        '''
        Resample the audio according to the condition.
        condition: "tokenizer" to set the sampling rate to the tokenizer's rate
                   "org" to set the sampling rate back to the original rate
        '''
        device = x.device

        assert len(x.shape) == 3, "Input tensor must have 3 dimensions [Batch, Channels, Time]"
        B, C, T = x.shape
        assert C == 1, "Input tensor must be mono-channel [Batch, 1, Time]"

        if condition == "tokenizer":
            x_resamp = self.dac_sampler(x)
        elif condition == "org":
            x_resamp = self.org_sampler(x)
        else:
            raise ValueError("Unknown condition for resampling: {}".format(condition))
        
        x_resamp = x_resamp / torch.max(x_resamp.abs(), dim=2, keepdim=True)[0]

        return x_resamp.to(device)

    def get_encoded_features(self, x):
        '''
        x should be a torch tensor with dimensions [Batch, Channel, Time]
        '''
        original_length = x.shape[-1]

        # Resample the audio to the tokenizer's sample rate
        x = self.resample_audio(x, "tokenizer")
    
        # Remove channel dimensions for the audio data tensor
        x = x.squeeze(1)
        
        # Generate features and discrete codes
        bandwidth_id = torch.tensor([0]).to(x.device)
        features, _, _ = self.model.feature_extractor(x, bandwidth_id=bandwidth_id)
        return features, original_length
    
    def get_quantized_features(self, x, bandwidth_id=None):
        '''
        Expects input [B, D, T] where D is the encoded continuous representation of input.
        Returns quantized features, codes, latents, commitment loss, and codebook loss in the same format as DACWrapper.
        '''
        if bandwidth_id is None:
            bandwidth_id = torch.tensor([0]).to(x.device)

        # Ensure the tensor has 3 dimensions [Batch, Channels, Time]
        if x.ndim != 3:
            raise ValueError(f"Expected input to have 3 dimensions [Batch, Channels, Time], but got {x.ndim} dimensions.")

        # Perform the quantization directly on the encoded features
        q_res = self.model.feature_extractor.encodec.quantizer(
            x, 
            frame_rate=self.model.feature_extractor.frame_rate, 
            bandwidth=self.model.feature_extractor.bandwidths[bandwidth_id]
        )

        # Extract necessary outputs
        quantized = q_res.quantized
        codes = q_res.codes
        latents = x  # The input x itself is the latent representation after encoding
        commit_loss = q_res.penalty

        # Placeholder for codebook_loss (not directly available, could be None)
        codebook_loss = None

        # Return the outputs in the expected format
        return quantized, codes, latents, commit_loss, codebook_loss

    def get_decoded_signal(self, features, original_length):
        '''
        Decodes the features back to the audio signal.
        '''
        # Decode the features to get the waveform
        bandwidth_id = torch.tensor([0]).to(features.device)

        x = self.model.backbone(features, bandwidth_id=bandwidth_id)
        y_hat = self.model.head(x)

        # Ensure the output has three dimensions [Batch, Channels, Time] before resampling
        if y_hat.ndim == 2:
            y_hat = y_hat.unsqueeze(1)  # Add a channel dimension if it's missing

        # Resample the decoded signal back to the original sampling rate
        y_hat_resampled = self.resample_audio(y_hat, "org")

        # Ensure the output shape matches the original length
        if y_hat_resampled.shape[-1] != original_length:
            T_origin = original_length
            T_est = y_hat_resampled.shape[-1]

            if T_origin > T_est:
                y_hat_resampled = F.pad(y_hat_resampled, (0, T_origin - T_est))
            else:
                y_hat_resampled = y_hat_resampled[:, :, :T_origin]

        return y_hat_resampled