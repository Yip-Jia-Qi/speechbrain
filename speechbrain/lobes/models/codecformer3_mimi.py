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

from huggingface_hub import hf_hub_download
from moshi.models import loaders, LMGen
import math

def load_pretrained_model(
        init_param: str,
        model: torch.nn.Module,
        ignore_init_mismatch: bool,
        map_location: str = "cpu",
    ):
    """Load a model state and set it to the model.

    Args:
        init_param: <file_path>:<src_key>:<dst_key>:<exclude_Keys>

    Examples:
        >>> load_pretrained_model("somewhere/model.pth", model)
        >>> load_pretrained_model("somewhere/model.pth:decoder:decoder", model)
        >>> load_pretrained_model("somewhere/model.pth:decoder:decoder:", model)
        >>> load_pretrained_model(
        ...     "somewhere/model.pth:decoder:decoder:decoder.embed", model
        ... )
        >>> load_pretrained_model("somewhere/decoder.pth::decoder", model)
    """
    sps = init_param.split(":", 4)
    if len(sps) == 4:
        path, src_key, dst_key, excludes = sps
    elif len(sps) == 3:
        path, src_key, dst_key = sps
        excludes = None
    elif len(sps) == 2:
        path, src_key = sps
        dst_key, excludes = None, None
    else:
        (path,) = sps
        src_key, dst_key, excludes = None, None, None
    if src_key == "":
        src_key = None
    if dst_key == "":
        dst_key = None

    if dst_key is None:
        obj = model
    else:
        def get_attr(obj: Any, key: str):
            """Get an nested attribute.

            >>> class A(torch.nn.Module):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.linear = torch.nn.Linear(10, 10)
            >>> a = A()
            >>> assert A.linear.weight is get_attr(A, 'linear.weight')

            """
            if key.strip() == "":
                return obj
            for k in key.split("."):
                obj = getattr(obj, k)
            return obj

        obj = get_attr(model, dst_key)

    src_state = torch.load(path, map_location=map_location)
    if excludes is not None:
        for e in excludes.split(","):
            src_state = {k: v for k, v in src_state.items() if not k.startswith(e)}

    if src_key is not None:
        src_state = {
            k[len(src_key) + 1 :]: v
            for k, v in src_state.items()
            if k.startswith(src_key)
        }

    dst_state = obj.state_dict()
    if ignore_init_mismatch:
        src_state = filter_state_dict(dst_state, src_state)
    dst_state.update(src_state)
    obj.load_state_dict(dst_state)

class MimiWrapper():
    '''
    Wrapper model for Mimi Codec
    '''
    def __init__(self, Freeze=True):
        '''
        mimi_sample_rate: 24000. Please resample your audio to this rate before any training.
        '''
        super(MimiWrapper, self).__init__()
        
        mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)   
        self.model = loaders.get_mimi(mimi_weight, device='cuda')
        self.model.set_num_codebooks(8)

        def count_all_parameters(model): return sum(p.numel() for p in model.parameters())
        def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)

        if Freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            
            print(f'Model frozen with {count_parameters(self.model)/1000000:.2f}M trainable parameters remaining')
            print(f'Model has {count_all_parameters(self.model)/1000000:.2f}M parameters in total')

        else:
            print(f'Model with {count_parameters(self.model)/1000000:.2f}M trainable parameters loaded')

    def full_pipeline(self, x):
        '''
        Walkthrough tutorial of the entire mimi codec pipeline 
        from input speech x [B, 1, L] to reconstructed speech x_hat [B, 1, L]
        '''
        
        # [B, D, 2S], L
        encode_transformer_emb, original_length = self.get_encoded_features(x)
        
        # [B, D, S]
        encode_transformer_emb = self.reduce_framerate_for_vq(encode_transformer_emb)

        # [B, D/2, S]
        semantic_latent = self.get_latent_before_semantic_vq(encode_transformer_emb)
        
        # [B, D/2, S]
        acoustic_latent = self.get_latent_before_acoustic_vq(encode_transformer_emb)

        # [B, 1, S]
        semantic_codes = self.get_semantic_codes_from_latent(semantic_latent)
        
        # [B, 7, S]
        acoustic_codes = self.get_acoustic_codes_from_latent(acoustic_latent)
        
        # [B, 8, S]
        combined_codes = torch.cat([semantic_codes, acoustic_codes], dim=1)

        # [B, D, S]
        quantized_emb = self.get_quantized_features(combined_codes)
        
        # [B, D, 2S]
        quantized_emb = self.increase_framerate_for_decode(quantized_emb)
        
        # [B, 1, L]
        x_hat = self.get_decoded_signal(quantized_emb, original_length)
        return x_hat 

    def get_encoded_features(self, x):
        '''
        x should be the torch tensor as per the data loader
        Expects x to be of dimensions [Batch, Channel, Time]
        '''
        #keep original lengths from pre resampling
        original_length = x.shape[-1]

        #Now to perform the encoding
        with self.model._context_for_encoder_decoder:
            encode_emb = self.model.encoder(x)

        (encode_transformer_emb,) = self.model.encoder_transformer(encode_emb)
            
        return encode_transformer_emb, original_length #the codec outputs a different length due to the masking
    
    def reduce_framerate_for_vq(self, x):
        encode_emb_to_framerate = self.model._to_framerate(x)
        return encode_emb_to_framerate

    def get_latent_before_semantic_vq(self, x):
        latent = self.model.quantizer.rvq_first.input_proj(x)
        return latent

    def get_semantic_codes_from_latent(self, x):
        n_q = self.model.quantizer.rvq_first.n_q
        codes = self.model.quantizer.rvq_first.vq.encode(x, n_q=n_q)
        codes = codes.transpose(0, 1)
        return codes

    def get_latent_before_acoustic_vq(self, x):
        latent = self.model.quantizer.rvq_rest.input_proj(x)
        return latent

    def get_acoustic_codes_from_latent(self, x):
        n_q = self.model.quantizer.rvq_rest.n_q
        codes = self.model.quantizer.rvq_rest.vq.encode(x, n_q=n_q)
        codes = codes.transpose(0, 1)
        return codes

    # equavelent to running:
    # get_latent_before_semantic_vq
    # get_semantic_codes_from_latent
    # get_latent_before_acoustic_vq
    # get_acoustic_codes_from_latent
    def get_codes(self, x):
        semantic_codes = self.model.quantizer.rvq_first.encode(x) # [B, 1, S]
        acoustic_codes = self.model.quantizer.rvq_rest.encode(x) # [B, 7, S]
        return semantic_codes, acoustic_codes      
    
    def get_quantized_features(self, x):
        decode_codes = self.model.decode_latent(x)
        return decode_codes
    
    def increase_framerate_for_decode(self, x):
        decode_emb_to_encoder_framerate = self.model._to_encoder_framerate(x)
        return decode_emb_to_encoder_framerate

    def get_decoded_signal(self, x, original_length):
        (decode_transformer_emb,) = self.model.decoder_transformer(x)
        with self.model._context_for_encoder_decoder:
            decode_emb = self.model.decoder(decode_transformer_emb)
        
        # T might have changed due to model. If so, fix it here
        if decode_emb.shape[-1] != original_length:
            T_origin = original_length
            T_est = decode_emb.shape[-1]
            
            if T_origin > T_est:
                decode_emb = F.pad(decode_emb, (0, T_origin - T_est))
            else:
                decode_emb = decode_emb[:, :, :T_origin]
        return decode_emb


class simpleSeparator2(nn.Module):
    def __init__(self, num_spks, channels, block, block_channels, Freeze=False):
        super(simpleSeparator2, self).__init__()
        self.num_spks = num_spks 
        self.channels = channels #this is dependent on the dac model
        self.block = block #this should be a seq2seq model with identical input and output sizes
        self.ch_down = nn.Conv1d(channels, block_channels,1,bias=False)
        self.ch_up = nn.Conv1d(block_channels, channels,1,bias=False)
        #self.time_mix = nn.Conv1d(channels,channels,1,bias=False)
        self.masker = weight_norm(nn.Conv1d(channels, channels*num_spks, 1, bias=False))

        self.activation = Snake1d(channels) #nn.Tanh() #nn.ReLU() #Snake1d(channels)
        # gated output layer
        self.output = nn.Sequential(
            nn.Conv1d(channels, channels, 1), Snake1d(channels) #nn.Tanh() #, Snake1d(channels)#
        )
        self.output_gate = nn.Sequential(
            nn.Conv1d(channels, channels, 1), nn.Sigmoid()
        )

    def forward(self,x):
        
        x = self.ch_down(x)
        #[B,N,L]
        x = x.permute(0,2,1)
        #[B,L,N]
        x_b = self.block(x)
        #[B,L,N]
        x_b = x_b.permute(0,2,1)
        #[B,N,L]
        x = self.ch_up(x_b)

        B, N, L = x.shape
        masks = self.masker(x)
        
        #[B,N*num_spks,L]
        masks = masks.view(B*self.num_spks,-1,L)
        
        #[B*num_spks, N, L]
        x = self.output(masks) * self.output_gate(masks)
        x = self.activation(x)

        #[B*num_spks, N, L]
        _, N, L = x.shape
        x = x.view(B, self.num_spks, N, L)
        
        # [B, spks, N, L]
        x = x.transpose(0,1)
        # [spks, B, N, L]

        return x

