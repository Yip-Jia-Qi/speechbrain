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
import yaml
from torch.nn.utils import weight_norm
from dac.nn.layers import Snake1d
from speechbrain.lobes.models.transformer.Transformer import TransformerDecoder
from speechbrain.lobes.models.transformer.Transformer import PositionalEncoding

from speechbrain.lobes.models.gan_codec.soundstream import SoundStream

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

class DACWrapper():
    '''
    Wrapper model for Descript Audio Codec
    '''
    def __init__(self, input_sample_rate=8000, DAC_model_path = None, DAC_sample_rate = 16000, Freeze=True, epoch=120):
        '''
        input_sample_rate: defaults to 8000 as it common in speech separation
        Model Path: Please provde the model path to the DAC model, otherwise the 16KHz model will be automatically downloaded.
        DAC_sample_rate: defaults to 16000. If using a DAC model other than the 16khz model, please specify this number.
        '''
        super(DACWrapper, self).__init__()
        self.input_sample_rate = input_sample_rate
        self.DAC_sample_rate = DAC_sample_rate
        if DAC_model_path == None:
            raise Exception("model path needs to be provided")
            model_path = dac.utils.download(model_type="16khz")
            self.model = dac.DAC.load(model_path)
        else:
            with open(f'{DAC_model_path}/config.yaml') as f:
                # use safe_load instead load
                dataMap = yaml.safe_load(f)
            
            # generator_params = dataMap['codec_conf']['generator_params']

            # print(dataMap)
            # raise Exception
            # print(dataMap['codec_conf']['sampling_rate'])
            self.model = SoundStream(**dataMap['codec_conf'])

            

            model_path = f'{DAC_model_path}/{epoch}epoch.pth'
            # print(model_path)
            load_pretrained_model(
                    init_param = f'{model_path}:codec',
                    model = self.model,
                    ignore_init_mismatch = False,
                    map_location = "cpu",
                )
            print("ptmodel loaded")

            self.model = self.model.generator
            
        
        self.dac_sampler = T.Resample(input_sample_rate, DAC_sample_rate)
        self.org_sampler = T.Resample(DAC_sample_rate, input_sample_rate)

        def count_all_parameters(model): return sum(p.numel() for p in model.parameters())
        def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)

        if Freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            
            print(f'Model frozen with {count_parameters(self.model)/1000000:.2f}M trainable parameters remaining')
            print(f'Model has {count_all_parameters(self.model)/1000000:.2f}M parameters in total')

        else:
            print(f'Model with {count_parameters(self.model)/1000000:.2f}M trainable parameters loaded')

    def resample_audio(self, x, condition):
        '''
        torchaudio resample function used here only requires last dimension to be time.
        condition: "dac" to set the sampling rate to the DAC sampling rate
                   "org" to set the sampling rate back to the original sampling rate

        it sucks that i have to go to cpu for this. need to think how i can make this stay in gpu
        '''
        # get device
        device = x.device

        # Implement some checks on the input
        assert len(x.shape) == 3
        B, C, T = x.shape
        assert C == 1 #model should only be handling single channel

        # Resamples the audio from the input rate to the dac model's rate
        if condition == "dac":
            x_resamp = self.dac_sampler(x)
        elif condition == "org":
            x_resamp = self.org_sampler(x)
        
        # normalize the resampled audio, otherwise we will run into clipping issues
        x_resamp = x_resamp / torch.max(x_resamp.abs(),dim=2,keepdim=True)[0]

        return x_resamp.to(device)

    def get_encoded_features(self, x):
        '''
        x should be the torch tensor as per the data loader
        Expects x to be of dimensions [Batch, Channel, Time]
        '''
        #keep original lengths from pre resampling
        original_length = x.shape[-1]

        #make the input into the desired sampling rate
        x = self.resample_audio(x, "dac")

        #Now to perform the encoding
        # x = self.model.preprocess(x, self.DAC_sample_rate) #not neeeded for espnet model
        x_enc = self.model.encoder(x)

        return x_enc, original_length #the codec outputs a different length due to the masking
    
    def get_quantized_features(self, x):
        '''
        expects input [B, D, T] where D is the Quantized continuous representation of input
        '''

        # x_qnt, codes_hat, latents_hat, commitment_loss_hat, codebook_loss_hat = self.model.quantizer(x, None) #for differently for espnet

        x_qnt, codes_hat, _, commitment_loss_hat = self.model.quantizer(x, self.DAC_sample_rate)
        latents_hat = None
        codebook_loss_hat = None
        
        return x_qnt, codes_hat, latents_hat, commitment_loss_hat, codebook_loss_hat

    def get_decoded_signal(self, x, original_length):
        '''
        expects input [B, D, T] where D is the Quantized continuous representation of input
        original length is the original length of the input audio
        '''
        y_hat = self.model.decoder(x)

        # out = y_hat[...,:original_length]
        # out = self.resample_audio(out, "org")
        out = self.resample_audio(y_hat, "org")
        
        # T might have changed due to model. If so, fix it here
        if out.shape[-1] != original_length:
            T_origin = original_length
            T_est = out.shape[-1]
            
            if T_origin > T_est:
                out = F.pad(out, (0, T_origin - T_est))
            else:
                out = out[:, :, :T_origin]
        return out

class simpleSeparator(nn.Module):
    def __init__(self, num_spks, channels, block):
        super(simpleSeparator, self).__init__()
        self.num_spks = num_spks 
        self.channels = channels #this is dependent on the dac model
        self.block = block #this should be a seq2seq model with identical input and output sizes
        self.masker = weight_norm(nn.Conv1d(channels, channels*num_spks, 1, bias=False))

        self.activation = Snake1d(channels) #nn.Tanh() #nn.ReLU() #Snake1d(channels)
        # gated output layer
        self.output = nn.Sequential(
            nn.Conv1d(channels, channels, 1), nn.Tanh() #, Snake1d(channels)#
        )
        self.output_gate = nn.Sequential(
            nn.Conv1d(channels, channels, 1), nn.Sigmoid()
        )

    def forward(self,x):
        #[B,N,L]
        x = x.permute(0,2,1)
        #[B,L,N]
        x_b = self.block(x)
        #[B,L,N]
        x = x.permute(0,2,1)
        #[B,N,L]
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

class simpleSeparator2(nn.Module):
    def __init__(self, num_spks, channels, block, block_channels, activation = "Snake", activation_params = {}):
        super(simpleSeparator2, self).__init__()
        self.num_spks = num_spks 
        self.channels = channels #this is dependent on the dac model
        self.block = block #this should be a seq2seq model with identical input and output sizes
        self.ch_down = nn.Conv1d(channels, block_channels,1,bias=False)
        self.ch_up = nn.Conv1d(block_channels, channels,1,bias=False)
        #self.time_mix = nn.Conv1d(channels,channels,1,bias=False)
        self.masker = weight_norm(nn.Conv1d(channels, channels*num_spks, 1, bias=False))

        if activation == "Snake":
            self.activation = Snake1d(channels)
        else:
            act = getattr(nn, activation)
            self.activation = act(**activation_params)
        # self.activation = Snake1d(channels) #nn.Tanh() #nn.ReLU() #Snake1d(channels)
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

class simpleSeparatorLM(nn.Module):
    def __init__(self, num_spks, channels, EncoderBlock, DecoderBlock, block_channels):
        super(simpleSeparator, self).__init__()

        self.num_spks = num_spks 
        self.channels = channels #this is dependent on the dac model

        self.EncoderBlock = EncoderBlock #this should be a seq2seq model with identical input and output sizes
        self.DecoderBlock = DecoderBlock #this should be a seq2seq model with src, tgt input and output of same size

        self.ch_down_enc = nn.Conv1d(channels, block_channels,1,bias=False, padding="same")
        self.ch_down_dec = nn.Conv1d(channels, block_channels,1,bias=False, padding="same")

        self.ch_up = nn.Conv1d(block_channels, channels,1,bias=False, padding="same")
        
        self.masker = weight_norm(nn.Conv1d(channels, channels*num_spks, 1, bias=False, padding="same"))

        self.activation = Snake1d(channels) #nn.Tanh() #nn.ReLU() #Snake1d(channels)
        # gated output layer
        self.output = nn.Sequential(
            nn.Conv1d(channels, channels, 1), Snake1d(channels) #nn.Tanh() #, Snake1d(channels)#
        )
        self.output_gate = nn.Sequential(
            nn.Conv1d(channels, channels, 1), nn.Sigmoid()
        )

    def forward(self,x):
        x_enc = self.ch_down_enc(x)
        x_dec = self.ch_down_dec(x)
        #[B,N,L]
        x_enc = x_enc.permute(0,2,1)
        x_dec = x_dec.permute(0,2,1)
        #[B,L,N]
        x_b = self.EncoderBlock(x_enc)
        x_b = self.DecoderBlock(x_dec,x_b)
        #[B,L,N]
        x_b = x_b.permute(0,2,1)
        #[B,N,L]
        x_b = self.ch_up(x_b)

        B, N, L = x.shape
        masks = self.masker(x_b)
        
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


class simpleUNetSeparator(nn.Module):
    def __init__(self, num_spks, channels, block):
        super(simpleUNetSeparator, self).__init__()
        
        self.num_spks = num_spks 
        self.channels = channels #this is dependent on the dac model

        self.ch_down1 = nn.Conv1d(channels, channels//2 ,1,bias=False, padding="same")
        self.ch_down2 = nn.Conv1d(channels//2, channels//4 ,1,bias=False, padding="same")
        self.ch_down3 = nn.Conv1d(channels//4, channels//8 ,1,bias=False, padding="same")

        self.block = block

        self.ch_up1 = nn.Conv1d(channels//4, channels//4,1,bias=False, padding="same")
        self.ch_up2 = nn.Conv1d(channels//2, channels//2,1,bias=False, padding="same")
        self.ch_up3 = nn.Conv1d(channels, channels,1,bias=False, padding="same")
        
        self.masker = weight_norm(nn.Conv1d(channels, channels*num_spks, 1, bias=False, padding="same"))

        self.activation = Snake1d(channels) #nn.Tanh() #nn.ReLU() #Snake1d(channels)
        # gated output layer
        self.output = nn.Sequential(
            nn.Conv1d(channels, channels, 1), Snake1d(channels) #nn.Tanh() #, Snake1d(channels)#
        )
        self.output_gate = nn.Sequential(
            nn.Conv1d(channels, channels, 1), nn.Sigmoid()
        )

    def forward(self,x):

         #[B,N,L]
        x1 = self.ch_down1(x)
        x2 = self.ch_down2(x1)
        x3 = self.ch_down3(x2)
        
        xq = x3.permute(0,2,1)
        xq = self.block(xq)
        xq = xq.permute(0,2,1)

        x4 = self.ch_up1(torch.cat([x3,xq],dim=1))
        x5 = self.ch_up2(torch.cat([x2,x4],dim=1))
        x6 = self.ch_up3(torch.cat([x1,x5],dim=1))

        B, N, L = x6.shape
        masks = self.masker(x6)
        
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


class SBTransformerDecoderBlock(nn.Module):
    """A wrapper for the SpeechBrain implementation of the transformer decoder.

    Arguments
    ---------
    num_layers : int
        Number of layers.
    d_model : int
        Dimensionality of the representation.
    nhead : int
        Number of attention heads.
    d_ffn : int
        Dimensionality of positional feed forward.
    input_shape : tuple
        Shape of input.
    kdim : int
        Dimension of the key (Optional).
    vdim : int
        Dimension of the value (Optional).
    dropout : float
        Dropout rate.
    activation : str
        Activation function.
    use_positional_encoding : bool
        If true we use a positional encoding.
    norm_before: bool
        Use normalization before transformations.

    Example
    ---------
    >>> x = torch.randn(10, 100, 64)
    >>> block = SBTransformerDecoderBlock(1, 64, 8)
    >>> x = block(x)
    >>> x.shape
    torch.Size([10, 100, 64])
    """

    def __init__(
        self,
        num_layers,
        d_model,
        nhead,
        d_ffn=2048,
        input_shape=None,
        kdim=None,
        vdim=None,
        dropout=0.1,
        activation="relu",
        use_positional_encoding=False,
        norm_before=False,
        attention_type="regularMHA",
    ):
        super(SBTransformerDecoderBlock, self).__init__()
        self.use_positional_encoding = use_positional_encoding

        if activation == "relu":
            activation = nn.ReLU
        elif activation == "gelu":
            activation = nn.GELU
        else:
            raise ValueError("unknown activation")

        self.mdl = TransformerDecoder(
            num_layers=num_layers,
            nhead=nhead,
            d_ffn=d_ffn,
            #input_shape=input_shape,
            d_model=d_model,
            kdim=kdim,
            vdim=vdim,
            dropout=dropout,
            activation=activation,
            normalize_before=norm_before,
            attention_type=attention_type,
        )

        if use_positional_encoding:
            self.pos_enc = PositionalEncoding(input_size=d_model)

    def forward(self, src, tgt):
        """Returns the transformed output.

        Arguments
        ---------
        src, tgt : torch.Tensor
            Tensor shape [B, L, N],
            where, B = Batchsize,
                   L = time points
                   N = number of filters
        
        src is the original mixture
        tgt is the output of the encoder
        """
        if self.use_positional_encoding:
            pos_enc = self.pos_enc(src)
            return self.mdl(src + pos_enc, tgt + pos_enc)[0]
        else:
            return self.mdl(x)[0]


class NonMaskingRNN(nn.Module):
    def __init__(self, num_spks, channels, block):
        super(NonMaskingRNN, self).__init__()
        self.num_spks = num_spks 
        self.channels = channels #this is dependent on the dac model
        self.block = block #this should be a bidirectional rnn block
        self.ch_down = nn.Conv1d(channels, channels,1,bias=False)
        self.ch_up = nn.Conv1d(channels*num_spks, channels*num_spks,1,bias=False)
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
        #[B,N*num_spks,L]
        x = self.ch_up(x_b)

        #[B,N*num_spks,L]
        B, _ , L = x.shape
        x = x.view(B*self.num_spks,-1,L)
        
        #[B*num_spks, N, L]
        x = self.output(x) * self.output_gate(x)
        x = self.activation(x)

        #[B*num_spks, N, L]
        _, N, L = x.shape
        x = x.view(B, self.num_spks, N, L)
        
        # [B, spks, N, L]
        x = x.transpose(0,1)
        # [spks, B, N, L]

        return x

class RVQPredictorBlock(nn.Module):
    """
    This model predicts the codes in an RVQ module
    Example
    ---------
    >>> x = torch.randn(10, 100, 64)
    >>> block = SBRWKVBlock(1, 64, 8)
    >>> x = block(x)
    >>> x.shape
    torch.Size([10, 100, 64])
    """

    def __init__(
        self,
        num_codes,
        channels
    ):
        super(RVQPredictorBlock, self).__init__()
        self.channels = channels
        self.layers = torch.nn.ModuleList(
            [
                nn.Conv1d(channels, channels,1,bias=False)
                for i in range(num_codes)
            ]
        )

    def forward(self, x):
        """Returns the transformed output.

        Arguments
        ---------
        x : torch.Tensor
            Tensor shape [B, N, L],
            where, B = Batchsize,
                   N = number of filters
                   L = time points
                   
        
        """
        preds = []
        for i, layer in enumerate(self.layers):
            comb = layer(x)
            preds.append(comb.unsqueeze(2))
        out = torch.cat(preds,dim=2).unsqueeze(1)
        return out

class RVQPredictorBlock2(nn.Module):
    """
    This model predicts the codes in an RVQ module
    Example
    ---------
    >>> x = torch.randn(10, 100, 64)
    >>> block = SBRWKVBlock(1, 64, 8)
    >>> x = block(x)
    >>> x.shape
    torch.Size([10, 100, 64])
    """

    def __init__(
        self,
        num_codes,
        channels
    ):
        super(RVQPredictorBlock2, self).__init__()
        self.channels = channels
        self.layers = torch.nn.ModuleList(
            [
                nn.Conv1d(channels, channels*2,1,bias=False)
                for i in range(num_codes)
            ]
        )

    def forward(self, x):
        """Returns the transformed output.

        Arguments
        ---------
        x : torch.Tensor
            Tensor shape [B, N, L],
            where, B = Batchsize,
                   N = number of filters
                   L = time points
                   
        
        """
        preds = []
        for i, layer in enumerate(self.layers):
            comb = layer(x)
            preds.append(comb[:,:self.channels,:].unsqueeze(2))
            x = comb[:,self.channels:, :] + x
        out = torch.cat(preds,dim=2).unsqueeze(1)
        return out

if __name__ == '__main__':
    from speechbrain.lobes.models.gan_codec.dac import DACGenerator
    import yaml
    with open('/home/jiaqi006/.cache/huggingface/hub/models--espnet--amuse_dac_16k/snapshots/c8247b0e171a33c2586a681b38a5ca634a2bbf0d/exp_16k/codec_train_dac_fs16000_raw_fs16000/config.yaml') as f:
        # use safe_load instead load
        dataMap = yaml.safe_load(f)
    
    # print(type(dataMap['codec_conf']['sampling_rate']))
    # print(dataMap['codec_conf']['generator_params'].keys())
    model_path = '/home/jiaqi006/.cache/huggingface/hub/models--espnet--amuse_dac_16k/snapshots/c8247b0e171a33c2586a681b38a5ca634a2bbf0d/exp_16k/codec_train_dac_fs16000_raw_fs16000/120epoch.pth'
    generator_params = dataMap['codec_conf']['generator_params']
    generator = DACGenerator(**generator_params)
    load_pretrained_model(
            init_param = f'{model_path}:codec.generator',
            model = generator,
            ignore_init_mismatch = False,
            map_location = "cpu",
        )
    print("ptmodel loaded")
    # model = DAC(
    #     sampling_rate = dataMap['codec_conf']['sampling_rate'],
    #     generator_params = dataMap['codec_conf']['generator_params'],
    #     discriminator_params = dataMap['codec_conf']['discriminator_params'],
    # )
    # print(model)