# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""HiFiGAN-related loss modules.

This code is modified from https://github.com/kan-bayashi/ParallelWaveGAN.

"""


from typing import Any, List, Dict, Optional, Tuple, Sequence, Union

import torch
import torch.nn.functional as F

import humanfriendly
import logging

from typeguard import typechecked
from abc import ABC, abstractmethod

import librosa
import numpy as np

from torch_complex import functional as FC

from torch_complex.tensor import ComplexTensor
from torch_complex.tensor import ComplexTensor
from packaging.version import parse as V

from typeguard import typechecked

# from espnet2.tts.feats_extract.log_mel_fbank import LogMelFbank
# from espnet2.layers.log_mel import LogMel
# from espnet2.layers.stft import Stft
# from espnet2.tts.feats_extract.abs_feats_extract import AbsFeatsExtract
# from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
# from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
# from espnet2.enh.layers.complex_utils import to_complex
# from espnet2.layers.inversible_interface import InversibleInterface

is_torch_1_10_plus = V(torch.__version__) >= V("1.10.0")

def to_complex(c):
    # Convert to torch native complex
    if isinstance(c, ComplexTensor):
        c = c.real + 1j * c.imag
        return c
    elif torch.is_complex(c):
        return c
    else:
        return torch.view_as_complex(c)

class InversibleInterface(ABC):
    @abstractmethod
    def inverse(
        self, input: torch.Tensor, input_lengths: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # return output, output_lengths
        raise NotImplementedError

def make_pad_mask(lengths, xs=None, length_dim=-1, maxlen=None):
    """Make mask tensor containing indices of padded part.

    Args:
        lengths (LongTensor or List): Batch of lengths (B,).
        xs (Tensor, optional): The reference tensor.
            If set, masks will be the same shape as this tensor.
        length_dim (int, optional): Dimension indicator of the above tensor.
            See the example.

    Returns:
        Tensor: Mask tensor containing indices of padded part.
                dtype=torch.uint8 in PyTorch 1.2-
                dtype=torch.bool in PyTorch 1.2+ (including 1.2)

    Examples:
        With only lengths.

        >>> lengths = [5, 3, 2]
        >>> make_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]

        With the reference tensor.

        >>> xs = torch.zeros((3, 2, 4))
        >>> make_pad_mask(lengths, xs)
        tensor([[[0, 0, 0, 0],
                 [0, 0, 0, 0]],
                [[0, 0, 0, 1],
                 [0, 0, 0, 1]],
                [[0, 0, 1, 1],
                 [0, 0, 1, 1]]], dtype=torch.uint8)
        >>> xs = torch.zeros((3, 2, 6))
        >>> make_pad_mask(lengths, xs)
        tensor([[[0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1]],
                [[0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1]],
                [[0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1]]], dtype=torch.uint8)

        With the reference tensor and dimension indicator.

        >>> xs = torch.zeros((3, 6, 6))
        >>> make_pad_mask(lengths, xs, 1)
        tensor([[[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1]],
                [[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1]],
                [[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1]]], dtype=torch.uint8)
        >>> make_pad_mask(lengths, xs, 2)
        tensor([[[0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1]],
                [[0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1]],
                [[0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1]]], dtype=torch.uint8)

    """
    if length_dim == 0:
        raise ValueError("length_dim cannot be 0: {}".format(length_dim))

    # If the input dimension is 2 or 3,
    # then we use ESPnet-ONNX based implementation for tracable modeling.
    # otherwise we use the traditional implementation for research use.
    if isinstance(lengths, list):
        logging.warning(
            "Using make_pad_mask with a list of lengths is not tracable. "
            + "If you try to trace this function with type(lengths) == list, "
            + "please change the type of lengths to torch.LongTensor."
        )

    if (
        (xs is None or xs.dim() in (2, 3))
        and length_dim <= 2
        and (not isinstance(lengths, list) and lengths.dim() == 1)
    ):
        return _make_pad_mask_traceable(lengths, xs, length_dim, maxlen)
    else:
        return _make_pad_mask(lengths, xs, length_dim, maxlen)

class Stft(torch.nn.Module, InversibleInterface):
    @typechecked
    def __init__(
        self,
        n_fft: int = 512,
        win_length: Optional[int] = None,
        hop_length: int = 128,
        window: Optional[str] = "hann",
        center: bool = True,
        normalized: bool = False,
        onesided: bool = True,
    ):
        super().__init__()
        self.n_fft = n_fft
        if win_length is None:
            self.win_length = n_fft
        else:
            self.win_length = win_length
        self.hop_length = hop_length
        self.center = center
        self.normalized = normalized
        self.onesided = onesided
        if window is not None and not hasattr(torch, f"{window}_window"):
            raise ValueError(f"{window} window is not implemented")
        self.window = window

    def extra_repr(self):
        return (
            f"n_fft={self.n_fft}, "
            f"win_length={self.win_length}, "
            f"hop_length={self.hop_length}, "
            f"center={self.center}, "
            f"normalized={self.normalized}, "
            f"onesided={self.onesided}"
        )

    def forward(
        self, input: torch.Tensor, ilens: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """STFT forward function.

        Args:
            input: (Batch, Nsamples) or (Batch, Nsample, Channels)
            ilens: (Batch)
        Returns:
            output: (Batch, Frames, Freq, 2) or (Batch, Frames, Channels, Freq, 2)

        """
        bs = input.size(0)
        if input.dim() == 3:
            multi_channel = True
            # input: (Batch, Nsample, Channels) -> (Batch * Channels, Nsample)
            input = input.transpose(1, 2).reshape(-1, input.size(1))
        else:
            multi_channel = False

        # NOTE(kamo):
        #   The default behaviour of torch.stft is compatible with librosa.stft
        #   about padding and scaling.
        #   Note that it's different from scipy.signal.stft

        # output: (Batch, Freq, Frames, 2=real_imag)
        # or (Batch, Channel, Freq, Frames, 2=real_imag)
        if self.window is not None:
            window_func = getattr(torch, f"{self.window}_window")
            window = window_func(
                self.win_length, dtype=input.dtype, device=input.device
            )
        else:
            window = None

        # For the compatibility of ARM devices, which do not support
        # torch.stft() due to the lack of MKL (on older pytorch versions),
        # there is an alternative replacement implementation with librosa.
        # Note: pytorch >= 1.10.0 now has native support for FFT and STFT
        # on all cpu targets including ARM.
        if input.is_cuda or torch.backends.mkl.is_available() or is_torch_1_10_plus:
            stft_kwargs = dict(
                n_fft=self.n_fft,
                win_length=self.win_length,
                hop_length=self.hop_length,
                center=self.center,
                window=window,
                normalized=self.normalized,
                onesided=self.onesided,
            )
            stft_kwargs["return_complex"] = True
            output = torch.stft(input, **stft_kwargs)
            output = torch.view_as_real(output)
        else:
            if self.training:
                raise NotImplementedError(
                    "stft is implemented with librosa on this device, which does not "
                    "support the training mode."
                )

            # use stft_kwargs to flexibly control different PyTorch versions' kwargs
            # note: librosa does not support a win_length that is < n_ftt
            # but the window can be manually padded (see below).
            stft_kwargs = dict(
                n_fft=self.n_fft,
                win_length=self.n_fft,
                hop_length=self.hop_length,
                center=self.center,
                window=window,
                pad_mode="reflect",
            )

            if window is not None:
                # pad the given window to n_fft
                n_pad_left = (self.n_fft - window.shape[0]) // 2
                n_pad_right = self.n_fft - window.shape[0] - n_pad_left
                stft_kwargs["window"] = torch.cat(
                    [torch.zeros(n_pad_left), window, torch.zeros(n_pad_right)], 0
                ).numpy()
            else:
                win_length = (
                    self.win_length if self.win_length is not None else self.n_fft
                )
                stft_kwargs["window"] = torch.ones(win_length)

            output = []
            # iterate over istances in a batch
            for i, instance in enumerate(input):
                stft = librosa.stft(input[i].numpy(), **stft_kwargs)
                output.append(torch.tensor(np.stack([stft.real, stft.imag], -1)))
            output = torch.stack(output, 0)
            if not self.onesided:
                len_conj = self.n_fft - output.shape[1]
                conj = output[:, 1 : 1 + len_conj].flip(1)
                conj[:, :, :, -1].data *= -1
                output = torch.cat([output, conj], 1)
            if self.normalized:
                output = output * (stft_kwargs["window"].shape[0] ** (-0.5))

        # output: (Batch, Freq, Frames, 2=real_imag)
        # -> (Batch, Frames, Freq, 2=real_imag)
        output = output.transpose(1, 2)
        if multi_channel:
            # output: (Batch * Channel, Frames, Freq, 2=real_imag)
            # -> (Batch, Frame, Channel, Freq, 2=real_imag)
            output = output.view(bs, -1, output.size(1), output.size(2), 2).transpose(
                1, 2
            )

        if ilens is not None:
            if self.center:
                pad = self.n_fft // 2
                ilens = ilens + 2 * pad

            olens = (
                torch.div(ilens - self.n_fft, self.hop_length, rounding_mode="trunc")
                + 1
            )
            output.masked_fill_(make_pad_mask(olens, output, 1), 0.0)
        else:
            olens = None

        return output, olens

    def inverse(
        self, input: Union[torch.Tensor, ComplexTensor], ilens: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Inverse STFT.

        Args:
            input: Tensor(batch, T, F, 2) or ComplexTensor(batch, T, F)
            ilens: (batch,)
        Returns:
            wavs: (batch, samples)
            ilens: (batch,)
        """
        input = to_complex(input)

        if self.window is not None:
            window_func = getattr(torch, f"{self.window}_window")
            datatype = input.real.dtype
            window = window_func(self.win_length, dtype=datatype, device=input.device)
        else:
            window = None

        input = input.transpose(1, 2)

        wavs = torch.functional.istft(
            input,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=self.center,
            normalized=self.normalized,
            onesided=self.onesided,
            length=ilens.max() if ilens is not None else ilens,
            return_complex=False,
        )

        return wavs, ilens

class LogMel(torch.nn.Module):
    """Convert STFT to fbank feats

    The arguments is same as librosa.filters.mel

    Args:
        fs: number > 0 [scalar] sampling rate of the incoming signal
        n_fft: int > 0 [scalar] number of FFT components
        n_mels: int > 0 [scalar] number of Mel bands to generate
        fmin: float >= 0 [scalar] lowest frequency (in Hz)
        fmax: float >= 0 [scalar] highest frequency (in Hz).
            If `None`, use `fmax = fs / 2.0`
        htk: use HTK formula instead of Slaney
    """

    def __init__(
        self,
        fs: int = 16000,
        n_fft: int = 512,
        n_mels: int = 80,
        fmin: float = None,
        fmax: float = None,
        htk: bool = False,
        log_base: float = None,
    ):
        super().__init__()

        fmin = 0 if fmin is None else fmin
        fmax = fs / 2 if fmax is None else fmax
        _mel_options = dict(
            sr=fs,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            htk=htk,
        )
        self.mel_options = _mel_options
        self.log_base = log_base

        # Note(kamo): The mel matrix of librosa is different from kaldi.
        melmat = librosa.filters.mel(**_mel_options)
        # melmat: (D2, D1) -> (D1, D2)
        self.register_buffer("melmat", torch.from_numpy(melmat.T).float())

    def extra_repr(self):
        return ", ".join(f"{k}={v}" for k, v in self.mel_options.items())

    def forward(
        self,
        feat: torch.Tensor,
        ilens: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # feat: (B, T, D1) x melmat: (D1, D2) -> mel_feat: (B, T, D2)
        mel_feat = torch.matmul(feat, self.melmat)
        mel_feat = torch.clamp(mel_feat, min=1e-10)

        if self.log_base is None:
            logmel_feat = mel_feat.log()
        elif self.log_base == 2.0:
            logmel_feat = mel_feat.log2()
        elif self.log_base == 10.0:
            logmel_feat = mel_feat.log10()
        else:
            logmel_feat = mel_feat.log() / torch.log(self.log_base)

        # Zero padding
        if ilens is not None:
            logmel_feat = logmel_feat.masked_fill(
                make_pad_mask(ilens, logmel_feat, 1), 0.0
            )
        else:
            ilens = feat.new_full(
                [feat.size(0)], fill_value=feat.size(1), dtype=torch.long
            )
        return logmel_feat, ilens

class AbsFeatsExtract(torch.nn.Module, ABC):
    @abstractmethod
    def output_size(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

class LogMelFbank(AbsFeatsExtract):
    """Conventional frontend structure for TTS.

    Stft -> amplitude-spec -> Log-Mel-Fbank
    """

    @typechecked
    def __init__(
        self,
        fs: Union[int, str] = 16000,
        n_fft: int = 1024,
        win_length: Optional[int] = None,
        hop_length: int = 256,
        window: Optional[str] = "hann",
        center: bool = True,
        normalized: bool = False,
        onesided: bool = True,
        n_mels: int = 80,
        fmin: Optional[int] = 80,
        fmax: Optional[int] = 7600,
        htk: bool = False,
        log_base: Optional[float] = 10.0,
    ):
        super().__init__()
        if isinstance(fs, str):
            fs = humanfriendly.parse_size(fs)

        self.fs = fs
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.fmin = fmin
        self.fmax = fmax

        self.stft = Stft(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            window=window,
            center=center,
            normalized=normalized,
            onesided=onesided,
        )

        self.logmel = LogMel(
            fs=fs,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            htk=htk,
            log_base=log_base,
        )

    def output_size(self) -> int:
        return self.n_mels

    def get_parameters(self) -> Dict[str, Any]:
        """Return the parameters required by Vocoder"""
        return dict(
            fs=self.fs,
            n_fft=self.n_fft,
            n_shift=self.hop_length,
            window=self.window,
            n_mels=self.n_mels,
            win_length=self.win_length,
            fmin=self.fmin,
            fmax=self.fmax,
        )

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Domain-conversion: e.g. Stft: time -> time-freq
        input_stft, feats_lens = self.stft(input, input_lengths)

        assert input_stft.dim() >= 4, input_stft.shape
        # "2" refers to the real/imag parts of Complex
        assert input_stft.shape[-1] == 2, input_stft.shape

        # NOTE(kamo): We use different definition for log-spec between TTS and ASR
        #   TTS: log_10(abs(stft))
        #   ASR: log_e(power(stft))

        # input_stft: (..., F, 2) -> (..., F)
        input_power = input_stft[..., 0] ** 2 + input_stft[..., 1] ** 2
        input_amp = torch.sqrt(torch.clamp(input_power, min=1.0e-10))
        input_feats, _ = self.logmel(input_amp, feats_lens)
        return input_feats, feats_lens

class GeneratorAdversarialLoss(torch.nn.Module):
    """Generator adversarial loss module."""

    def __init__(
        self,
        average_by_discriminators: bool = True,
        loss_type: str = "mse",
    ):
        """Initialize GeneratorAversarialLoss module.

        Args:
            average_by_discriminators (bool): Whether to average the loss by
                the number of discriminators.
            loss_type (str): Loss type, "mse" or "hinge".

        """
        super().__init__()
        self.average_by_discriminators = average_by_discriminators
        assert loss_type in ["mse", "hinge"], f"{loss_type} is not supported."
        if loss_type == "mse":
            self.criterion = self._mse_loss
        else:
            self.criterion = self._hinge_loss

    def forward(
        self,
        outputs: Union[List[List[torch.Tensor]], List[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Calcualate generator adversarial loss.

        Args:
            outputs (Union[List[List[Tensor]], List[Tensor], Tensor]): Discriminator
                outputs, list of discriminator outputs, or list of list of discriminator
                outputs..

        Returns:
            Tensor: Generator adversarial loss value.

        """
        if isinstance(outputs, (tuple, list)):
            adv_loss = 0.0
            for i, outputs_ in enumerate(outputs):
                if isinstance(outputs_, (tuple, list)):
                    # NOTE(kan-bayashi): case including feature maps
                    outputs_ = outputs_[-1]
                adv_loss += self.criterion(outputs_)
            if self.average_by_discriminators:
                adv_loss /= i + 1
        else:
            adv_loss = self.criterion(outputs)

        return adv_loss

    def _mse_loss(self, x):
        return F.mse_loss(x, x.new_ones(x.size()))

    def _hinge_loss(self, x):
        return -x.mean()


class DiscriminatorAdversarialLoss(torch.nn.Module):
    """Discriminator adversarial loss module."""

    def __init__(
        self,
        average_by_discriminators: bool = True,
        loss_type: str = "mse",
    ):
        """Initialize DiscriminatorAversarialLoss module.

        Args:
            average_by_discriminators (bool): Whether to average the loss by
                the number of discriminators.
            loss_type (str): Loss type, "mse" or "hinge".

        """
        super().__init__()
        self.average_by_discriminators = average_by_discriminators
        assert loss_type in ["mse", "hinge"], f"{loss_type} is not supported."
        if loss_type == "mse":
            self.fake_criterion = self._mse_fake_loss
            self.real_criterion = self._mse_real_loss
        else:
            self.fake_criterion = self._hinge_fake_loss
            self.real_criterion = self._hinge_real_loss

    def forward(
        self,
        outputs_hat: Union[List[List[torch.Tensor]], List[torch.Tensor], torch.Tensor],
        outputs: Union[List[List[torch.Tensor]], List[torch.Tensor], torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calcualate discriminator adversarial loss.

        Args:
            outputs_hat (Union[List[List[Tensor]], List[Tensor], Tensor]): Discriminator
                outputs, list of discriminator outputs, or list of list of discriminator
                outputs calculated from generator.
            outputs (Union[List[List[Tensor]], List[Tensor], Tensor]): Discriminator
                outputs, list of discriminator outputs, or list of list of discriminator
                outputs calculated from groundtruth.

        Returns:
            Tensor: Discriminator real loss value.
            Tensor: Discriminator fake loss value.

        """
        if isinstance(outputs, (tuple, list)):
            real_loss = 0.0
            fake_loss = 0.0
            for i, (outputs_hat_, outputs_) in enumerate(zip(outputs_hat, outputs)):
                if isinstance(outputs_hat_, (tuple, list)):
                    # NOTE(kan-bayashi): case including feature maps
                    outputs_hat_ = outputs_hat_[-1]
                    outputs_ = outputs_[-1]
                real_loss += self.real_criterion(outputs_)
                fake_loss += self.fake_criterion(outputs_hat_)
            if self.average_by_discriminators:
                fake_loss /= i + 1
                real_loss /= i + 1
        else:
            real_loss = self.real_criterion(outputs)
            fake_loss = self.fake_criterion(outputs_hat)

        return real_loss, fake_loss

    def _mse_real_loss(self, x: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(x, x.new_ones(x.size()))

    def _mse_fake_loss(self, x: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(x, x.new_zeros(x.size()))

    def _hinge_real_loss(self, x: torch.Tensor) -> torch.Tensor:
        return -torch.mean(torch.min(x - 1, x.new_zeros(x.size())))

    def _hinge_fake_loss(self, x: torch.Tensor) -> torch.Tensor:
        return -torch.mean(torch.min(-x - 1, x.new_zeros(x.size())))


class FeatureMatchLoss(torch.nn.Module):
    """Feature matching loss module."""

    def __init__(
        self,
        average_by_layers: bool = True,
        average_by_discriminators: bool = True,
        include_final_outputs: bool = False,
    ):
        """Initialize FeatureMatchLoss module.

        Args:
            average_by_layers (bool): Whether to average the loss by the number
                of layers.
            average_by_discriminators (bool): Whether to average the loss by
                the number of discriminators.
            include_final_outputs (bool): Whether to include the final output of
                each discriminator for loss calculation.

        """
        super().__init__()
        self.average_by_layers = average_by_layers
        self.average_by_discriminators = average_by_discriminators
        self.include_final_outputs = include_final_outputs

    def forward(
        self,
        feats_hat: Union[List[List[torch.Tensor]], List[torch.Tensor]],
        feats: Union[List[List[torch.Tensor]], List[torch.Tensor]],
    ) -> torch.Tensor:
        """Calculate feature matching loss.

        Args:
            feats_hat (Union[List[List[Tensor]], List[Tensor]]): List of list of
                discriminator outputs or list of discriminator outputs calcuated
                from generator's outputs.
            feats (Union[List[List[Tensor]], List[Tensor]]): List of list of
                discriminator outputs or list of discriminator outputs calcuated
                from groundtruth..

        Returns:
            Tensor: Feature matching loss value.

        """
        feat_match_loss = 0.0
        for i, (feats_hat_, feats_) in enumerate(zip(feats_hat, feats)):
            feat_match_loss_ = 0.0
            if not self.include_final_outputs:
                feats_hat_ = feats_hat_[:-1]
                feats_ = feats_[:-1]
            for j, (feat_hat_, feat_) in enumerate(zip(feats_hat_, feats_)):
                feat_match_loss_ += F.l1_loss(feat_hat_, feat_.detach())
            if self.average_by_layers:
                feat_match_loss_ /= j + 1
            feat_match_loss += feat_match_loss_
        if self.average_by_discriminators:
            feat_match_loss /= i + 1

        return feat_match_loss


class MelSpectrogramLoss(torch.nn.Module):
    """Mel-spectrogram loss."""

    def __init__(
        self,
        fs: int = 22050,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: Optional[int] = None,
        window: str = "hann",
        n_mels: int = 80,
        fmin: Optional[int] = 0,
        fmax: Optional[int] = None,
        center: bool = True,
        normalized: bool = False,
        onesided: bool = True,
        log_base: Optional[float] = 10.0,
    ):
        """Initialize Mel-spectrogram loss.

        Args:
            fs (int): Sampling rate.
            n_fft (int): FFT points.
            hop_length (int): Hop length.
            win_length (Optional[int]): Window length.
            window (str): Window type.
            n_mels (int): Number of Mel basis.
            fmin (Optional[int]): Minimum frequency for Mel.
            fmax (Optional[int]): Maximum frequency for Mel.
            center (bool): Whether to use center window.
            normalized (bool): Whether to use normalized one.
            onesided (bool): Whether to use oneseded one.
            log_base (Optional[float]): Log base value.

        """
        super().__init__()
        self.wav_to_mel = LogMelFbank(
            fs=fs,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            center=center,
            normalized=normalized,
            onesided=onesided,
            log_base=log_base,
        )

    def forward(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
        spec: Optional[torch.Tensor] = None,
        use_mse: bool = False,
    ) -> torch.Tensor:
        """Calculate Mel-spectrogram loss.

        Args:
            y_hat (Tensor): Generated waveform tensor (B, 1, T).
            y (Tensor): Groundtruth waveform tensor (B, 1, T).
            spec (Optional[Tensor]): Groundtruth linear amplitude spectrum tensor
                (B, T, n_fft // 2 + 1).  if provided, use it instead of groundtruth
                waveform.
            use_l2 (bool): Whether to use mse_loss instead of l1

        Returns:
            Tensor: Mel-spectrogram loss value.

        """
        mel_hat, _ = self.wav_to_mel(y_hat.squeeze(1))
        if spec is None:
            mel, _ = self.wav_to_mel(y.squeeze(1))
        else:
            mel, _ = self.wav_to_mel.logmel(spec)
        if use_mse:
            mel_loss = F.mse_loss(mel_hat, mel)
        else:
            mel_loss = F.l1_loss(mel_hat, mel)

        return mel_loss
