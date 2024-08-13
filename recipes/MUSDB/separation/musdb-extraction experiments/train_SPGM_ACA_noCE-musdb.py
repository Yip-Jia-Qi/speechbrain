#!/usr/bin/env/python3
"""Recipe for training a neural speech separation system on wsjmix the
dataset. The system employs an encoder, a decoder, and a masking network.

To run this recipe, do the following:
> python train.py hparams/sepformer.yaml
> python train.py hparams/dualpath_rnn.yaml
> python train.py hparams/convtasnet.yaml

The experiment file is flexible enough to support different neural
networks. By properly changing the parameter files, you can try
different architectures. The script supports both wsj2mix and
wsj3mix.


Authors
 * Cem Subakan 2020
 * Mirco Ravanelli 2020
 * Samuele Cornell 2020
 * Mirko Bronzi 2020
 * Jianyuan Zhong 2020
 
Modified the Speech Separation Recipe for Speaker Extraction using SpEx+
"""

import os
import sys
import torch
import torch.nn.functional as F
import torchaudio
import speechbrain as sb
import speechbrain.nnet.schedulers as schedulers
from speechbrain.utils.distributed import run_on_main
from torch.cuda.amp import autocast
from hyperpyyaml import load_hyperpyyaml
import numpy as np
from tqdm import tqdm
import csv
import logging
from multi_scale_spex_sisnr import compute_spex_loss
from metric import si_snr

# Define training procedure
class Extraction(sb.Brain):
    def compute_forward(self, mix, target, stage, noise=None):
        """
        Forward computations from the mixture to the extracted signals.
        mix: music
        target: vocals
        """

        # Unpack lists and put tensors in the right device
        mix, mix_lens = mix
        mix, mix_lens = mix.to(self.device), mix_lens.to(self.device)
        target, target_lens = target
        target, target_lens = target.to(self.device), target_lens.to(self.device)

        # # Convert targets to tensor
        # target = torch.cat(
        #     [target[i][0].unsqueeze(-1) for i in range(self.hparams.num_spks)],
        #     dim=-1,
        # ).to(self.device)
        
#Skip this data augmentation for now, I have other fish to fry...

#         # Add speech distortions
#         if stage == sb.Stage.TRAIN:
#             with torch.no_grad():
#                 if self.hparams.use_speedperturb or self.hparams.use_rand_shift:
#                     mix, target = self.add_speed_perturb(target, mix_lens)
#                     mix = target.sum(-1)

#                 if self.hparams.use_wavedrop:
#                     mix = self.hparams.wavedrop(mix, mix_lens)

#                 if self.hparams.limit_training_signal_len:
#                     mix, target = self.cut_signals(mix, target)

#This is ideally how i want to do this, but I'll use the full spex+ script for now and work on separating out the different parts of the model later
#         # Separation
#         mix_w = self.hparams.Encoder(mix)
#         est_mask = self.hparams.MaskNet(mix_w)
#         mix_w = torch.stack([mix_w] * self.hparams.num_spks)
#         sep_h = mix_w * est_mask

#         #Decoding
#         est_source = torch.cat(
#             [
#                 self.hparams.Decoder(sep_h[i]).unsqueeze(-1)
#                 for i in range(self.hparams.num_spks)
#             ],
#             dim=-1,
#         )
        
        # #this is for spex+_conv_tas_net
        # ests, ests2, ests3, spk_pred = self.hparams.ExtractNet(mix, aux, aux_len)
        # est_sources = [ests,ests2,ests3]
        
        #this is for spex+_conv_tas_net_v2 and v3
        # est_sources, spk_pred = self.hparams.ExtractNet(mix, aux, aux_len)
        
        # When inference, only one utt
        if mix.dim() == 1:
            mix = torch.unsqueeze(mix,0)
        
        mixlen1 = mix.shape[-1]

        # Simple Encoder
        mix_w = self.hparams.Encoder(mix)
        
        # Obtain Speaker embeddings and speaker predictions
        
        aux = self.hparams.compute_features(mix)
        
        if len(aux.shape) == 2:
            aux = aux.unsqueeze(2)

        aux_emb = self.hparams.SpeakerEncoder(aux)
        # aux_emb = self.hparams.SpeakerEncoder(mix_w.shape[0])
        
        # aux_emb = aux_emb.squeeze().unsqueeze(0)
        aux_emb = aux_emb.squeeze(2)
        
        # raise Exception
        # Extraction Network conditioned on speaker embeddings
        est_mask, mask_emb = self.hparams.MaskNet(mix_w, aux_emb)
                
        # mix_w = torch.stack([mix_w] * self.hparams.num_spks)
        sep_h = mix_w * est_mask
    
        #Decoding
        est_source = self.hparams.Decoder(sep_h[0])
        
        #Speaker Classification
        # spk_pred = F.softmax(self.hparams.Classifier(aux_emb),dim=1)
        
        return est_source, target#, spk_pred

    def compute_objectives(self, predictions, targets, batch, stage):
        """Computes the sinr loss"""        

        
        if stage == sb.Stage.TRAIN:

            sisnr_loss = self.hparams.sisnr_loss(targets.permute(1,0).unsqueeze(2),predictions.permute(1,0).unsqueeze(2))

            # cal_si_snr(source, estimate_source)
            # PITWrapper(estimate_source, source)
            # they are swapped! take note!

#             onehot = torch.zeros(self.hparams.n_spk_cls)
#             onehot[batch["spk_idx"]]=1
#             onehot = onehot.to(self.device)
#             cls_loss_model = torch.nn.BCELoss()
#             cls_loss_model = cls_loss_model.to(self.device)    
#             cls_loss = cls_loss_model(spk_pred.squeeze(), onehot)
#             # snr_loss, ce_loss = compute_spex_loss(predictions, targets, spk_pred, batch)
            
#             factor = self.hparams.factor
            
#             loss = (factor * sisnr_loss) + (cls_loss * (1-factor))

            loss = sisnr_loss

            
        else:
            # loss = si_snr(targets.cpu().detach().numpy(),predictions[0].cpu().detach().numpy())

            loss = -self.hparams.sisnr_loss(predictions.permute(1,0).unsqueeze(2), targets.permute(1,0).unsqueeze(2))
            

            

            # loss = self.hparams.sisnr_loss(targets.cpu().detach().numpy(), predictions.cpu().detach().numpy())
            # loss = torch.tensor(loss)
            

        return loss
        # return self.hparams.loss(targets, predictions)

    def fit_batch(self, batch):
        """Trains one batch"""
        # Unpacking batch list
        mixture = batch['mix']
        target = batch['ref']
       
        if self.auto_mix_prec:
            with autocast():

                predictions, target = self.compute_forward(
                    mixture, target, sb.Stage.TRAIN
                )
                
                loss = self.compute_objectives(predictions, targets)

                # hard threshold the easy dataitems
                if self.hparams.threshold_byloss:
                    th = self.hparams.threshold
                    loss_to_keep = loss[loss > th]
                    if loss_to_keep.nelement() > 0:
                        loss = loss_to_keep.mean()
                else:
                    loss = loss.mean()

            if (
                loss < self.hparams.loss_upper_lim and loss.nelement() > 0
            ):  # the fix for computational problems
                self.scaler.scale(loss).backward()
                if self.hparams.clip_grad_norm >= 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.modules.parameters(), self.hparams.clip_grad_norm,
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.nonfinite_count += 1
                logger.info(
                    "infinite loss or empty loss! it happened {} times so far - skipping this batch".format(
                        self.nonfinite_count
                    )
                )
                loss.data = torch.tensor(0).to(self.device)
        else:

            predictions, targets = self.compute_forward(
                mixture, target, sb.Stage.TRAIN
            )
            
            loss = self.compute_objectives(predictions, targets, batch, sb.Stage.TRAIN)

            if self.hparams.threshold_byloss:
                th = self.hparams.threshold
                loss_to_keep = loss[loss > th]
                if loss_to_keep.nelement() > 0:
                    loss = loss_to_keep.mean()
            else:
                loss = loss.mean()

            if (
                loss < self.hparams.loss_upper_lim and loss.nelement() > 0
            ):  # the fix for computational problems
                loss.backward()
                if self.hparams.clip_grad_norm >= 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.modules.parameters(), self.hparams.clip_grad_norm
                    )
                self.optimizer.step()
            else:
                self.nonfinite_count += 1
                logger.info(
                    "infinite loss or empty loss! it happened {} times so far - skipping this batch".format(
                        self.nonfinite_count
                    )
                )
                loss.data = torch.tensor(0).to(self.device)
        self.optimizer.zero_grad()

        return loss.detach().cpu()

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""        
        mixture = batch['mix']
        target = batch['ref']

        with torch.no_grad():
            predictions, targets = self.compute_forward(mixture, target, stage)
            loss = self.compute_objectives(predictions, targets, batch, stage)

        # Manage audio file saving
        if stage == sb.Stage.TEST and self.hparams.save_audio:
            snt_id = batch.id #This is no longer part of the dataset outputs, if we want to do audio file saving, need to figure out how to fix this.
            if hasattr(self.hparams, "n_audio_to_save"):
                if self.hparams.n_audio_to_save > 0:
                    self.save_audio(snt_id[0], mixture, targets, predictions)
                    self.hparams.n_audio_to_save += -1
            else:
                self.save_audio(snt_id[0], mixture, targets, predictions)

        return loss.detach()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        # Compute/store important stats
        stage_stats = {"si-snr": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:

            # Learning rate annealing
            if isinstance(
                self.hparams.lr_scheduler, schedulers.ReduceLROnPlateau
            ):
                current_lr, next_lr = self.hparams.lr_scheduler(
                    [self.optimizer], epoch, stage_loss
                )
                schedulers.update_learning_rate(self.optimizer, next_lr)
            else:
                # if we do not use the reducelronplateau, we do not change the lr
                current_lr = self.hparams.optimizer.optim.param_groups[0]["lr"]

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": current_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"si-snr": stage_stats["si-snr"]}, min_keys=["si-snr"],
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )

    def add_speed_perturb(self, targets, targ_lens):
        """Adds speed perturbation and random_shift to the input signals"""

        min_len = -1
        recombine = False

        if self.hparams.use_speedperturb:
            # Performing speed change (independently on each source)
            new_targets = []
            recombine = True

            for i in range(targets.shape[-1]):
                new_target = self.hparams.speedperturb(
                    targets[:, :, i], targ_lens
                )
                new_targets.append(new_target)
                if i == 0:
                    min_len = new_target.shape[-1]
                else:
                    if new_target.shape[-1] < min_len:
                        min_len = new_target.shape[-1]

            if self.hparams.use_rand_shift:
                # Performing random_shift (independently on each source)
                recombine = True
                for i in range(targets.shape[-1]):
                    rand_shift = torch.randint(
                        self.hparams.min_shift, self.hparams.max_shift, (1,)
                    )
                    new_targets[i] = new_targets[i].to(self.device)
                    new_targets[i] = torch.roll(
                        new_targets[i], shifts=(rand_shift[0],), dims=1
                    )

            # Re-combination
            if recombine:
                if self.hparams.use_speedperturb:
                    targets = torch.zeros(
                        targets.shape[0],
                        min_len,
                        targets.shape[-1],
                        device=targets.device,
                        dtype=torch.float,
                    )
                for i, new_target in enumerate(new_targets):
                    targets[:, :, i] = new_targets[i][:, 0:min_len]

        mix = targets.sum(-1)
        return mix, targets

    def cut_signals(self, mixture, targets):
        """This function selects a random segment of a given length within the mixture.
        The corresponding targets are selected accordingly"""
        randstart = torch.randint(
            0,
            1 + max(0, mixture.shape[1] - self.hparams.training_signal_len),
            (1,),
        ).item()
        targets = targets[
            :, randstart : randstart + self.hparams.training_signal_len, :
        ]
        mixture = mixture[
            :, randstart : randstart + self.hparams.training_signal_len
        ]
        return mixture, targets

    def reset_layer_recursively(self, layer):
        """Reinitializes the parameters of the neural networks"""
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
        for child_layer in layer.modules():
            if layer != child_layer:
                self.reset_layer_recursively(child_layer)

    def save_results(self, test_data):
        """This script computes the SDR and SI-SNR metrics and saves
        them into a csv file"""
        raise Exception

        # This package is required for SDR computation
        from mir_eval.separation import bss_eval_sources

        # Create folders where to store audio
        save_file = os.path.join(self.hparams.output_folder, "test_results.csv")

        # Variable init
        all_sdrs = []
        all_sdrs_i = []
        all_sisnrs = []
        all_sisnrs_i = []
        csv_columns = ["snt_id", "sdr", "sdr_i", "si-snr", "si-snr_i"]
        
        if isinstance(train_dataloader,DataLoader):
            test_loader = test_data
        else:
            test_loader = sb.dataio.dataloader.make_dataloader(
                test_data, **self.hparams.dataloader_opts
            )

        with open(save_file, "w") as results_csv:
            writer = csv.DictWriter(results_csv, fieldnames=csv_columns)
            writer.writeheader()

            # Loop over all test sentence
            with tqdm(test_loader, dynamic_ncols=True) as t:
                for i, batch in enumerate(t):
                    
                    # Apply Separation
                    mixture = batch['mix']
                    target = batch['ref']
                    aux = batch['aux']
                    aux_len = batch['aux_len']
                    spk_id = batch['spk_idx']
                    # mixture = batch.mix
                    #snt_id = batch.id
                    # targets = [batch.s1_sig, batch.s2_sig]

                    with torch.no_grad():
                        predictions, targets = self.compute_forward(mixture, target, aux, aux_len, sb.Stage.TEST)
                    # with torch.no_grad():
                    #     predictions, targets = self.compute_forward(
                    #         batch.mix_sig, targets, sb.Stage.TEST
                    #     )

                    # Compute SI-SNR
                    sisnr = self.compute_objectives(predictions, targets, batch, sb.Stage.TEST)
                    # sisnr = self.compute_objectives(predictions, targets)

                    # Compute SI-SNR improvement
                    mixture = mixture.to(self.device)
                    sisnr_baseline = self.compute_objectives(mixture, targets, batch, sb.Stage.TEST)

                    # mixture_signal = torch.stack(
                    #     [mixture] * self.hparams.num_spks, dim=-1
                    # )
                    # mixture_signal = mixture_signal.to(targets.device)
                    # sisnr_baseline = self.compute_objectives(
                    #     mixture_signal, targets
                    # )
                    sisnr_i = sisnr - sisnr_baseline

                    # Compute SDR
                    sdr, _, _, _ = bss_eval_sources(
                        targets[0].t().cpu().numpy(),
                        predictions[0].squeeze().t().detach().cpu().numpy(),
                    )

                    sdr_baseline, _, _, _ = bss_eval_sources(
                        targets[0].t().cpu().numpy(),
                        mixture[0].t().detach().cpu().numpy(),
                    )

                    sdr_i = sdr.mean() - sdr_baseline.mean()

                    # Saving on a csv file
                    row = {
                        # "snt_id": snt_id[0], #add this back later
                        "sdr": sdr.mean(),
                        "sdr_i": sdr_i,
                        "si-snr": -sisnr.item(),
                        "si-snr_i": -sisnr_i.item(),
                    }
                    writer.writerow(row)

                    # Metric Accumulation
                    all_sdrs.append(sdr.mean())
                    all_sdrs_i.append(sdr_i.mean())
                    all_sisnrs.append(-sisnr.item())
                    all_sisnrs_i.append(-sisnr_i.item())

                row = {
                    # "snt_id": "avg", #add this back later
                    "sdr": np.array(all_sdrs).mean(),
                    "sdr_i": np.array(all_sdrs_i).mean(),
                    "si-snr": np.array(all_sisnrs).mean(),
                    "si-snr_i": np.array(all_sisnrs_i).mean(),
                }
                writer.writerow(row)

        logger.info("Mean SISNR is {}".format(np.array(all_sisnrs).mean()))
        logger.info("Mean SISNRi is {}".format(np.array(all_sisnrs_i).mean()))
        logger.info("Mean SDR is {}".format(np.array(all_sdrs).mean()))
        logger.info("Mean SDRi is {}".format(np.array(all_sdrs_i).mean()))

    def save_audio(self, snt_id, mixture, targets, predictions):
        "saves the test audio (mixture, targets, and estimated sources) on disk"

        # Create outout folder
        save_path = os.path.join(self.hparams.save_folder, "audio_results")
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        for ns in range(self.hparams.num_spks):

            # Estimated source
            signal = predictions[0, :, ns]
            signal = signal / signal.abs().max()
            save_file = os.path.join(
                save_path, "item{}_source{}hat.wav".format(snt_id, ns + 1)
            )
            torchaudio.save(
                save_file, signal.unsqueeze(0).cpu(), self.hparams.sample_rate
            )

            # Original source
            signal = targets[0, :, ns]
            signal = signal / signal.abs().max()
            save_file = os.path.join(
                save_path, "item{}_source{}.wav".format(snt_id, ns + 1)
            )
            torchaudio.save(
                save_file, signal.unsqueeze(0).cpu(), self.hparams.sample_rate
            )

        # Mixture
        signal = mixture[0][0, :]
        signal = signal / signal.abs().max()
        save_file = os.path.join(save_path, "item{}_mix.wav".format(snt_id))
        torchaudio.save(
            save_file, signal.unsqueeze(0).cpu(), self.hparams.sample_rate
        )

def dataio_prep(hparams):
    """Creates data processing pipeline"""
    
    def load_audio_with_random_offset(song_path, num_frames, sample_rate):
        """
        Loads a random segment of audio from a file using torchaudio.

        Args:
          mixture_path (str): Path to the audio file.
          vocal_path (str): Path to the audio file.
          minus_one_path (str): Path to the audio file.
          num_frames (int): The number of frames to load.
          sample_rate (int, optional): The desired sample rate for the loaded audio.

        Returns:
            tuple: A tuple containing the loaded audio as a torch.Tensor and the original
            sample rate (if not provided).
        """
        # Get audio info using torchaudio.info
        mixture_path, vocal_path = [os.path.join(song_path, stem) for stem in ['mixture.wav','vocals.wav']]

        mix_info = torchaudio.info(mixture_path)
        vocal_info = torchaudio.info(vocal_path)

        total_frames = mix_info.num_frames
        assert total_frames == vocal_info.num_frames

        # Check if num_frames is valid
        if num_frames <= 0 or num_frames > total_frames:
            raise ValueError(f"num_frames must be between 1 and {total_frames}")

        # Choose a random offset within the valid range
        max_offset = total_frames - num_frames
        random_offset = torch.randint(0, max_offset, size=(1,))  # Ensure tensor for consistent type

        # Load the audio with the chosen offset and specified number of frames
        mix_audio, original_sample_rate = torchaudio.load(
            mixture_path, frame_offset=random_offset.item(), num_frames=num_frames
        )
        if original_sample_rate != sample_rate:
            raise Exception("unexpected sample rate, please check if you are searching for the correct sample rate")

        vocal_audio, original_sample_rate = torchaudio.load(
            vocal_path, frame_offset=random_offset.item(), num_frames=num_frames
        )
        if original_sample_rate != sample_rate:
            raise Exception("unexpected sample rate, please check if you are searching for the correct sample rate")

        return mix_audio[0,:], vocal_audio[0,:]

    data_folder = hparams["data_folder"]
    num_frames = hparams["training_signal_len"]
    sample_rate = hparams["sample_rate"]
    
    # 1. Define datasets
    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_data"],
        replacements={"data_root": hparams["data_folder"]},
    )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_data"],
        replacements={"data_root": hparams["data_folder"]},
    )

    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["test_data"],
        replacements={"data_root": hparams["data_folder"]},
    )

    datasets = [train_data, valid_data, test_data]
    
    # 2. Provide audio pipelines
    # from create_speaker_list import getSpeakerIDsFromFileName
    
    @sb.utils.data_pipeline.takes("song_path")
    @sb.utils.data_pipeline.provides("mix","ref")
    def audio_pipeline(song_path):
        mix, ref = load_audio_with_random_offset(song_path, num_frames, sample_rate)
        # mix = sb.dataio.dataio.read_audio(mix_wav)
        return mix, ref
    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)
        
    sb.dataio.dataset.set_output_keys(
        datasets, ["mix","ref"]
    )
    return train_data, valid_data, test_data

if __name__ == "__main__":

    # This flag enables the inbuilt cudnn auto-tuner
    torch.backends.cudnn.benchmark = True
    
    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    
    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)
    
    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Logger info
    logger = logging.getLogger(__name__)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )
    # Check if wsj0_tr is set with dynamic mixing
    if hparams["dynamic_mixing"] and not os.path.exists(
        hparams["base_folder_dm"]
    ):
        print(
            "Please, specify a valid base_folder_dm folder when using dynamic mixing"
        )
        sys.exit(1)

    
    # Data preparation
    # from recipes.WSJ0Mix.prepare_data import prepare_wsjmix  # noqa

    # run_on_main(
    #     prepare_wsjmix,
    #     kwargs={
    #         "datapath": hparams["data_folder"],
    #         "savepath": hparams["save_folder"],
    #         "n_spks": hparams["num_spks"],
    #         "skip_prep": hparams["skip_prep"],
    #         "fs": hparams["sample_rate"],
    #     },
    # )
    
    #data has already been prepared in the dataset folder

    # Create dataset objects
    if hparams["dynamic_mixing"]:
        from dynamic_mixing import dynamic_mix_data_prep

        # if the base_folder for dm is not processed, preprocess them
        if "processed" not in hparams["base_folder_dm"]:
            # if the processed folder already exists we just use it otherwise we do the preprocessing
            if not os.path.exists(
                os.path.normpath(hparams["base_folder_dm"]) + "_processed"
            ):
                from recipes.WSJ0Mix.meta.preprocess_dynamic_mixing import (
                    resample_folder,
                )

                print("Resampling the base folder")
                run_on_main(
                    resample_folder,
                    kwargs={
                        "input_folder": hparams["base_folder_dm"],
                        "output_folder": os.path.normpath(
                            hparams["base_folder_dm"]
                        )
                        + "_processed",
                        "fs": hparams["sample_rate"],
                        "regex": "**/*.wav",
                    },
                )
                # adjust the base_folder_dm path
                hparams["base_folder_dm"] = (
                    os.path.normpath(hparams["base_folder_dm"]) + "_processed"
                )
            else:
                print(
                    "Using the existing processed folder on the same directory as base_folder_dm"
                )
                hparams["base_folder_dm"] = (
                    os.path.normpath(hparams["base_folder_dm"]) + "_processed"
                )

        # Colleting the hparams for dynamic batching
        dm_hparams = {
            "train_data": hparams["train_data"],
            "data_folder": hparams["data_folder"],
            "base_folder_dm": hparams["base_folder_dm"],
            "sample_rate": hparams["sample_rate"],
            "num_spks": hparams["num_spks"],
            "training_signal_len": hparams["training_signal_len"],
            "dataloader_opts": hparams["dataloader_opts"],
        }
        train_data = dynamic_mix_data_prep(dm_hparams)
        _, valid_data, test_data = dataio_prep(hparams)
    else:
        train_data, valid_data, test_data = dataio_prep(hparams)
    
    # Load pretrained model if pretrained_separator is present in the yaml
    if "pretrained_separator" in hparams:
        run_on_main(hparams["pretrained_separator"].collect_files)
        hparams["pretrained_separator"].load_collected()
        print("loaded pretrained")
        
    for param in hparams["modules"]["Encoder"].parameters():
        param.requires_grad = False
    print("Encoder Frozen")
    
    for param in hparams["modules"]["Decoder"].parameters():
        param.requires_grad = False
    print("Decoder Frozen")
    
    for param in hparams["modules"]["MaskNet"].parameters():
        param.requires_grad = False
    print("MaskNet Frozen")
    
    # Brain class initialization
    extractor = Extraction(
        modules=hparams["modules"],
        opt_class=hparams["optimizer"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    
    # re-initialize the parameters if we don't use a pretrained model
    if "pretrained_separator" not in hparams:
        for module in extractor.modules.values():
            extractor.reset_layer_recursively(module)
    
#     from dataloader import DataLoader
        
#     train_dataloader = DataLoader(train_data, batch_size = hparams["batch_size"])
#     valid_dataloader = DataLoader(valid_data, batch_size = hparams["batch_size"])
#     test_dataloader = DataLoader(test_data, batch_size = hparams["batch_size"])
    
    if not hparams["test_only"]:
        # Training
        extractor.fit(
            extractor.hparams.epoch_counter,
            train_data,
            valid_data,
            # train_loader_kwargs=hparams["dataloader_opts"],
            # valid_loader_kwargs=hparams["dataloader_opts"],
        )

    # Eval
    # extractor.evaluate(test_data, min_key="si-snr")
    # extractor.save_results(test_data)
    extractor.evaluate(test_data, min_key="si-snr")
    extractor.save_results(test_data)
