import json
import os

import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset
from tqdm import tqdm

from FastSpeech2.DurationCalculator import DurationCalculator
from FastSpeech2.EnergyCalculator import EnergyCalculator
from FastSpeech2.PitchCalculator import Dio
from PreprocessingForTTS.ProcessAudio import AudioPreprocessor
from PreprocessingForTTS.ProcessText import TextFrontend


class FastSpeechDataset(Dataset):

    def __init__(self,
                 path_to_transcript_dict,
                 acoustic_model,
                 cache_dir,
                 diagonal_attention_head_id=None,  # every transformer has one attention head
                 # that is the most diagonal. Look for it manually (e.g. using run_visualization.py)
                 # and then provide it here.
                 speaker_embedding=False,
                 lang="en",
                 min_len_in_seconds=1,
                 max_len_in_seconds=20,
                 reduction_factor=1,
                 device=torch.device("cpu"),
                 rebuild_cache=False,
                 path_blacklist=None  # because for some datasets, some of the alignments
                 # simply fail because attention heads do weird things. Those need to be
                 # found in the duration_vis folder and manually added to a list of samples
                 # to be excluded from the dataset.
                 ):
        self.speaker_embedding = speaker_embedding
        if not os.path.exists(os.path.join(cache_dir, "fast_train_cache.json")) or rebuild_cache:
            if not os.path.isdir(os.path.join(cache_dir, "durations_visualization")):
                os.makedirs(os.path.join(cache_dir, "durations_visualization"))
            self.path_to_transcript_dict = path_to_transcript_dict
            key_list = list(self.path_to_transcript_dict.keys())
            # build cache
            print("... building dataset cache ...")
            self.datapoints = list()

            tf = TextFrontend(language=lang, use_word_boundaries=False, use_explicit_eos=False)
            _, sr = sf.read(key_list[0])
            if speaker_embedding:
                wav2mel = torch.jit.load("Models/Use/SpeakerEmbedding/wav2mel.pt")
                dvector = torch.jit.load("Models/Use/SpeakerEmbedding/dvector-step250000.pt").eval()
            ap = AudioPreprocessor(input_sr=sr, output_sr=16000, melspec_buckets=80, hop_length=256, n_fft=1024)
            acoustic_model = acoustic_model.to(device)
            dc = DurationCalculator(reduction_factor=reduction_factor, diagonal_attention_head_id=diagonal_attention_head_id)
            dio = Dio(reduction_factor=reduction_factor)
            energy_calc = EnergyCalculator(reduction_factor=reduction_factor)
            for index, path in tqdm(enumerate(key_list)):
                transcript = self.path_to_transcript_dict[path]
                with open(path, "rb") as audio_file:
                    wave, sr = sf.read(audio_file)
                if min_len_in_seconds <= len(wave) / sr <= max_len_in_seconds:
                    # print("Processing {} out of {}.".format(index, len(path_list)))
                    norm_wave = ap.audio_to_wave_tensor(audio=wave, normalize=True, mulaw=False)
                    norm_wave_length = torch.LongTensor([len(norm_wave)])
                    melspec = ap.audio_to_mel_spec_tensor(norm_wave, normalize=False).transpose(0, 1)
                    melspec_length = torch.LongTensor([len(melspec)])
                    text = tf.string_to_tensor(transcript).long()
                    cached_text = tf.string_to_tensor(transcript).squeeze(0).numpy().tolist()
                    cached_text_lens = len(cached_text)
                    cached_speech = ap.audio_to_mel_spec_tensor(wave).transpose(0, 1).numpy().tolist()
                    cached_speech_lens = len(cached_speech)
                    if not speaker_embedding:
                        os.path.join(cache_dir, "durations_visualization")
                        cached_durations = dc(acoustic_model.inference(text=text.squeeze(0).to(device),
                                                                       speech=melspec.to(device),
                                                                       use_teacher_forcing=True,
                                                                       speaker_embeddings=None)[2],
                                              vis=os.path.join(cache_dir, "durations_visualization", path.split("/")[-1].rstrip(".wav") + ".png"))[0].cpu()
                    else:
                        wav_tensor, sample_rate = torchaudio.load(path)
                        mel_tensor = wav2mel(wav_tensor, sample_rate)
                        cached_speaker_embedding = dvector.embed_utterance(mel_tensor)
                        cached_durations = dc(acoustic_model.inference(text=text.squeeze(0).to(device),
                                                                       speech=melspec.to(device),
                                                                       use_teacher_forcing=True,
                                                                       speaker_embeddings=cached_speaker_embedding.to(device))[2],
                                              vis=os.path.join(cache_dir, "durations_visualization", path.split("/")[-1].rstrip(".wav") + ".png"))[0].cpu()
                    cached_energy = energy_calc(input=norm_wave.unsqueeze(0), input_lengths=norm_wave_length, feats_lengths=melspec_length, durations=cached_durations.unsqueeze(0),
                                                durations_lengths=torch.LongTensor([len(cached_durations)]))[0].squeeze(0)
                    cached_pitch = dio(input=norm_wave.unsqueeze(0), input_lengths=norm_wave_length, feats_lengths=melspec_length, durations=cached_durations.unsqueeze(0),
                                       durations_lengths=torch.LongTensor([len(cached_durations)]))[0].squeeze(0)
                    if not self.speaker_embedding:
                        self.datapoints.append([cached_text,
                                                cached_text_lens,
                                                cached_speech,
                                                cached_speech_lens,
                                                cached_durations.numpy().tolist(),
                                                cached_energy.numpy().tolist(),
                                                cached_pitch.numpy().tolist(),
                                                path])
                    else:
                        self.datapoints.append([cached_text,
                                                cached_text_lens,
                                                cached_speech,
                                                cached_speech_lens,
                                                cached_durations.numpy().tolist(),
                                                cached_energy.numpy().tolist(),
                                                cached_pitch.numpy().tolist(),
                                                cached_speaker_embedding.detach().numpy().tolist(),
                                                path])

            # save to json so we can rebuild cache quickly
            with open(os.path.join(cache_dir, "fast_train_cache.json"), 'w') as fp:
                json.dump(self.datapoints, fp)
        else:
            # just load the datapoints
            with open(os.path.join(cache_dir, "fast_train_cache.json"), 'r') as fp:
                self.datapoints = json.load(fp)

        if path_blacklist is not None:
            bl = set(path_blacklist)
            datapoints_with_durations_that_make_sense = list()
            for el in self.datapoints:
                if el[-1] not in bl:
                    datapoints_with_durations_that_make_sense.append(el)
            self.datapoints = datapoints_with_durations_that_make_sense
        print("Prepared {} datapoints.".format(len(self.datapoints)))

    def __getitem__(self, index):
        if not self.speaker_embedding:
            return self.datapoints[index][0], self.datapoints[index][1], self.datapoints[index][2], self.datapoints[index][3], self.datapoints[index][4], \
                   self.datapoints[index][5], self.datapoints[index][6]
        else:
            return self.datapoints[index][0], self.datapoints[index][1], self.datapoints[index][2], self.datapoints[index][3], self.datapoints[index][4], \
                   self.datapoints[index][5], self.datapoints[index][6], self.datapoints[index][7]

    def __len__(self):
        return len(self.datapoints)
