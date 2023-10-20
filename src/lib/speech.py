import torch
import os
import json
import scipy
import resampy
import numpy as np
from scipy.io.wavfile import write
from tacotron2.model import Tacotron2
from tacotron2.hparams import create_hparams
from tacotron2.layers import TacotronSTFT
from tacotron2.audio_processing import griffin_lim
from tacotron2.text import text_to_sequence
from hifigan.env import AttrDict
from hifigan.models import Generator
from hifigan.denoiser import Denoiser
from hifigan.meldataset import mel_spectrogram, MAX_WAV_VALUE

HIFIGAN_CONFIG = "config_v1"
USE_ARPABET = False
SUPERRES_STRENGTH = 10

thisdict = {}
for line in reversed((open('data/merged.dict.txt', "r").read()).splitlines()):
    thisdict[(line.split(" ",1))[0]] = (line.split(" ",1))[1].strip()

def ARPA(text, punctuation=r"!?,.;", EOS_Token=True):
    out = ''
    for word_ in text.split(" "):
        word=word_; end_chars = ''
        while any(elem in word for elem in punctuation) and len(word) > 1:
            if word[-1] in punctuation: end_chars = word[-1] + end_chars; word = word[:-1]
            else: break
        try:
            word_arpa = thisdict[word.upper()]
            word = "{" + str(word_arpa) + "}"
        except KeyError: pass
        out = (out + " " + word + end_chars).strip()
    
    if EOS_Token and out[-1] != ";": out += ";"
    return out

class Speech:
    def __init__(self, hifigan_model: str, tacotron2_model: str) -> None:
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"
        
        # Load HiFi-GAN
        hifigan_path = "models/" + hifigan_model

        conf = os.path.join("src/hifigan", HIFIGAN_CONFIG + ".json")
        with open(conf) as f:
            json_config = json.loads(f.read())
        h = AttrDict(json_config)
        torch.manual_seed(h.seed)
        hifigan = Generator(h).to(torch.device(self.device))
        state_dict_g = torch.load(hifigan_path, map_location=torch.device(self.device))
        hifigan.load_state_dict(state_dict_g["generator"])
        hifigan.eval()
        hifigan.remove_weight_norm()
        denoiser = Denoiser(hifigan, mode="normal")

        self.hifigan = hifigan
        self.h = h
        self.denoiser = denoiser

        # Load Superres HiFi-GAN
        hifigan_path = "models/superres_hifigan"

        conf = os.path.join("src/hifigan", "config_32k" + ".json")
        with open(conf) as f:
            json_config = json.loads(f.read())
        h = AttrDict(json_config)
        torch.manual_seed(h.seed)
        hifigan = Generator(h).to(torch.device(self.device))
        state_dict_g = torch.load(hifigan_path, map_location=torch.device(self.device))
        hifigan.load_state_dict(state_dict_g["generator"])
        hifigan.eval()
        hifigan.remove_weight_norm()
        denoiser = Denoiser(hifigan, mode="normal")

        self.hifigan_sr = hifigan
        self.h2 = h
        self.denoiser_sr = denoiser

        # Load Tacotron2
        tacotron2_path = "models/" + tacotron2_model

        # Load Tacotron2 and Config
        hparams = create_hparams()
        hparams.sampling_rate = 22050
        hparams.max_decoder_steps = 3000 # Max Duration
        hparams.gate_threshold = 0.25 # Model must be 25% sure the clip is over before ending generation
        model = Tacotron2(hparams)
        state_dict = torch.load(tacotron2_path)['state_dict']
        model.load_state_dict(state_dict)
        _ = model.cuda().eval().half()

        self.model = model
        self.hparams = hparams

    def say(self, text: str) -> None:
        print("Generating '" + text + "'")

        for i in [x for x in text.split("\n") if len(x)]:
            if not USE_ARPABET:
                if i[-1] != ";": i=i+";"
            else: i = ARPA(i)
            with torch.no_grad(): # save VRAM by not including gradients
                sequence = np.array(text_to_sequence(i, ['english_cleaners']))[None, :]
                sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
                mel_outputs, mel_outputs_postnet, _, alignments = self.model.inference(sequence)

                y_g_hat = self.hifigan(mel_outputs_postnet.float())
                audio = y_g_hat.squeeze()
                audio = audio * MAX_WAV_VALUE
                audio_denoised = self.denoiser(audio.view(1, -1), strength=35)[:, 0]

                # Resample to 32k
                audio_denoised = audio_denoised.cpu().numpy().reshape(-1)

                normalize = (MAX_WAV_VALUE / np.max(np.abs(audio_denoised))) ** 0.9
                audio_denoised = audio_denoised * normalize
                wave = resampy.resample(
                    audio_denoised,
                    self.h.sampling_rate,
                    self.h2.sampling_rate,
                    filter="sinc_window",
                    window=scipy.signal.windows.hann,
                    num_zeros=8,
                )
                wave_out = wave.astype(np.int16)

                # HiFi-GAN super-resolution
                wave = wave / MAX_WAV_VALUE
                wave = torch.FloatTensor(wave).to(torch.device("cuda"))
                new_mel = mel_spectrogram(
                    wave.unsqueeze(0),
                    self.h2.n_fft,
                    self.h2.num_mels,
                    self.h2.sampling_rate,
                    self.h2.hop_size,
                    self.h2.win_size,
                    self.h2.fmin,
                    self.h2.fmax,
                )
                y_g_hat2 = self.hifigan_sr(new_mel)
                audio2 = y_g_hat2.squeeze()
                audio2 = audio2 * MAX_WAV_VALUE
                audio2_denoised = self.denoiser(audio2.view(1, -1), strength=35)[:, 0]

                # High-pass filter, mixing and denormalizing
                audio2_denoised = audio2_denoised.cpu().numpy().reshape(-1)
                b = scipy.signal.firwin(
                    101, cutoff=10500, fs=self.h2.sampling_rate, pass_zero=False
                )
                y = scipy.signal.lfilter(b, [1.0], audio2_denoised)
                y *= SUPERRES_STRENGTH
                y_out = y.astype(np.int16)
                y_padded = np.zeros(wave_out.shape)
                y_padded[: y_out.shape[0]] = y_out
                sr_mix = wave_out + y_padded
                sr_mix = sr_mix / normalize

                write("output/test.wav", self.h2.sampling_rate, sr_mix.astype(np.int16))