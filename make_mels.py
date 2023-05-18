# adapted from https://github.com/NVIDIA/BigVGAN under the MIT License

import os
import argparse
import librosa
import torch
import json
import numpy as np

def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output

def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def get_mel(x):
    return mel_spectrogram(x, 
                           h.n_fft, 
                           h.num_mels, 
                           h.sampling_rate, 
                           h.hop_size, 
                           h.win_size, 
                           h.fmin, 
                           h.fmax)
  
def mel_spectrogram(y, 
                    n_fft, 
                    num_mels, 
                    sampling_rate, 
                    hop_size, 
                    win_size, 
                    fmin, fmax, 
                    center=False):
  if torch.min(y) < -1.:
      print('min value is ', torch.min(y))
  if torch.max(y) > 1.:
      print('max value is ', torch.max(y))

  if fmax not in mel_basis:
      mel = librosa.filters.mel(sr=sampling_rate, 
                                n_fft=n_fft, 
                                n_mels=num_mels, 
                                fmin=fmin, 
                                fmax=fmax)
      mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
      hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

  y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
  y = y.squeeze(1)

  # complex tensor as default, then use view_as_real for future pytorch compatibility
  spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                    center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)
  spec = torch.view_as_real(spec)
  spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

  spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
  spec = spectral_normalize_torch(spec)

  return spec

def make_mel_spec(wav_dir, output_dir):
  # create mel specs from wav files in wav_dir
  for filename in os.listdir(wav_dir):
    wav, sr = librosa.load(os.path.join(wav_dir, filename), sr=h.sampling_rate, mono=True)
    wav = torch.FloatTensor(wav).to(device)
    # compute mel spectrogram from the ground truth audio
    x = get_mel(wav.unsqueeze(0))
    # save spectrogram
    save_path = os.path.join(output_dir,filename[:-4])
    np.save(save_path, x)
    
def main():
  print('Converting waves to mel spectrograms for BigVGAN')

  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint_file', default='/content/drive/MyDrive/bigvgan_24khz_100band-20230502T202754Z-001.zip')
  parser.add_argument('--input_wav_dir', default='/content/drive/MyDrive/BigVGAN_clip_source')
  parser.add_argument('--output_mel_dir', default='/content/drive/MyDrive/BigVGAN_melSpects')

  a = parser.parse_args()

  config_file = os.path.join(os.path.split(a.checkpoint_file)[0], 'config.json')
  with open(config_file) as f:
      data = f.read()

  class AttrDict(dict):
      def __init__(self, *args, **kwargs):
          super(AttrDict, self).__init__(*args, **kwargs)
          self.__dict__ = self

  global h #hyperparameters for audio & MEL # don't need to define as global here.
  json_config = json.loads(data)
  h = AttrDict(json_config)

  global device # don't need to declare global here, but may move inside function calls
  if torch.cuda.is_available():
      torch.cuda.manual_seed(h.seed)
      device = torch.device('cuda')
  else:
      device = torch.device('cpu')
  MAX_WAV_VALUE = 32768.0
  global mel_basis, hann_window
  mel_basis = {}
  hann_window={}
  make_mel_spec(a.input_wav_dir,a.output_mel_dir)

 if __name__ == "__main__":
  main()
