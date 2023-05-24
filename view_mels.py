# approaches for viewing mels generated for DL GANs
# initial format is: 

# imports
import os, argparse
import matplotlib.pyplot as plt
import librosa
import json
import random
import numpy as np

# functions

def get_config(checkpoint_file):
  config_file = os.path.join(os.path.split(checkpoint_file)[0], 'config.json').replace("\\","/")
  with open(config_file) as f:
      data = f.read()

  class AttrDict(dict):
      def __init__(self, *args, **kwargs):
          super(AttrDict, self).__init__(*args, **kwargs)
          self.__dict__ = self

  global h #hyperparameters for audio & MEL # don't need to define as global here.
  json_config = json.loads(data)

  return AttrDict(json_config)

def plot_mel(mel,
             h,
             ax=None,
             tytle='MEL spectrogram'):
  # get dims for mel, compress to 2-D if needed.
    # if no axis specified create singleton.
    if not ax:
        fig, ax = plt.subplots(figsize=(10,5))

    # Display the spectrogram

    plt.sca(ax)
    librosa.display.specshow(mel, y_axis='mel', 
                             fmax=h.fmax, 
                             hop_length=h.hop_size, 
                             sr=h.sampling_rate,
                             x_axis='time')    
    # fig.suptitle(tytle)

    ax.set(xlabel='time', ylabel='frequency')
    plt.colorbar(format='%+2.0f dB')
    plt.title(tytle)
    plt.tight_layout()
    plt.show()

def compare_mels(mel0, mel1):
  print('working on it')
   

# main
def main():
  print('Viewing mels')

  parser = argparse.ArgumentParser()
  parser.add_argument('--mel_dir', default='mel_spects')
  parser.add_argument('--output_dir', default='mel_display') #None)
  parser.add_argument('--checkpoint_file',  default='bigvgan_24khz_100band-20230502T202754Z/bigvgan_24khz_100band/g_05000000.zip')

  a = parser.parse_args()

  if a.output_dir:
     if not os.path.exists(a.output_dir):
        os.makedirs(a.output_dir)
  
  mel_list = os.listdir(a.mel_dir)
  mel_samples = random.sample(mel_list,2)
  print(mel_samples)

  mel_file = mel_samples[0]
  mel = np.load(os.path.join(a.mel_dir),mel_file)

  h = get_config(a.checkpoint_file)
  print(h)

  plot_mel(mel, h, ax=None,tytle=mel_file)


if __name__ == "__main__":
  main()





