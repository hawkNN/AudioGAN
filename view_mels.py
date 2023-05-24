# approaches for viewing mels generated for DL GANs
# initial format is: 

# imports
import os, argparse
import matplotlib.pyplot as plt
import librosa
import json
import random
import numpy as np
from IPython.display import Audio
from IPython.display import display

# functions

def get_config(file_path):
  if file_path.endswith('.zip'):
     config_file = os.path.join(os.path.split(file_path)[0], 'config.json').replace("\\","/")
  elif file_path.endswith('.json'):
     config_file = file_path
  else:
     config_file = input('Please provide config file path: ')
  print(f'Config file: {config_file}')
  
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
   
def show_mel_audio(mel_file,
                   audio_file,
                   h,
                   ax=None):
   
   fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10,5))

   # Display the spectrogram
   plt.sca(axes[0])
   mel = np.load(mel_file)
   if len(mel.shape)==3:
      mel = mel.squeeze(0)
   librosa.display.specshow(mel, 
                            y_axis='mel', 
                            fmax=h.fmax, 
                            hop_length=h.hop_size, 
                            sr=h.sampling_rate,
                            x_axis='time')  
   axes[0].set(xlabel='time', ylabel='frequency')
   plt.colorbar(format='%+2.0f dB')
   plt.title(os.path.split(mel_file)[-1])
   plt.tight_layout()
   plt.show()
   
   # Display audio play button
   plt.sca(axes[1])
   wn = Audio(audio_file, autoplay=True, rate=h.sampling_rate)
   display(wn) 


# main
def main():
  print('Viewing mels')

  parser = argparse.ArgumentParser()
  parser.add_argument('--mel_dir', default='mel_spects')
  parser.add_argument('--audio_dir', default='speech_clips')
  parser.add_argument('--output_dir', default='mel_display') #None)
  parser.add_argument('--config_file',  default='bigvgan_24khz_100band_config.json')
  parser.add_argument('--checkpoint_file',  default='bigvgan_24khz_100band-20230502T202754Z/bigvgan_24khz_100band/g_05000000.zip')

  a = parser.parse_args()

  if a.output_dir:
     if not os.path.exists(a.output_dir):
        os.makedirs(a.output_dir)
  
  mel_list = os.listdir(a.mel_dir)
  mel_samples = random.sample(mel_list,2)
  print(mel_samples)

  mel_file = mel_samples[0]
  mel = np.load(os.path.join(a.mel_dir,mel_file))
  if len(mel.shape)==3:
      mel = mel.squeeze(0)

  if os.path.exists(a.config_file):
     h = get_config(a.config_file)
  elif os.path.exists(a.checkpoint_file): 
     h = get_config(a.checkpoint_file)
  else: 
     config_file = input('Please enter the config file path')
     h = get_config(config_file)

  plot_mel(mel, h, ax=None,tytle=mel_file)
#   mel_path = os.path.join(a.mel_dir,mel_file)
#   audio_path = os.path.join(a.audio_dir,mel_file)
#   show_mel_audio(mel_path,audio_path,h)


if __name__ == "__main__":
  main()





