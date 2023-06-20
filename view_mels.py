# approaches for viewing mels generated for DL GANs
# initial format is: 

# imports
import os, argparse
import matplotlib.pyplot as plt
import librosa, librosa.display
import json
import random
import numpy as np
import threading
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
    plt.show(block=False)

def compare_mels(mel0, mel1):
  print('working on it')
   
def show_mel_audio(mel,
                   audio,
                   h,
                   ax=None):
   #### RUN FROM mel & audio raw, not filename...
   fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10,5))

   # Display the spectrogram
   plt.sca(axes[0])

   # Handle mel input
   # allow input of string for mel
   if isinstance(mel,str):
      if not os.path.exists(mel):
         mel = input('Please specify path to MEL spectrogram:')
      mel = np.load(mel)
   # Useful if mel is torch-derived
   if len(mel.shape)==3:
      mel = mel.squeeze(0)
   # end up with np array
   assert(isinstance(mel,np.ndarray))
  
  # Handle audio input
   if isinstance(audio,str):
     if not os.path.exists(audio):
         audio = input('Please specify path to audio file:')
     audio = librosa.load(audio, sr=h.sampling_rate, mono=True)
   assert(isinstance(audio,np.ndarray))
   
   librosa.display.specshow(mel, 
                            y_axis='mel', 
                            fmax=h.fmax, 
                            hop_length=h.hop_size, 
                            sr=h.sampling_rate,
                            x_axis='time')  
   axes[0].set(xlabel='time', ylabel='frequency')
   plt.colorbar(format='%+2.0f dB')
   # plt.title(os.path.split(mel_file)[-1])
   plt.tight_layout()
   plt.show(block=False)
   
   music_thread = threading.Thread(target=play, args=(sound,))
   
   #ADD ANIMATION...
   #ADD AUDIO


   # Display audio play button
   # plt.sca(axes[1])
   # wn = Audio(audio_file, autoplay=True, rate=h.sampling_rate)
   # display(wn) 


# main
def main():
  print('Viewing mels')

  parser = argparse.ArgumentParser()
  parser.add_argument('--mel_dir', default='mel_spects')
  parser.add_argument('--audio_dir', default='speech_clips')
  parser.add_argument('--random_sample', default=True)
  parser.add_argument('--sample_amount', default=2)
  parser.add_argument('--output_dir', default='None')
  parser.add_argument('--config_file',  default='bigvgan_24khz_100band_config.json')
  parser.add_argument('--checkpoint_file',  default='bigvgan_24khz_100band-20230502T202754Z/bigvgan_24khz_100band/g_05000000.zip')

  a = parser.parse_args()

  # Get config path. Contains MEL parameters, e.g. hoplength & sampling rate
  if os.path.exists(a.config_file):
     h = get_config(a.config_file)
  elif os.path.exists(a.checkpoint_file): 
     h = get_config(a.checkpoint_file)
  else: 
     config_file = input('Please enter the config file path')
     h = get_config(config_file)

  # Get random samples, can also make it st file list works as input.
  if a.random_sample:
    # Find random file
    mel_list = os.listdir(a.mel_dir)
    mel_samples = random.sample(mel_list,a.sample_amount)
    print(mel_samples)
  else: 
    input('Please input mel spec file path:')

  # For each sample: load mel, load audio, then make a figure.
  for file_name in mel_samples:
    mel = np.load(os.path.join(a.mel_dir,file_name))
    if len(mel.shape)==3:
      mel = mel.squeeze(0)
    audio_fn = os.path.splitext(file_name)[0]+'.mp3'
    audio, sr = librosa.load(os.path.join(a.audio_dir, audio_fn), sr=h.sampling_rate, mono=True)
   #  plot_mel(mel, h, ax=None, tytle=file_name)
    show_mel_audio(mel, audio, h)


  # If saving create the directory, then save the figure
  if a.output_dir:
     if not os.path.exists(a.output_dir):
        os.makedirs(a.output_dir)
      # save it

  #Necessary to keep the debugger mode from closing figures.
  plt.show()

if __name__ == "__main__":
  main()





