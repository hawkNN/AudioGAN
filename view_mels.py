# approaches for viewing mels generated for DL GANs
# initial format is: 

# imports
import os, argparse
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import librosa, librosa.display
import json
import random
import torch
import numpy as np
import threading
from IPython.display import Audio, display
from pydub.playback import play
from matplotlib.animation import FuncAnimation
from moviepy.editor import VideoFileClip, AudioFileClip

from make_a_mel import generate_mel, mel_Tensor_to_np

# Overall Status:
#  Produces reasonable quality audio-MEL display video with good timing and window
# Issues: 
#  1. Current MEL generation in animate_mel_spectrogram isn't a good range & probably won't work well with bigVgan
#      Solution: incorporate make_mels.py methods (from bigVgan) to produce the MEL.

#Improvements: 
#  1. Add a transverse wave representation panel? Definitely here in this function as a two-panel view. Compression wave?
#  2. Should paired comparisons be in a separate function? I think so, it would be better to be able to call each function to achieve a goal
#   What about libraries? Can I plug related functions into a common sub-library contained within the overarching AudioGAN library
#  3. I need to setup comments/help so argument role is clear, copy other files, e.g. look at numpy files.
#     For instance, audio_dir & audio_sample are independent but only one is required.
#   
#Remnants:
#  1. Is plot_mel used any more? Should I keep it or make a simpler static demo? 
#  2. Is mel_dir input useful any more?


# functions
def get_config(file_path):    # produce global h variable with mel config parameters
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
  h=AttrDict(json_config)
  return h

def spectral_normalize_torch(magnitudes): # torch-implemented mel dB conversion
   output = dynamic_range_compression_torch(magnitudes)
   return output

def dynamic_range_decompression_torch(x, C=1): # torch-implemented mel range 
   return torch.exp(x) / C

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5): # torch-implemented dB conversion
   return torch.log(torch.clamp(x, min=clip_val) * C)

def mel_from_wav(input_wav, h, center=False):   # produce a single torch-implemented mel spectogram
    # MAX_WAV_VALUE = 32768.0
    if 'mel_basis' not in globals():
        global mel_basis
        mel_basis = {}
    if 'hann_window' not in globals():
        global hann_window
        hann_window={}

    if torch.min(input_wav) < -1.:
        print('min value is ', torch.min(input_wav))
    if torch.max(input_wav) > 1.:
        print('max value is ', torch.max(input_wav))

    if h.fmax not in mel_basis:
        mel = librosa.filters.mel(sr=h.sampling_rate, 
                                    n_fft=h.n_fft, 
                                    n_mels=h.num_mels, 
                                    fmin=h.fmin, 
                                    fmax=h.fmax)
        mel_basis[str(h.fmax)+'_'+str(input_wav.device)] = torch.from_numpy(mel).float().to(input_wav.device)
        hann_window[str(input_wav.device)] = torch.hann_window(h.win_size).to(input_wav.device)

    input_wav = torch.nn.functional.pad(input_wav.unsqueeze(1), (int((h.n_fft-h.hop_size)/2), int((h.n_fft-h.hop_size)/2)), mode='reflect')
    input_wav = input_wav.squeeze(1)

    # complex tensor as default, then use view_as_real for future pytorch compatibility
    spec = torch.stft(input_wav, h.n_fft, hop_length=h.hop_size, win_length=h.win_size, window=hann_window[str(input_wav.device)],
                        center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)
    spec = torch.view_as_real(spec)
    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

    spec = torch.matmul(mel_basis[str(h.fmax)+'_'+str(input_wav.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec

def generate_mel(audio_path,h): # produce mel spectrogram as tensor

    if 'device' not in globals():
        global device 
        if torch.cuda.is_available():
            torch.cuda.manual_seed(h.seed)
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')


    wav, sr = librosa.load(audio_path, sr=h.sampling_rate, mono=True)
    wav = torch.FloatTensor(wav).to(device)
    # compute mel spectrogram from the ground truth audio
    mel_spectrogram = mel_from_wav(wav.unsqueeze(0),h)
    return mel_spectrogram

def mel_Tensor_to_np(mel_tensor): # convert mel as tensor to mel as np for plotting
    mel_np=mel_tensor.detach().numpy()
    mel_np=np.reshape(mel_np,(np.shape(mel_np)[1],np.shape(mel_np)[2]))
    print(f'shape of mel is {np.shape(mel_np)} of type {mel_np.dtype}')
    return mel_np

def paint_it_black(fig, ax, cb): # white-on-black figure conversion
   fig.patch.set_facecolor('xkcd:black')
   ax.set_facecolor((0.06,0.06,0.06))
   ax.spines['bottom'].set_color('white')
   ax.spines['top'].set_color('white')
   ax.spines['left'].set_color('white')
   ax.spines['right'].set_color('white')
   ax.xaxis.label.set_color('white')
   ax.yaxis.label.set_color('white')
   ax.grid(alpha=0.1)
   ax.title.set_color('white')
   ax.tick_params(axis='x', colors='white')
   ax.tick_params(axis='y', colors='white')
   # cb.set_label('dB', color='white')
   cb.ax.yaxis.set_tick_params(color='white')
   plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='white')

def plot_mel(mel, # single figure for MEL spectrogram
            h,
            fig=None,
            ax=None,
            tytle='MEL spectrogram',
            figure_size = (10,5)):  #
   # get dims for mel, compress to 2-D if needed.
   # if no axis specified create singleton.
   if not ax:
      fig, ax = plt.subplots(figsize=figure_size)

   # Display the spectrogra
   plt.sca(ax)
   mel_plot = librosa.display.specshow(mel, 
                                       y_axis='mel', 
                                       fmax=h.fmax, 
                                       hop_length=h.hop_size, 
                                       sr=h.sampling_rate,
                                       x_axis='time')    

   ax.set(xlabel='time', ylabel='frequency')
   cb = plt.colorbar(format='%+2.0f dB')
   paint_it_black(fig, ax, cb) # convert to black background with white foreground 
   plt.title(tytle)
   plt.tight_layout()
   plt.show(block=False)

   return mel_plot

def plot_difference_spectrogram(mel_spectrogram1, # plot difference between to spectrograms
                                mel_spectrogram2, 
                                h=None,
                                fig=None,
                                ax=None,
                                figure_size=(10,5),
                                title='Difference Spectrogram'):
   
   # if no axis specified create singleton.
   if not ax:
      fig, ax = plt.subplots(figsize=figure_size)

   # default hop & sample rate if not defined with h
   # Get parameter values used for bigvgan, input through variable 'h'
   if h:
      sr = h.sampling_rate
      fmax = h.fmax
      hop = h.hop_size
   else: 
      # Defaults based on bigvgan 24khz 100band config
      Warning('No spectrogram parameters input in plot_differenc_spectrogram(), using default values')
      sr = 24000 #samples/second
      fmax = 12000 # maximal frequency
      hop = 256 # mel hop size used for fft to create mel spectrogram
      # Calculate the absolute difference spectrogram
      difference_spectrogram = np.abs(mel_spectrogram1 - mel_spectrogram2)

    # Apply logarithmic scaling for better visualization
    difference_spectrogram = np.log(1 + difference_spectrogram)

    plt.title(title)
    librosa.display.specshow(difference_spectrogram, cmap='viridis', sr=sr, hop_length=hop)
    plt.colorbar(format='%+2.0f dB')

    return 

def mel_input_check(mel): # handles mel if path or np array
   # Handle mel if path used as input
   if isinstance(mel,str):
      if not os.path.exists(mel):
         mel = input('Please specify path to MEL spectrogram:')
      mel = np.load(mel)
   # Useful if mel is torch-derived
   if len(mel.shape)==3:
      mel = mel.squeeze(0)
   # end up with np array
   assert(isinstance(mel,np.ndarray))
   return mel

def audio_input_check(audio, sampling_rate=None): # handles audio path to wav conversion
   # Handle audio if path used as input
   if isinstance(audio,str):
      if not os.path.exists(audio):
         audio = input('Please specify path to audio file:')
      if sampling_rate:
         audio, sampling_rate = librosa.load(audio, sr=sampling_rate, mono=True)
      else:
         audio, sampling_rate = librosa.load(audio,sr=None,mono=True)
   assert(isinstance(audio,np.ndarray))
   return audio, sampling_rate

def animate_mel_spectrogram(audio_file_path, # creates animated mel visualization 
                            output_video_path='../GitOutbox',
                            h=None, # parameters of audio => MEL conversion, based on bigvgan
                            mel_window= 5, # seconds
                            tytle='Mel spectrogram',
                            verbose=None):
   # Get parameter values used for bigvgan, input through variable 'h'
   if h:
      sr = h.sampling_rate
      fmax = h.fmax
      hop = h.hop_size
   else: 
      # Defaults based on bigvgan 24khz 100band config
      Warning('No spectrogram parameters input in animate_mel_spectrogram(), using default values')
      sr = 24000 #samples/second
      fmax = 12000 # maximal frequency
      hop = 256 # mel hop size used for fft to create mel spectrogram

   ### Replacing with functions from make_a_mel.py 
   ### based on bigvgan settings for mel spectrogram
   # # load audio
   # audio_wav, sr = audio_input_check(audio_file_path,
   #                           sampling_rate=sr)
   # # Compute mel spectrogram
   # mel_spectrogram = librosa.feature.melspectrogram(y=y,
   #                                                  fmax=fmax, 
   #                                                  hop_length=hop, 
   #                                                  win_length=mel_window,
   #                                                  sr=sr)
   # mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

   mel_spectrogram = generate_mel(audio_file_path, h)
   mel_spectrogram = mel_Tensor_to_np(mel_spectrogram)
   # note: now requiring h, so make sure it has the default parameter values assigned above

   # Calculate maximum MEL duration, mel_window, to display in frames for max_frame
   ###### CORRECT CALCULATION HERE  #########
   max_frame = int(np.ceil((mel_window * h.sampling_rate) / h.hop_size))
   min_mel_value = np.min(mel_spectrogram)
   if verbose:
      print(f'max_frame is: {max_frame} MEL frames')
      print(f'mel shape is: {np.shape(mel_spectrogram)} ')
      print(f'min_mel_value is: {min_mel_value} dB')
      print(f'mel_window: {mel_window} secs')
      print(f'h.sampling rate: {h.sampling_rate} samples/sec')
      print(f'h.hop_size: {h.hop_size} samples')
   
   # Create figure and axis for animation
   fig, ax = plt.subplots()
   im = ax.imshow(mel_spectrogram, aspect='auto', origin='lower', cmap='viridis')
   
   # using set_bad/NaN to update unused frames as black
   current_cmap = cm.get_cmap()
   current_cmap.set_bad(color='black')

   # improve illustration of plot
   cb = plt.colorbar(im, format="%+2.0f dB")
   ax.set_title(tytle)
   ax.set(xlabel='time', ylabel='frequency')
   paint_it_black(fig, ax, cb) # convert to black background with white foreground 
   plt.title(tytle)
   plt.tight_layout()
   plt.show(block=False)

   # # ORIGINAL METHOD for frame update, works but xlim dynamic (labels not)
   # # Define a function to update the animation
   # def update(frame):
   #    im.set_data(mel_spectrogram_db[:, :frame]) 
   #    return im,

   # Attempting to keep constant xlim...
   # Define a function to update the animation
   # WORKS great in terms of timing, but I don't think the cmap is working out...
   def update(frame):
      # make a single zero-filled frame up to max_frame, then rolling window after filled 
      if frame<max_frame:
         # make zeros to fill with current data.
         mel_dummy = np.empty((np.shape(mel_spectrogram)[0],(max_frame-1))) # MAYBE NANs better? 
         mel_dummy.fill(np.nan)
         mel_dummy[:,:frame] = mel_spectrogram[:, :frame]
         im.set_data(mel_dummy) 
      else:
         im.set_data(mel_spectrogram[:, (frame-max_frame):frame]) 
      return im,

   # Calculate total frames for animation
   total_frames = mel_spectrogram.shape[1]
   output_fps = int(sr/hop) # mel 'frames' relate to video time by sr/hop_size

   # Create the animation
   ani = FuncAnimation(fig, update, frames=total_frames, blit=False, repeat=False)

   if output_video_path:
      output_video_path = output_video_path.lower()
      if output_video_path.endswith(('.mp4', '.avi', '.gif')):
         ani.save(output_video_path, writer='ffmpeg', fps=output_fps)
         print(f"Saved animation video to: {output_video_path}")
      else:
         print("Output video format not supported. Please use '.mp4', '.avi', or '.gif'.")
   else:
      plt.show()

def combine_audio_and_video(audio_file_path, # adds synchronized audio to mel animation
                            video_file_path, 
                            output_video_path,
                            sampling_rate=24000):
    #Load audio & video clips
    audio_clip = AudioFileClip(audio_file_path, fps=sampling_rate)
    video_clip = VideoFileClip(video_file_path, audio_fps=sampling_rate)
    
    # Set audio of the video to the provided audio clip
    video_clip = video_clip.set_audio(audio_clip)
    
    # Write the combined video to the output path
    video_clip.write_videofile(output_video_path, codec='libx264', audio_codec='aac')

# main
def main():
  
  print('Viewing mels')

  parser = argparse.ArgumentParser()
  parser.add_argument('--audio_dir', default='speech_clips')
  parser.add_argument('--mel_dir', default='mel_spects')
  parser.add_argument('--audio_sample', default=False) # Change to name... if empty, create random?
  parser.add_argument('--sample_amount', default=1)
  parser.add_argument('--output_dir', default='../GitOutbox')
  parser.add_argument('--config_file',  default='bigvgan_24khz_100band_config.json')
  parser.add_argument('--checkpoint_file',  default='bigvgan_24khz_100band-20230502T202754Z/bigvgan_24khz_100band/g_05000000.zip')

  a = parser.parse_args()


  # If saving, create the directory, then save the figure
  if a.output_dir:
     if not os.path.exists(a.output_dir):
        os.makedirs(a.output_dir)
  
  # Get config path. Contains MEL parameters, e.g. hoplength & sampling rate
  if os.path.exists(a.config_file):
     h = get_config(a.config_file)
  elif os.path.exists(a.checkpoint_file): 
     h = get_config(a.checkpoint_file)
  else: 
     config_file = input('Please enter the config file path')
     h = get_config(config_file)

  # Get random samples, can also make it st file list works as input.
  # BASE around audio_dir, because I do NOT need pre-existing MELs
  if a.audio_sample:
    # Create list with audio sample.
    audio_samples = {a.audio_sample}
  else: 
    # Find random file
    audio_list = os.listdir(a.audio_dir)
    audio_samples = random.sample(audio_list,a.sample_amount)
    print(f'creating mel spectrogram animations for: {audio_samples}')
   #  input('Please input mel spec file path:')

#  For each sample: load mel, load audio, then make a figure.
  for file_name in audio_samples:
   #  Note: changing to just ID path, then enter, no need to have pre-existing mel or preloaded audio
   #   if a.mel_dir:
   #      mel = np.load(os.path.join(a.mel_dir,file_name))
   #       if len(mel.shape)==3:
   #          mel = mel.squeeze(0)
    audio_file_path = os.path.join(a.audio_dir, file_name)
    print(f'Examining audio file: {audio_file_path}')
    #  audio, sr = librosa.load(os.path.join(a.audio_dir, audio_file_path), sr=h.sampling_rate, mono=True)
   #  plot_mel(mel, h, ax=None, tytle=file_name)
    #  show_mel_audio(mel, audio, h, tytle=file_name)

    video_file_path = os.path.join(a.output_dir,'temp.mp4')
    av_file_path = os.path.join(a.output_dir,os.path.splitext(file_name)[0]+'.mp4')

    animate_mel_spectrogram(audio_file_path, 
                            video_file_path, 
                            h,
                            tytle=file_name)
    combine_audio_and_video(audio_file_path,
                            video_file_path,
                            av_file_path, 
                            sampling_rate=h.sampling_rate)
  #Necessary to keep the debugger mode from closing figures.
  plt.show()

if __name__ == "__main__":
  main()
