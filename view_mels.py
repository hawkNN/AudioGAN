# approaches for viewing mels generated for DL GANs
# initial format is: 

# imports
import os, argparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import librosa, librosa.display
import json
import random
import numpy as np
import threading
from IPython.display import Audio, display
from pydub.playback import play
from matplotlib.animation import FuncAnimation
from moviepy.editor import VideoFileClip, AudioFileClip

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

def paint_it_black(fig, ax, cb):
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

def plot_mel(mel,
            h,
            clip_duration,
            ax=None,
            fig=None,
            tytle='MEL spectrogram',
            figure_size = (10,5)):
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
   ax.set_xlim([0,clip_duration]) # set xlim for entire clip duration
   cb = plt.colorbar(format='%+2.0f dB')
   paint_it_black(fig, ax, cb) # convert to black background with white foreground 
   plt.title(tytle)
   plt.tight_layout()
   plt.show(block=False)

   return mel_plot

def update_mel(mel, 
               h,
               tnow):
   freq_bins = mel.shape[0] # number of y axis frequency bins
   mel_frame = np.floor_divide(tnow*h.sampling_rate,h.hop_size) # convert tnow in seconds to mel frame
   mel_fragment = mel[0:freq_bins,0:mel_frame] # partial mel at tnow(sec) converted to mel_frame
   return mel_fragment

def compare_mels(mel0, mel1):
   print('side-by-side & delta images?')

def mel_input_check(mel):
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

def audio_input_check(audio, sampling_rate=None):
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

def animate_mel_spectrogram(audio_file_path, 
                            output_video_path='../GitOutbox',
                            h=None,
                            tytle='Mel spectrogram'):
   # NEED TO DO: set h default values to work
   if h:
      sr = h.sampling_rate
      fmax = h.fmax
      hop = h.hop_size
   else: 
      # I NEED TO CHOOSE Defaults that generally work...
      Warning('No spectrogram parameters input in animate_mel_spectrogram(), using default values')
      sr = None
      fmax = None
      hop = None
   # Load audio if path
   y, sr = audio_input_check(audio_file_path,
                             sampling_rate=sr)

   # Compute mel spectrogram
   mel_spectrogram = librosa.feature.melspectrogram(y=y,
                                                    fmax=fmax, 
                                                    hop_length=hop, 
                                                    sr=sr)
   mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

   clip_duration = np.floor_divide(len(y),sr) # duration of entire audio file in seconds

   # Create figure and axis for animation
   fig, ax = plt.subplots()
   im = ax.imshow(mel_spectrogram_db, aspect='auto', origin='lower', cmap='viridis')
   cb = plt.colorbar(im, format="%+2.0f dB")
   ax.set_title(tytle)
   ax.set(xlabel='time', ylabel='frequency')
   ## FIX. this changes label without actually changing the range... WHY?????
   # ax.set_xlim([0,clip_duration]) # set xlim for entire clip duration
   paint_it_black(fig, ax, cb) # convert to black background with white foreground 
   plt.title(tytle)
   plt.tight_layout()
   plt.show(block=False)

   # Define a function to update the animation
   def update(frame):
      im.set_data(mel_spectrogram_db[:, :frame])
      # really, I want to set data to entirety but just black from 'frame:-1'
      return im,

   # Calculate total frames for animation
   total_frames = mel_spectrogram_db.shape[1]
   output_fps = int(sr/hop) # mel 'frames' relate to video time by sr/hop_size

   # Create the animation
   ani = FuncAnimation(fig, update, frames=total_frames, blit=True, repeat=False)

   if output_video_path:
      output_video_path = output_video_path.lower()
      if output_video_path.endswith(('.mp4', '.avi', '.gif')):
         ani.save(output_video_path, writer='ffmpeg', fps=output_fps)
         print(f"Saved animation video to: {output_video_path}")
      else:
         print("Output video format not supported. Please use '.mp4', '.avi', or '.gif'.")
   else:
      plt.show()

def combine_audio_and_video(audio_file_path, 
                            video_file_path, 
                            output_video_path,
                            sampling_rate=22050):
    #Load audio & video clips
    audio_clip = AudioFileClip(audio_file_path, fps=sampling_rate)
    video_clip = VideoFileClip(video_file_path, audio_fps=sampling_rate)
    
    # Set audio of the video to the provided audio clip
    video_clip = video_clip.set_audio(audio_clip)
    
    # Write the combined video to the output path
    video_clip.write_videofile(output_video_path, codec='libx264', audio_codec='aac')

def show_mel_audio(mel,
                  audio,
                  h,
                  fig=None,
                  axes=None,
                  select_ax=0,
                  tytle='MEL spectrogram',
                  figure_size = (10,5)):
   # create figure if not input.
   if (not axes) or (not fig):
      fig, axes = plt.subplots(nrows=1, ncols=1, figsize=figure_size)

   current_ax = fig.axes[select_ax]  # attend to axes of interest if multiaxis figure

   #Handle input variance
   mel = mel_input_check(mel)
   audio = audio_input_check(audio)

   clip_duration = np.floor_divide(len(audio),h.sampling_rate) # duration of entire audio file in seconds

   # Plot background MEL
   tnow = 10 #in seconds, adjust in a loop synchronized with playing audio
   # tnow needs to be update in a loop to animate plot.

   mel_fragment = update_mel(mel,h,tnow) # create mel fragment updated based on current time...
   # 'artist' should update with this...

   plt.sca(current_ax) 
   plot_mel(mel_fragment,
            h,
            clip_duration,
            ax=current_ax,
            fig=fig,
            tytle=tytle,
            figure_size = figure_size)


   #ADD ANIMATION to indicate time in MEL spect, update mel_plot over time.
   # Grow MEL, or plot increasing x components... FuncAnimation()
   # simply limit 'mel_fragment' to tnow & plot with plot_mel....

   #ADD AUDIO
   music_thread = threading.Thread(target=play, args=(audio,))

   # Display audio play button
   # plt.sca(axes[1])
   # wn = Audio(audio_file, autoplay=True, rate=h.sampling_rate)
   # display(wn) 

   # Call animate function to animate fig
   # anim = FuncAnimation(fig, animate, init_func=init, interval=55)
   # plt.show()

# main
def main():
  
  # Overall notes:
  #   1. Timing not synchronized, is it related to frame rate of images? 
  #   2. If no h parameters loaded, create h with the parameters that are default of bigvgan
  print('Viewing mels')

  parser = argparse.ArgumentParser()
  parser.add_argument('--audio_dir', default='speech_clips')
  parser.add_argument('--mel_dir', default='mel_spects')
  parser.add_argument('--random_sample', default=True)
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
  if a.random_sample:
    # Find random file
    audio_list = os.listdir(a.audio_dir)
    audio_samples = random.sample(audio_list,a.sample_amount)
    print(f'creating mel spectrogram animations for: {audio_samples}')
  else: 
    input('Please input mel spec file path:')

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
    ## ISSUE: animation is stretch from 10sec to 46sec
    # B/C of hop length, video is 938 frames: 10sec * 24000sr/256hopsize
    # video fps should be determined by these numbers: int(sr/hop_size)?
    # 
    ###
    # Also, video is ugly, paint_it_black() & better labeling
    animate_mel_spectrogram(audio_file_path, 
                            video_file_path, 
                            h,
                            tytle=file_name)
    combine_audio_and_video(audio_file_path,
                            video_file_path,
                            av_file_path, 
                            sampling_rate=h.sampling_rate)

  #Necessary to keep the debugger mode from closing figures.
#   plt.show()

if __name__ == "__main__":
  main()
