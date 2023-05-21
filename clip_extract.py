# from each file in 'audio_dir'
# extracts 'sample_number' clips of duration 'sample_duration' seconds

# imports
import os
import argparse
import soundfile
import librosa
import random
import numpy as np


# functions
def clip_strip(source_dir, 
               clip_dir, 
               clip_duration):
  # Break all audio files in source directory into short clips 
  # create target directory if it doesn't exist
  if not os.path.exists(clip_dir):
      os.makedirs(clip_dir)

  # loop through each audio file in the source directory
  for filename in os.listdir(source_dir):
      if filename.endswith('.mp3'):  # check if file is an MP3
          # load the audio file and get its duration in seconds
          filepath = os.path.join(source_dir, filename)
          print(f'clip being created from: {filepath}')
          y, sr = librosa.load(filepath)
          duration = librosa.get_duration(y=y, sr=sr)
          
          # loop through each 10-second clip and save it to the target directory
          for i in range(int(duration / clip_duration)):
              start_time = i * clip_duration
            #   end_time = start_time + clip_duration
              clip, _ = librosa.load(filepath, offset=start_time, duration=clip_duration)
              clip_filename = f'{os.path.splitext(filename)[0]}_{i+1}.mp3'
              clip_filepath = os.path.join(clip_dir, clip_filename)
              soundfile.write(clip_filepath, clip, sr) #, 'PCM_24')

def sample_clips(source_dir, 
                 clip_dir, 
                 clip_duration,
                 clip_number):
  # Sample short clips from each audio files in source directory 
  # create target directory if it doesn't exist
  if not os.path.exists(clip_dir):
      os.makedirs(clip_dir)

  # loop through each audio file in the source directory
  for filename in os.listdir(source_dir):
      if filename.endswith('.mp3'):  # check if file is an MP3
          # load the audio file and get its duration in seconds
          filepath = os.path.join(source_dir, filename)
          print(f'clip being created from: {filepath}')
          y, sr = librosa.load(filepath)
          duration = librosa.get_duration(y=y, sr=sr)
          # start sample at a random time before 'duration' - clip_duration
          # get clips
          for i in np.arange(clip_number):
              start_time = random.randrange(np.floor(duration)-clip_duration)
              clip, _ = librosa.load(filepath, offset=start_time, duration=clip_duration)
              clip_filename = f'{os.path.splitext(filename)[0]}_{i+1}.mp3'
              clip_filepath = os.path.join(clip_dir, clip_filename)
              soundfile.write(clip_filepath, clip, sr) #, 'PCM_24')   


# main

def main():

  parser = argparse.ArgumentParser()
  parser.add_argument('--audio_dir', default='speech_audio')
  parser.add_argument('--output_dir', default='speech_clips')
  parser.add_argument('--sample_number', default=1)
  parser.add_argument('--sample_duration', default=10)

  a = parser.parse_args()


  if not a.sample_number:
    #extract all 
    clip_strip(a.audio_dir,
               a.output_dir,
               a.sample_duration)
  else: 
    # extract a.sample_number
    sample_clips(a.audio_dir, 
                 a.output_dir, 
                 a.sample_duration,
                 a.sample_number)


if __name__ == "__main__":
  main()