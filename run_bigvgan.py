# implements https://github.com/NVIDIA/BigVGAN under the MIT License

import os, argparse
from git import Repo
# requirements for bigvgan
import torch
import numpy as np
import librosa
import scipy
import tensorboard
import soundfile
import matplotlib as plt
import pesq
import auraloss
import tqdm

# !git clone https://github.com/NVIDIA/BigVGAN
# %cd /BigVGAN
# !pip install -r requirements.txt

# !python inference_e2e.py \
# --checkpoint_file $checkpoint_path \
# --input_mels_dir $MELspect_dir  \
# --output_dir $BigVGANout_dir #/path/to/your/output_wav
# Set MEL band to 100

def confirm_repo(repo_dir,
                 repo_url = 'https://github.com/NVIDIA/BigVGAN',
                 sample_file = 'inference_e2e.py'
                 ):
  if os.path.exists(os.path.join(repo_dir,sample_file)):
    print(f'repo found at: {os.path.abspath(repo_dir) }')
    return os.path.abspath(repo_dir) 
  else: 
    Repo.clone_from (repo_url, repo_dir)
    print(f'repo cloned to: {os.path.abspath(repo_dir) }')
    return os.path.abspath(repo_dir) 



def main():
  
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint_path', default='..\checkpoint_files\bigvgan_24khz_100band-20230502T202754Z\bigvgan_24khz_100band\g_05000000.zip')
  parser.add_argument('--mel_dir', default='mel_spects')
  parser.add_argument('--bigvgan_dir', default='./bigvgan')
  parser.add_argument('--output_dir', default='./GitOutbox/bigvgan_output')
  parser.add_argument('--verbose', default=True)

  a = parser.parse_args()

  if a.verbose:
    print(f'Downloading mp3 files from {a.target_url}')
    print(f'Download limit {a.max_files} files')
    
  if not os.path.exists(a.output_dir):
    try:
      os.mkdir(a.output_dir)
    except:
      a.output_dir = input("Please specify output path:")
      if not os.path.exists(a.output_dir):
        os.mkdir(a.output_dir)
  if a.verbose:
    print(f'storing bigVgan output in {a.output_dir}')

  repo_dir = confirm_repo(a.bigvgan_dir) # assign abs_path output? May help direct other commands

  inference_cmd = (os.path.join(repo_dir,'inference_e2e.py') + ' '
                   '--checkpoint_file ' + a.checkpoint_path +
                   '--input_mels_dir ' + a.mel_dir +
                   '--output_dir ' + a.output_dir)
  print(inference_cmd)
  os.system(inference_cmd)


  # check requirements file... currently just added as imports above



  

if __name__ == "__main__":
  main()