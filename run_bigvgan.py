# implements https://github.com/NVIDIA/BigVGAN under the MIT License


# Notes, 
    # May not want os.chdir in find_bigvgan

# If bigvgan repo present, cd to that repo.
# Else clone it. 
# Then feed mel-directory into inference_e2e.py

import os, argparse
from git import Repo


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
    os.chdir(repo_dir) # may be sloppy, perhaps better to just build right paths for execution.
    print(f'repo found at: {os.path.abspath(repo_dir) }')
    return os.path.abspath(repo_dir) 
  else: 
    # os.makedirs(repo_dir)
    Repo.clone_from (repo_url, repo_dir)
    os.chdir(repo_dir) # may be sloppy, perhaps better to just build right paths for execution.
    print(f'repo cloned to: {os.path.abspath(repo_dir) }')
    return os.path.abspath(repo_dir) 



def main():
  
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint_file', default='..\checkpoint_files\bigvgan_24khz_100band-20230502T202754Z\bigvgan_24khz_100band\g_05000000.zip')
  parser.add_argument('--mel_dir', default='mel_spects')
  parser.add_argument('--bigvgan_dir', default='../bigvgan')
  parser.add_argument('--output_dir', default='bigvgan_output')

  a = parser.parse_args()

#   print(f'Downloading mp3 files from {a.target_url}')
#   print(f'Download limit {a.max_files} files')
#   print(f'Saving in {a.output_dir}')
  
  if not os.path.exists(a.output_dir):
    os.mkdir(a.output_dir)

  repo_dir = confirm_repo(a.bigvgan_dir) # assign abs_path output? May help direct other commands

  

if __name__ == "__main__":
  main()