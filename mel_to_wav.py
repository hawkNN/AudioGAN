# Adapted from https://github.com/NVIDIA/BigVGAN
# which was adapted from https://github.com/jik876/hifi-gan under the MIT license.
#   LICENSE is in incl_licenses directory.

from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import numpy as np
import argparse
import json
import torch
from scipy.io.wavfile import write
from env import AttrDict
from meldataset import MAX_WAV_VALUE
from models import BigVGAN as Generator

h = None
device = None
torch.backends.cudnn.benchmark = False


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]


def inference(a, h):
    generator = Generator(h).to(device)

    state_dict_g = load_checkpoint(a.checkpoint_file, device)
    generator.load_state_dict(state_dict_g['generator'])

    filelist = os.listdir(a.input_mels_dir)

    os.makedirs(a.output_dir, exist_ok=True)

    generator.eval()
    generator.remove_weight_norm()
    with torch.no_grad():
        for i, filname in enumerate(filelist):
            # load the mel spectrogram in .npy format
            x = np.load(os.path.join(a.input_mels_dir, filname))
            x = torch.FloatTensor(x).to(device)
            if len(x.shape) == 2:
                x = x.unsqueeze(0)

            y_g_hat = generator(x)

            audio = y_g_hat.squeeze()
            audio = audio * MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype('int16')

            output_file = os.path.join(a.output_dir, os.path.splitext(filname)[0] + '_generated_e2e.wav')
            write(output_file, h.sampling_rate, audio)
            print(output_file)


def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_mels_dir', default='mel_spects')
    parser.add_argument('--output_dir', default='bigvgan_output')
    parser.add_argument('--config_file', default='bigvgan_24khz_100band_config.json')
    parser.add_argument('--checkpoint_file', default='../checkpoint_files/bigvgan_24khz_100band-20230502T202754Z.zip') #required=True)

    a = parser.parse_args()

    config_path = a.config_file
    if not os.path.exists(config_path):
        print(f'config filepath does not exist: {config_path}')
        config_path = os.path.join(os.path.split(a.checkpoint_file)[0], 'config.json').replace("\\","/")
        print(f'searching in checkpoint folder: {config_path}')
    if not os.path.exists(config_path):
        print(f'config filepath does not exist: {config_path}')
        config_path = input('Please supply config file path:')
        print(f'trying: {config_path}')
    if not os.path.exists(a.output_dir):
        os.mkdir(a.output_dir)
    if not os.path.exists(a.input_mels_dir):
        a.input_mels_dir = input('Please enter path to MEL directory:')
    
    with open(config_path) as f:
        data = f.read()
    print(f'Config path is: {config_path}')
    
    if not os.path.exists(a.checkpoint_file):
        print(f'checkpoint file does not exist: {a.checkpoint_file}')
        a.checkpoint_file = input('Please input path to checkpoint file:')
    print(f'checkpoint file is: {a.checkpoint_file}')

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    inference(a, h)


if __name__ == '__main__':
    main()

