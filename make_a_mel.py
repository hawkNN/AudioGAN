
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import json
import torch
import librosa, librosa.display

def get_h(config_path):
    global h #hyperparameters for audio & MEL # don't need to define as global here.

    # used to store h parameters for mel
    class AttrDict(dict):
     def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    if not os.path.exists(config_path):
        print(f'config filepath does not exist: {config_path}')
        config_path = input('Please supply config file path:')
    with open(config_path) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)

    ## moved to generate_mel()
    # global device 
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(h.seed)
    #     device = torch.device('cuda')
    # else:
    #     device = torch.device('cpu')
    return h
   
def spectral_normalize_torch(magnitudes):
   output = dynamic_range_compression_torch(magnitudes)
   return output

def dynamic_range_decompression_torch(x, C=1):
   return torch.exp(x) / C

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
   return torch.log(torch.clamp(x, min=clip_val) * C)

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
            fig=None,
            ax=None,
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
   cb = plt.colorbar(format='%+2.0f dB')
   paint_it_black(fig, ax, cb) # convert to black background with white foreground 
   plt.title(tytle)
   plt.tight_layout()
   plt.show(block=False)

   return mel_plot

def mel_from_wav(input_wav, h, center=False):
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

def mel_Tensor_to_np(mel_tensor):
    mel_np=mel_tensor.detach().numpy()
    mel_np=np.reshape(mel_np,(np.shape(mel_np)[1],np.shape(mel_np)[2]))
    print(f'shape of mel is {np.shape(mel_np)} of type {mel_np.dtype}')
    return mel_np

def generate_mel(audio_path,h):

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

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_wav_path', default='speech_clips/audioclipglory1_1.mp3')
    parser.add_argument('--config_path', default='bigvgan_24khz_100band_config.json')
    parser.add_argument('--input_wav_dir', default='speech_clips')
    parser.add_argument('--output_mel_dir', default=None)
    a = parser.parse_args()
    
    h = get_h(a.config_path)
    print(f'creating mel spectrogram for {a.input_wav_path}')
    mel=generate_mel(a.input_wav_path, h)
    mel=mel_Tensor_to_np(mel)
    plot_mel(mel, h)
    plt.show()

    if a.output_mel_dir:
       # save the mel
       filename=os.path.split(a.input_wav_path)[-1]
       save_path = os.path.join(a.output_dir,filename[:-4])
       print(f'saving MEL spectrogram to: {save_path}')
       np.save(save_path, mel)


if __name__ == "__main__":
  main()
