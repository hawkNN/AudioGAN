import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def animate_mel_spectrogram(audio_file_path, output_video_path=None, h=None, title='Mel spectrogram'):
    # Get parameter values used for mel spectrogram, input through variable 'h'
    if h:
        sr = h.sampling_rate
        fmax = h.fmax
        hop = h.hop_size
    else:
        # Defaults based on your configuration
        sr = 24000
        fmax = 12000
        hop = 256

    y, sr = librosa.load(audio_file_path, sr=sr)

    # Compute mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=y, fmax=fmax, hop_length=hop, sr=sr)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Create figure and axis for animation
    fig, ax = plt.subplots()
    im = ax.imshow(mel_spectrogram_db, aspect='auto', origin='lower', cmap='viridis')

    cb = plt.colorbar(im, format="%+2.0f dB")
    ax.set_title(title)
    ax.set_xlabel('Time')
    ax.set_ylabel('Frequency')
    plt.tight_layout()

    def update(frame):
        im = ax.imshow(mel_spectrogram_db, aspect='auto', origin='lower', cmap='viridis')
        im.set_data(mel_spectrogram_db[:, :frame])
        return im,

    total_frames = mel_spectrogram_db.shape[1]
    output_fps = int(sr / hop)

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

# Usage
audio_file_path = r'C:\Users\jshha\AudioGAN\speech_clips\audioclipglory1_1.mp3'
output_video_path = r'c:/Users/jshha/GitOutbox/output_animation.mp4'  # Optional, set to None if not saving
animate_mel_spectrogram(audio_file_path, output_video_path)
