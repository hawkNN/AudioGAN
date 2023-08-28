import numpy as np
import librosa
import matplotlib.pyplot as plt
from IPython.display import Audio, display
from matplotlib.animation import FuncAnimation
from moviepy.editor import VideoFileClip, AudioFileClip


def animate_mel_spectrogram(audio_file_path, output_video_path=None):
    # Load audio wave file
    y, sr = librosa.load(audio_file_path, sr=None)
    print(f'sampling rate is: {sr}')

    # Compute mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Create figure and axis for animation
    fig, ax = plt.subplots()
    im = ax.imshow(mel_spectrogram_db, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(im, format="%+2.0f dB")
    ax.set_title("Mel Spectrogram")

    # Define a function to update the animation
    def update(frame):
        im.set_data(mel_spectrogram_db[:, :frame])
        return im,

    # Calculate total frames for animation
    total_frames = mel_spectrogram_db.shape[1]

    # Create the animation
    ani = FuncAnimation(fig, update, frames=total_frames, blit=True, repeat=False)

    # # Display the animation synchronized with audio
    # audio_display = display(Audio(y, rate=sr, autoplay=True), ani)

    if output_video_path:
        output_video_path = output_video_path.lower()
        if output_video_path.endswith(('.mp4', '.avi', '.gif')):
            ani.save(output_video_path, writer='ffmpeg', fps=20)
            print(f"Saved animation video to: {output_video_path}")
        else:
            print("Output video format not supported. Please use '.mp4', '.avi', or '.gif'.")

    plt.show()

def combine_audio_and_video(audio_file_path, 
                            video_file_path, 
                            output_video_path,
                            sampling_rate=22500):
    audio_clip = AudioFileClip(audio_file_path, fps=sampling_rate)
    video_clip = VideoFileClip(video_file_path)
    
    # Set audio of the video to the provided audio clip
    video_clip = video_clip.set_audio(audio_clip)
    
    # Write the combined video to the output path
    video_clip.write_videofile(output_video_path, codec='libx264', audio_codec='aac')


# Usage
audio_file_path = 'speech_clips/audioclipglory1_1.mp3'
video_file_path = '../gitOutbox/temp.mp4' # Optional, set to None if not saving
av_file_path = '../gitOutbox/audioclipglory1_1.mp4'
# play_audio(audio_file_path)
animate_mel_spectrogram(audio_file_path, video_file_path)
combine_audio_and_video(audio_file_path,video_file_path,av_file_path)

