
import os
import numpy as np

test_path = '../bigvgan'
print(os.listdir(test_path))
print(os.path.exists(test_path))

# from pydub import AudioSegment
# from pydub.playback import play
# import threading

# sound = AudioSegment.from_mp3('speech_clips\audioclipglory1_1.mp3')
# t = threading.Thread(target=play, args=(sound,))
# t.start()

# print("I would like this line to be executed simultaneously with the audio playing")


print(f'937/10 is {np.floor_divide(937,10)}')
