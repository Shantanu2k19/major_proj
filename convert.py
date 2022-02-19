from os import path
from pydub import AudioSegment

import os
# os.chdir(os.path("C:\Users\Tarun Kumar Gola\Desktop\Extra\Project Work\Gramvaani_Train_100\Audio"))
for s in os.listdir():
    sound = AudioSegment.from_mp3(s)
    dst = "../"+s.split('.')[0]
    sound.export(dst, format="wav")

# files
#src = "transcript.mp3"
#dst = "test.wav"

# convert wav to mp3
#sound = AudioSegment.from_mp3(src)
#sound.export(dst, format="wav")
