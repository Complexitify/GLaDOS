import random
from lib.speech import Speech
from lib.listener import Listener
from lib.audio_util import play_audio

# Create core objects
print("Initializing GLaDOS-LISTENER...")
listener = Listener()
print("Initializing GLaDOS-SPEECH...")
speech = Speech("hifigan", "test_2")

speech.say("aperture science is happy to sponsor technicly robotics")