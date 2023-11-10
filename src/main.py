import random
from lib.speech import Speech
from lib.listener import Listener
from lib.audio_util import play_audio
from lib.kasa_utils import *

# Variables
LISTEN_TIME_THRESHOLD = 10 # Seconds GLaDOS should listen for a command before giving up.
KEYWORDS = ["glados", "glad us", "glattos", "gladdos", "glad those", "glad also", "gladys" # Words to wake GLaDOS and prompt listen.
            "galatos", "galaros", "glatos", "glow's", "galados", "glad i was", "Gladdles", "Glad Oz",
            "gladys", "clados", "claros", "glad os", "clark os", "clark oh", "glad das"
            ]
WAKE_DIALOGUES = ["What?", "Yes?", "What is it?"]

SMART_PLUG_IP = "192.168.1.190"
SMART_PLUG = None

COMMAND_KEYWORDS = [
    ["repeat", "echo", "eat after me", "after me"], # Repeat after me
    ["turn on light", "turn on the light", "on the light", "on light"], # Turn on the light
    ["turn off light", "turn off the light", "off the light", "off light"], # Turn off the light
]

COMMAND_LIST = [ # Corresponds to COMMAND_KEYWORDS to find a command.
    "repeat",
    "turn on plug",
    "turn off plug",
]

####################################
# Commands
####################################

def GetNextWord(listen_time: float=LISTEN_TIME_THRESHOLD):
    said = listener.listen_and_wait(listen_time, True)

    if said["success"] == True:
        return said["transcription"]
    else:
        play_audio("assets/error_listening.wav")
    
    return None

def GetCommand(transcription: str):
    print("Got transcription: " + transcription)
    print("Searching for command keywords...")
    transcription = transcription.lower()

    index = 0 # Current index counter for the list pair.
    got_command = False
    for possible_transcriptions in COMMAND_KEYWORDS:
        for keyword in possible_transcriptions:
            if keyword.lower() in transcription:
                command = COMMAND_LIST[index]
                got_command = True
                print("Got command: " + command)
                OnCommand(command)
                return
        
        index += 1 # Update index
    
    if not got_command:
        print("Failed to get command.")


def OnCommand(cmd: str):
    if cmd == "repeat":
        play_audio("assets/okay.wav")

        # Get word to repeat
        repeat = GetNextWord()
        if repeat != None:
            speech.say(repeat) # Say it if it exists.

    elif cmd == "turn on plug":
        if SMART_PLUG != None:
            try:
                asyncio.run(SMART_PLUG.turn_on())
            except:
                print("Kasa Failed.")
            speech.say("Power on.")
        else:
            speech.say("The plug is unusable at the moment.")

    elif cmd == "turn off plug":
        if SMART_PLUG != None:
            try:
                asyncio.run(SMART_PLUG.turn_off())
            except:
                print("Kasa Failed.")
        else:
            speech.say("The plug is unusable at the moment.")

####################################

# Initialize sound
play_audio("assets/powerup_initiated.wav")

# Create core objects
print("Initializing GLaDOS-LISTENER...")
listener = Listener()
print("Initializing GLaDOS-SPEECH...")
speech = Speech("hifigan", "test_2")

# Startup, play the Aperture Science jingle.
play_audio("assets/aperture_theme.wav")

speech.say("Running text to speech synthesis test.")

# Connect to tp-link kasa smart plug
play_audio("assets/plug_connect.wav")

SMART_PLUG = get_plug(SMART_PLUG_IP)

if SMART_PLUG == None:
    play_audio("assets/plug_fail.wav")
else:
    play_audio("assets/plug_success.wav")

# Initialization complete, let user know and start main loop.
print("GLaDOS Online.")
play_audio("assets/powerup_complete.wav")
while True:
    success = listener.wait_for_words(KEYWORDS)

    if success:
        # Say wake dialogue, to let user know when to speak.
        speech.say(WAKE_DIALOGUES[random.randint(0, (len(WAKE_DIALOGUES) - 1))])

        next_word = GetNextWord()

        if next_word != None:
            GetCommand(next_word)