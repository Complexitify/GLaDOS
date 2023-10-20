from lib.speech import Speech

print("Initializing GLaDOS-SPEECH...")
speech = Speech("hifigan", "test_2")

print("GLaDOS Online.")
speech.say("Welcome to the aperture science center.")