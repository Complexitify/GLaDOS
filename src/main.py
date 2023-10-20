from lib.speech import Speech

print("Initializing GLaDOS-SPEECH...")
speech = Speech("hifigan", "test_2")

print("GLaDOS Online.")
speech.say("Oh its you.")
speech.say("Its been a long time")
speech.say("How have you been")
speech.say("Welcome to the aperture science center.")