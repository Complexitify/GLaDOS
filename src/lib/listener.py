"""
    Listener class that listens for words.

    Requires a microphone.
"""
import time
import calendar
import speech_recognition as sr
from lib.audio_util import play_audio

class Listener:
    def __init__(self) -> None:
        self.recognizer = sr.Recognizer()
        try:
            self.mic = sr.Microphone()
        except OSError:
            print("No input device! Is the microphone connected?")
            exit()
    
    def wait_for_words(self, words):
        while True:
            response = self.listen()

            if response["success"] == True:
                for word in words:
                    if word.lower() in response["transcription"].lower():
                        print("Keyword successfully detected. Transcripted audio:")
                        print(response["transcription"])
                        return True
                
                print("No keyword detected. Transcripted audio:")
                print(response["transcription"])
            else:
                print(response["error"])

    def listen_and_wait(self, threshold_time: float, error_sound: bool=False):
        start_time = calendar.timegm(time.gmtime())

        while (calendar.timegm(time.gmtime()) - start_time) <= threshold_time:
            response = self.listen()

            if response["success"] == True:
                return response
            else:
                print(response["error"])
                if error_sound:
                    play_audio("assets/error_listening.wav")
        
        return {
            "success": False,
            "error": "No response",
            "transcription": None
        }

    def listen(self):
        with self.mic as source:
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source)
        
        response = {
            "success": True,
            "error": None,
            "transcription": None
        }

        # Try recognizing the speech in the recording
        # If a RequestError or UnknownValueError exception is caught,
        #     update the response object accordingly
        try:
            # OpenAI Whisper is the best offline speech recognizer I've ever seen.
            response["transcription"] = self.recognizer.recognize_whisper(audio)
        except sr.RequestError:
            # API was unreachable or unresponsive
            response["success"] = False
            response["error"] = "API unavailable"
        except sr.UnknownValueError:
            # speech was unintelligible
            response["error"] = "Unable to recognize speech"
        
        return response