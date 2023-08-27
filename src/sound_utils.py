import os
import speech_recognition as sr

# from elevenlabs import set_api_key, generate, play
import pvporcupine
import pyaudio
import struct
import platform
import numpy as np
import simpleaudio as sa
from dotenv import load_dotenv
import os
from google.cloud import texttospeech_v1
from pydub import AudioSegment
from pydub.playback import play
import io


# LOAD API KEYS
load_dotenv()
# set_api_key(os.getenv("ELEVENLABS_API_KEY"))

if platform.system().lower() == "windows":  # to develop on windows
    keyword_paths = ["data/wakeword/Cyrano_fr_windows_v2_2_0.ppn"]
    model_path = "data/wakeword/porcupine_params_fr.pv"
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\JeanLELONG\inno\cyrano\google_tts_api_id.json"

elif platform.system().lower() == "linux":
    # Raspberry Pi
    keyword_paths = ["/home/avo-k/code/cyrano/data/wakeword/cyrano_fr_raspberry-pi_v2_2_0.ppn"]
    model_path = "/home/avo-k/code/cyrano/data/wakeword/porcupine_params_fr.pv"
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"/home/avo-k/code/cyrano/google_tts_api_id.json"

    # # Docker
    # keyword_paths = ["/code/cyrano_fr_linux_v2_2_0.ppn"]
    # model_path = "/code/porcupine_params_fr.pv"
else:
    raise Exception("Unsupported platform")


def listen_for_wake_word():
    """
    Creates an input audio stream, instantiates an instance of Porcupine object, and monitors the audio stream for
    occurrences of the wake word(s). It prints the time of detection for each occurrence and the wake word.
    """
    # keywords = list()
    # for x in self._keyword_paths:
    #     keywords.append(os.path.basename(x).replace('.ppn', '').split('_')[0])

    porcupine = None
    pa = None
    audio_stream = None
    try:
        porcupine = pvporcupine.create(
            access_key=os.getenv("PORCUPINE_KEY"),
            keyword_paths=keyword_paths,
            model_path=model_path,
        )

        pa = pyaudio.PyAudio()

        audio_stream = pa.open(
            rate=porcupine.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=porcupine.frame_length,
            input_device_index=None,
        )

        while True:
            pcm = audio_stream.read(porcupine.frame_length)
            pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)
            result = porcupine.process(pcm)
            if result >= 0:
                return True

    except KeyboardInterrupt:
        print("Stopping because KeyboardInterrupt")
        return False

    finally:
        # print("Stop")
        if porcupine is not None:
            porcupine.delete()
        if audio_stream is not None:
            audio_stream.close()
        if pa is not None:
            pa.terminate()


# def eleven_speak(text: str) -> None:
#     audio = generate(text=text, voice="Rachel", model="eleven_multilingual_v1")
#     play(audio)


def make_beep(up: bool = True):
    """play beeps on any os"""
    # calculate note frequencies
    A_freq = 440
    Csh_freq = A_freq * 2 ** (4 / 12)
    E_freq = A_freq * 2 ** (7 / 12)

    # get timesteps for each sample, T is note duration in seconds
    sample_rate = 44100
    T = 0.15
    t = np.linspace(0, T, int(T * sample_rate), False)

    # generate sine wave notes
    A_note = np.sin(A_freq * t * 2 * np.pi)
    E_note = np.sin(E_freq * t * 2 * np.pi)

    # concatenate notes
    audio = np.hstack((E_note)) if up else np.hstack((A_note))
    # normalize to 16-bit range
    audio *= 32767 / np.max(np.abs(audio))
    # convert to 16-bit data
    audio = audio.astype(np.int16)

    # start playback
    play_obj = sa.play_buffer(audio, 1, 2, sample_rate)

    # wait for playback to finish before exiting
    play_obj.wait_done()


def listen():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print(end="j'écoute ... ")
        make_beep()
        audio = r.listen(source)
        print("j'ai bien entendu.")
        make_beep(up=False)

    return r.recognize_whisper_api(
        audio_data=audio,
        model="whisper-1",
        api_key=os.getenv("OPENAI_API_KEY"),
    )


def google_tts(text):
    # Create a client
    client = texttospeech_v1.TextToSpeechClient()

    # Initialize request argument(s)
    input = texttospeech_v1.SynthesisInput()
    input.text = text

    voice = texttospeech_v1.VoiceSelectionParams()
    voice.language_code = "FR-fr"
    voice.name = "fr-FR-Wavenet-D"

    audio_config = texttospeech_v1.AudioConfig()
    audio_config.audio_encoding = texttospeech_v1.AudioEncoding.LINEAR16

    request = texttospeech_v1.SynthesizeSpeechRequest(
        input=input,
        voice=voice,
        audio_config=audio_config,
    )

    # api call
    response = client.synthesize_speech(request=request)
    audio_content = response.audio_content

    # Convert audio_content to an AudioSegment
    audio_segment = AudioSegment.from_file(io.BytesIO(audio_content), format="wav")
    play(audio_segment)
