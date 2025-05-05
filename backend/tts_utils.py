from gtts import gTTS
import os
import uuid

def dialogue_to_speech(dialogue, output_dir):
    tts = gTTS(dialogue)
    audio_filename = f"dialogue_{uuid.uuid4().hex}.mp3"
    audio_path = os.path.join(output_dir, audio_filename)
    tts.save(audio_path)
    return audio_path 