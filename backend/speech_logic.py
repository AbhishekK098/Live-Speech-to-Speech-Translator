import sounddevice as sd
import numpy as np
import asyncio
import edge_tts
from googletrans import Translator
import tempfile
import os
import datetime
import scipy.io.wavfile
from faster_whisper import WhisperModel
import torch
import asyncio

# load the whisper model
model = WhisperModel("base", compute_type="int8")
translator = Translator()
RECORD_SECONDS = 5
SAMPLERATE = 16000
LOG_FILE = "transcripts.txt"
MULTI_LANGS = [
    ("Arabic", "ar", "ar-SA-ZariyahNeural"), ("Bengali", "bn", "bn-IN-TanishaaNeural"), ("Chinese (Simplified)", "zh-CN", "zh-CN-XiaoxiaoNeural"),("Chinese (Traditional)", "zh-TW", "zh-TW-HsiaoChenNeural"),
    ("Czech", "cs", "cs-CZ-VlastaNeural"), ("Danish", "da", "da-DK-ChristelNeural"), ("Dutch", "nl", "nl-NL-ColetteNeural"), ("English (US)", "en", "en-US-JennyNeural"),
    ("English (UK)", "en", "en-GB-MaisieNeural"), ("Estonian", "et", "et-EE-AnuNeural"), ("Finnish", "fi", "fi-FI-SelmaNeural"), ("French", "fr", "fr-FR-DeniseNeural"),
    ("German", "de", "de-DE-KatjaNeural"), ("Greek", "el", "el-GR-AthinaNeural"),("Gujarati", "gu", "gu-IN-DhwaniNeural"), ("Hebrew", "he", "he-IL-HilaNeural"),("Hindi", "hi", "hi-IN-MadhurNeural"), ("Hungarian", "hu", "hu-HU-NoemiNeural"),
    ("Indonesian", "id", "id-ID-GadisNeural"), ("Italian", "it", "it-IT-ElsaNeural"),("Japanese", "ja", "ja-JP-NanamiNeural"), ("Kannada", "kn", "kn-IN-SapnaNeural"),
    ("Korean", "ko", "ko-KR-SunHiNeural"), ("Latvian", "lv", "lv-LV-EveritaNeural"), ("Lithuanian", "lt", "lt-LT-OnaNeural"), ("Malay", "ms", "ms-MY-YasminNeural"),
    ("Malayalam", "ml", "ml-IN-SobhanaNeural"), ("Marathi", "mr", "mr-IN-AarohiNeural"), ("Norwegian", "no", "nb-NO-PernilleNeural"),("Polish", "pl", "pl-PL-ZofiaNeural"), ("Portuguese (Portugal)", "pt", "pt-PT-RaquelNeural"),
    ("Portuguese (Brazil)", "pt", "pt-BR-FranciscaNeural"), ("Punjabi", "pa", "pa-IN-GagandeepNeural"), ("Romanian", "ro", "ro-RO-AlinaNeural"), ("Russian", "ru", "ru-RU-DariyaNeural"),
    ("Slovak", "sk", "sk-SK-ViktoriaNeural"), ("Slovenian", "sl", "sl-SI-PetraNeural"), ("Spanish (Spain)", "es", "es-ES-ElviraNeural"), ("Spanish (Mexico)", "es", "es-MX-DaliaNeural"),
    ("Swedish", "sv", "sv-SE-SofieNeural"), ("Tamil", "ta", "ta-IN-PallaviNeural"), ("Telugu", "te", "te-IN-ShrutiNeural"), ("Thai", "th", "th-TH-PremNeural"), ("Turkish", "tr", "tr-TR-EmelNeural"),
    ("Ukrainian", "uk", "uk-UA-PolinaNeural"), ("Urdu", "ur", "ur-PK-AsadNeural"), ("Vietnamese", "vi", "vi-VN-HoaiMyNeural")]

def record_audio():
    audio = sd.rec(int(RECORD_SECONDS * SAMPLERATE), samplerate=SAMPLERATE, channels=1, dtype='int16')
    sd.wait()
    return audio

def save_audio(audio):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    scipy.io.wavfile.write(tmp.name, SAMPLERATE, audio)
    return tmp.name

def transcribe_and_detect(path):
    segments, info = model.transcribe(path, beam_size=5)
    text = " ".join([segment.text for segment in segments])
    detected_lang = info.language
    if not text or not detected_lang:
        raise ValueError("Transcription or language detection failed")
    return text, detected_lang

def translate_text(text, source_lang, target_lang="en"):
    try:
        # Map language names to codes if needed
        lang_map = {
            "Spanish": "es", "English (UK)": "en", "English (US)": "en",
            "French": "fr", "German": "de", "Italian": "it", "Portuguese": "pt",
            "Russian": "ru", "Chinese": "zh-CN", "Japanese": "ja", "Korean": "ko",
            "Arabic": "ar", "Hindi": "hi"
        }
        target_code = lang_map.get(target_lang, target_lang)
        
        translated = translator.translate(text, src=source_lang, dest=target_code).text
        if not translated:
            raise ValueError("Empty translation received")
        return translated
    except Exception as e:
        print(f"Translation error: {e}")
        return "Translation failed due to network or API issue."


def text_to_speech(text, target_lang="en", voice_code=None):
    if not text:
        raise ValueError("No text to speak")

    # Map languages to voice codes
    voice_map = {
        "Spanish": "es-ES-ElviraNeural",
        "English (UK)": "en-GB-MaisieNeural",
        "English (US)": "en-US-JennyNeural",
        "French": "fr-FR-DeniseNeural",
        "German": "de-DE-KatjaNeural",
        "Italian": "it-IT-ElsaNeural",
        "Portuguese": "pt-BR-FranciscaNeural",
        "Russian": "ru-RU-DariyaNeural",
        "Chinese": "zh-CN-XiaoxiaoNeural",
        "Japanese": "ja-JP-NanamiNeural",
        "Korean": "ko-KR-SunHiNeural",
        "Arabic": "ar-SA-ZariyahNeural",
        "Hindi": "hi-IN-MadhurNeural",
    }
    
    voice = voice_code or voice_map.get(target_lang, "en-US-JennyNeural")

    output_path = f"translated_output_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"

    async def generate():
        communicate = edge_tts.Communicate(text, voice=voice)
        await communicate.save(output_path)

    asyncio.run(generate())

    return output_path        

#async def speak_text(text, voice_code):
#    if not text:
#        raise ValueError("No text to speak")
#    try:
#        communicate = edge_tts.Communicate(text, voice=voice_code)
#        output_path = f"translated_output_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
#        await communicate.save(output_path)
#        await communicate.play()
#        print(f"🔊 Saved translated audio to {output_path}")
#    except Exception as e:
#        print(f"TTS error: {e}")

def save_transcript(original_text, detected_lang, translations):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}]\n")
        f.write(f"Original ({detected_lang}): {original_text}\n")
        for name, translated, _ in translations:
            f.write(f"{name}: {translated}\n")
        f.write("\n----------------------------\n\n")
