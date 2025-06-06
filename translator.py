import tkinter as tk
import sounddevice as sd
import numpy as np
import asyncio
import edge_tts
from googletrans import Translator
import tempfile
import os
import threading
import datetime
import scipy.io.wavfile
from faster_whisper import WhisperModel
import torch

# load the whisper model
model = WhisperModel("base", compute_type="int8")
translator = Translator()

RECORD_SECONDS = 5
SAMPLERATE = 16000
is_listening = False
LOG_FILE = "transcripts.txt"
#add languages want to translate
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
    ("Ukrainian", "uk", "uk-UA-PolinaNeural"), ("Urdu", "ur", "ur-PK-AsadNeural"), ("Vietnamese", "vi", "vi-VN-HoaiMyNeural")
]

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

#updated for exception handling
def translate_text(text, dest_lang):
    try:
        translated = translator.translate(text, dest=dest_lang).text
        if not translated:
            raise ValueError("Empty translation received")
        return translated
    except Exception as e:
        print(f"Translation error: {e}")
        return "⚠️ Translation failed due to network or API issue."

#updated for exception handling
'''async def speak_text(text, voice_code):
    if not text:
        raise ValueError("No text to speak")
    try:
        communicate = edge_tts.Communicate(text, voice=voice_code)
        await communicate.play()
    except Exception as e:
        print(f"TTS error: {e}")'''
async def speak_text(text, voice_code):
    if not text:
        raise ValueError("No text to speak")
    try:
        communicate = edge_tts.Communicate(text, voice=voice_code)
        output_path = f"translated_output_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
        await communicate.save(output_path)  # Save as MP3
        await communicate.play()             # Play the audio
        print(f"🔊 Saved translated audio to {output_path}")
    except Exception as e:
        print(f"TTS error: {e}")

def save_transcript(original_text, detected_lang, translations):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}]\n")
        f.write(f"Original ({detected_lang}): {original_text}\n")
        for name, translated, _ in translations:
            f.write(f"{name}: {translated}\n")
        f.write("\n----------------------------\n\n")
def listen_loop():
    global is_listening
    is_listening = True
    status_label.config(text="🎤 Listening... (say 'stop' to exit)")

    while is_listening:
        audio = record_audio()
        path = save_audio(audio)

        try:
            text, detected_lang = transcribe_and_detect(path)
            if not text or not detected_lang:
                transcription_label.config(text="Error: No transcription or language detection.")
                continue  # Skip the iteration if transcription failed

            transcription_label.config(text=f"👂 You said ({detected_lang}): {text}")

            if "stop" in text.lower():
                is_listening = False
                status_label.config(text="🛑 Stopped.")
                break

            user_choice = selected_lang.get()
            chosen = next((item for item in MULTI_LANGS if item[0] == user_choice), None)

            if chosen:
                name, lang_code, voice = chosen
                translated = translate_text(text, lang_code)
                if translated:
                    translation_label.config(text=f"🗣️ {name}: {translated}")
                    # Call speak_text here to play the translated speech
                    asyncio.run(speak_text(translated, voice))
                    save_transcript(text, detected_lang, [(name, translated, voice)])
                else:
                    translation_label.config(text="⚠️ Translation failed.")
            else:
                translation_label.config(text="⚠️ Invalid Language Selected!")

        except Exception as e:
            transcription_label.config(text=f"Error: {str(e)}")
        finally:
            os.remove(path)
def start_listening():
    threading.Thread(target=listen_loop, daemon=True).start()
def stop_listening():
    global is_listening
    is_listening = False
    status_label.config(text="🛑 Stopped by user.")

# tkinter GUI setup for app window
window = tk.Tk()
window.title("Speech-to-Speech Translation APP")
window.geometry("520x480")
window.configure(bg="#87CEEB")

tk.Label(window, text="🎧 Speech-to-Speech Translation", font=("Arial", 13, "bold")).pack(pady=5)

# frame to hold listbox and scrollbar
frame_lang = tk.Frame(window)
frame_lang.pack(pady=5, fill=tk.X, padx=20)

scrollbar = tk.Scrollbar(frame_lang)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

lang_listbox = tk.Listbox(frame_lang, yscrollcommand=scrollbar.set, height=8, font=("Arial", 11), exportselection=False)
for name, code, voice in MULTI_LANGS:
    lang_listbox.insert(tk.END, name)
lang_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

scrollbar.config(command=lang_listbox.yview)

# Select default language "English (UK)"
default_index = next((i for i, (name, _, _) in enumerate(MULTI_LANGS) if name == "English (UK)"), 0)
lang_listbox.selection_set(default_index)
lang_listbox.activate(default_index)
lang_listbox.see(default_index)

tk.Button(window, text="Start Listening", command=start_listening, bg="lightgreen", font=("Arial", 12)).pack(pady=10)
tk.Button(window, text="Stop Listening", command=stop_listening, bg="salmon", font=("Arial", 12)).pack(pady=5)

status_label = tk.Label(window, text="Click to start", font=("Arial", 12))
status_label.pack(pady=5)

transcription_label = tk.Label(window, text="👂 Transcription will appear here", wraplength=500, font=("Arial", 11))
transcription_label.pack(pady=10)

translation_label = tk.Label(window, text="🗣️ Translations will appear here", wraplength=500, font=("Arial", 11), justify="left")
translation_label.pack(pady=10)

tk.Button(window, text="Quit", command=window.destroy, bg="gray", font=("Arial", 12)).pack(pady=20)

window.mainloop()
