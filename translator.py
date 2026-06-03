
import tkinter as tk
import threading
import os
import asyncio
from flask import Flask, request
from speech_logic import (
    record_audio, save_audio, transcribe_and_detect, translate_text, speak_text, save_transcript, MULTI_LANGS
)

app = Flask(__name__)
is_listening = False

# --- Commented out duplicate logic functions (see speech_logic.py for active versions) ---
# def record_audio():
#     audio = sd.rec(int(RECORD_SECONDS * SAMPLERATE), samplerate=SAMPLERATE, channels=1, dtype='int16')
#     sd.wait()
#     return audio
#
# def save_audio(audio):
#     tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
#     scipy.io.wavfile.write(tmp.name, SAMPLERATE, audio)
#     return tmp.name
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
        return " Translation failed due to network or API issue."
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
                continue
            transcription_label.config(text=f" You said ({detected_lang}): {text}")
            if "stop" in text.lower():
                is_listening = False
                status_label.config(text=" Stopped.")
                break
            # Get selected language from Listbox
            selection = lang_listbox.curselection()
            user_choice = lang_listbox.get(selection[0]) if selection else "English (UK)"
            chosen = next((item for item in MULTI_LANGS if item[0] == user_choice), None)
            if chosen:
                name, lang_code, voice = chosen
                translated = translate_text(text, lang_code)
                if translated:
                    translation_label.config(text=f" {name}: {translated}")
                    asyncio.run(speak_text(translated, voice))
                    save_transcript(text, detected_lang, [(name, translated, voice)])
                else:
                    translation_label.config(text=" Translation failed.")
            else:
                translation_label.config(text=" Invalid Language Selected!")
        except Exception as e:
            transcription_label.config(text=f"Error: {str(e)}")
        finally:
            os.remove(path)
@app.route('/translate', methods=['POST'])
def translate():
    data = request.json
    target_lang = data.get("target_language")

    return {
        "transcription": "Hello",
        "translation": "Hola"
    }
    
# UI event handlers
def start_listening():
    threading.Thread(target=listen_loop, daemon=True).start()

def stop_listening():
    global is_listening
    is_listening = False
    status_label.config(text=" Stopped by user.")
tk.Label(window, text=" Speech-to-Speech Translation", font=("Arial", 13, "bold")).pack(pady=5)
tk.Button(window, text="Start Listening", command=start_listening, bg="lightgreen", font=("Arial", 12)).pack(pady=10)
tk.Button(window, text="Stop Listening", command=stop_listening, bg="salmon", font=("Arial", 12)).pack(pady=5)
tk.Button(window, text="Quit", command=window.destroy, bg="gray", font=("Arial", 12)).pack(pady=20)
# tkinter GUI setup for app window
window = tk.Tk()
window.title("Speech-to-Speech Translation APP")
window.geometry("520x480")
window.configure(bg="#87CEEB")
tk.Label(window, text=" Speech-to-Speech Translation", font=("Arial", 13, "bold")).pack(pady=5)
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
transcription_label = tk.Label(window, text=" Transcription will appear here", wraplength=500, font=("Arial", 11))
transcription_label.pack(pady=10)
translation_label = tk.Label(window, text=" Translations will appear here", wraplength=500, font=("Arial", 11), justify="left")
translation_label.pack(pady=10)
tk.Button(window, text="Quit", command=window.destroy, bg="gray", font=("Arial", 12)).pack(pady=20)
window.mainloop()
