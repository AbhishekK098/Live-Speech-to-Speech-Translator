from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import tempfile
import os
from speech_logic import transcribe_and_detect, translate_text, text_to_speech

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route("/")
def home():
    return "Backend is running"

@app.route("/translate", methods=["POST"])
def translate():
    try:
        print("[DEBUG] Received request")
        
        # Get audio file
        if 'audio' not in request.files:
            raise ValueError("No audio file provided")
        
        audio = request.files['audio']
        target_language = request.form.get('target_language', 'en')
        print(f"[DEBUG] Target language: {target_language}")

        # Save temp file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        audio.save(temp_file.name)
        print(f"[DEBUG] Audio saved to: {temp_file.name}")

        # Transcribe and detect language
        text, detected_lang = transcribe_and_detect(temp_file.name)
        print(f"[DEBUG] Transcribed: {text}, Detected language: {detected_lang}")
        
        # Translate to target language
        translated = translate_text(text, detected_lang, target_language)
        print(f"[DEBUG] Translated: {translated}")
        
        # Generate speech
        audio_file = text_to_speech(translated, target_language)
        print(f"[DEBUG] Audio file: {audio_file}")
        
        # Cleanup temp file
        os.remove(temp_file.name)

        return jsonify({
            "text": text,
            "detected_language": detected_lang,
            "translated": translated,
            "audio_url": f"http://{request.host}/audio/{audio_file}"
        })

    except Exception as e:
        print(f"[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/audio/<filename>")
def get_audio(filename):
    return send_from_directory(".", filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)


























#from flask import Flask, request, jsonify, send_from_directory
#import os
#import tempfile
#from speech_logic import transcribe_audio, translate_text, text_to_speech
#from speech_logic import transcribe_and_detect, translate_text, text_to_speech

#app = Flask(__name__)

#  Test route
#@app.route("/")
#def home():
#    return "Backend is running"

#  Main API
#@app.route("/translate", methods=["POST"])
#def translate():
#    try:
#        audio = request.files['audio']

        # Save temp file
#        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
#        audio.save(temp_file.name)

        # ✅ Use YOUR function#
#        text, detected_lang = transcribe_and_detect(temp_file.name)

        # ✅ Pass detected language
#        translated = translate_text(text, detected_lang)

        # ✅ Generate speech
#        audio_file = text_to_speech(translated)

#        return jsonify({
#            "text": text,
#            "detected_language": detected_lang,
#            "translated": translated,
#            "audio_url": f"http://{request.host}/audio/{audio_file}"
#        })

#    except Exception as e:
#        return jsonify({"error": str(e)}), 500





#@app.route("/translate", methods=["POST"])
#def translate():
#    try:
#        # Get audio from request
#        audio = request.files['audio']

        # Save temp file
 #       temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
 #       audio.save(temp_file.name)

        #  USE YOUR EXISTING LOGIC
 #       text = transcribe_audio(temp_file.name)
 #       translated = translate_text(text)
  #      audio_file = text_to_speech(translated)

  #      return jsonify({
  #          "text": text,
   #         "translated": translated,
   #         "audio_url": f"http://{request.host}/audio/{audio_file}"
   #     })

    #except Exception as e:
     #   return jsonify({"error": str(e)}), 500


#  Serve audio file
@app.route("/audio/<filename>")
def get_audio(filename):
    return send_from_directory(".", filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

