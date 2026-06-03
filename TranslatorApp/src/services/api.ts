let mediaRecorder: MediaRecorder | null = null;
let audioChunks: Blob[] = [];
let stream: MediaStream | null = null;

export const startRecording = async (): Promise<void> => {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);
    audioChunks = [];

    mediaRecorder.ondataavailable = (event) => {
      audioChunks.push(event.data);
    };

    mediaRecorder.start();
  } catch (err) {
    console.log("Microphone access denied:", err);
    throw err;
  }
};

export const stopRecording = (): Promise<Blob> => {
  return new Promise((resolve, reject) => {
    if (!mediaRecorder) {
      reject("No recording in progress");
      return;
    }

    mediaRecorder.onstop = () => {
      const audioBlob = new Blob(audioChunks, { type: "audio/wav" });
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
      }
      resolve(audioBlob);
    };

    mediaRecorder.stop();
  });
};





const BASE_URL = "http://192.168.1.9:5000"; // your PC IP

export const translateSpeech = async (
  targetLanguage: string
): Promise<{ transcription: string; translation: string; detectedLanguage: string } | null> => {
  try {
    // Record audio
    await startRecording();

    // Record for 5 seconds
    await new Promise((resolve) => setTimeout(resolve, 5000));

    // Stop recording and get audio blob
    const audioBlob = await stopRecording();

    // Create FormData with audio
    const formData = new FormData();
    formData.append("audio", audioBlob, "audio.wav");
    formData.append("target_language", targetLanguage);

    // Send to backend
    const res = await fetch(`${BASE_URL}/translate`, {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      throw new Error(`HTTP error! status: ${res.status}`);
    }

    const data = await res.json();

    if (data.error) {
      throw new Error(data.error);
    }

    return {
      transcription: data.text || "",
      translation: data.translated || "",
      detectedLanguage: data.detected_language || "unknown",
    };
  } catch (err) {
    console.log("API ERROR:", err);
    return null;
  }
};