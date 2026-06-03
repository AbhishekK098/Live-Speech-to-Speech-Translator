import React, { useState } from "react";
import { View, StyleSheet, Text } from "react-native";
import { translateSpeech, stopRecording, startRecording } from "../services/api";
import { getLanguageName } from "../utils/languages";

import { CustomButton } from "../components/Button";
import { LanguagePicker } from "../components/LanguagePicker";
import { TextBox } from "../components/TextBox";

const HomeScreen = () => {
  const [targetLanguage, setTargetLanguage] = useState("Spanish");
  const [sourceLanguage, setSourceLanguage] = useState("Detecting...");
  const [text, setText] = useState("");
  const [translation, setTranslation] = useState("");
  const [loading, setLoading] = useState(false);

  const startListening = async () => {
    setLoading(true);
    setText("Listening for 5 seconds...");
    setTranslation("");
    setSourceLanguage("Detecting...");

  const result = await translateSpeech(targetLanguage);

    await startRecording();

    setTimeout(async () => {
      const audioBlob = await stopRecording();

      const result = await translateSpeech(audioBlob, targetLanguage);

    }, 5000);

    //const result = await translateSpeech(targetLanguage);

    if (result) {
      setText(result.transcription);
      setTranslation(result.translation);
      setSourceLanguage(getLanguageName(result.detectedLanguage));
    } else {
      setText("Error: Could not process audio. Check microphone access.");
      setSourceLanguage("Error");
    }

    setLoading(false);
  };

  const stopListening = async () => {
    try {
      setLoading(false);
      setText("Stopped");
    } catch (err) {
      console.log("Stop error:", err);
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Speech Translator</Text>

      <View style={styles.languageContainer}>
        <View style={styles.languageBox}>
          <Text style={styles.label}>From</Text>
          <Text style={styles.languageText}>{sourceLanguage}</Text>
        </View>
        <Text style={styles.arrowText}>→</Text>
        <View style={styles.languageBox}>
          <Text style={styles.label}>To</Text>
          <LanguagePicker selected={targetLanguage} onSelect={setTargetLanguage} />
        </View>
      </View>

      <CustomButton 
        title={loading ? "Listening..." : "Start Listening"} 
        onPress={startListening}
        disabled={loading}
      />
      <CustomButton 
        title="Stop Listening" 
        onPress={stopListening}
        disabled={!loading}
      />

      <TextBox label="Transcription" value={text} />
      <TextBox label="Translation" value={translation} />
    </View>
  );
};

export default HomeScreen;

const styles = StyleSheet.create({
  container: { flex: 1, padding: 20, justifyContent: "center" },
  title: { fontSize: 22, fontWeight: "bold", marginBottom: 20 },
  languageContainer: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: 20,
    justifyContent: "space-between",
  },
  languageBox: {
    flex: 1,
    marginHorizontal: 5,
  },
  label: { fontSize: 12, color: "#666", marginBottom: 5 },
  languageText: {
    padding: 10,
    backgroundColor: "#f0f0f0",
    borderRadius: 4,
    fontSize: 14,
    fontWeight: "500",
  },
  arrowText: { fontSize: 20, marginHorizontal: 10, fontWeight: "bold" },
});