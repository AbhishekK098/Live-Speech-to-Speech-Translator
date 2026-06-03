export const LANGUAGES = [
  { name: "English (UK)", code: "en" },
  { name: "English (US)", code: "en" },
  { name: "Spanish", code: "es" },
  { name: "French", code: "fr" },
  { name: "German", code: "de" },
  { name: "Italian", code: "it" },
  { name: "Portuguese", code: "pt" },
  { name: "Russian", code: "ru" },
  { name: "Chinese", code: "zh" },
  { name: "Japanese", code: "ja" },
  { name: "Korean", code: "ko" },
  { name: "Arabic", code: "ar" },
  { name: "Hindi", code: "hi" },
];

export const getLanguageCode = (languageName: string) => {
  const lang = LANGUAGES.find((l) => l.name === languageName);
  return lang?.code || "en";
};

export const getLanguageName = (code: string) => {
  const lang = LANGUAGES.find((l) => l.code === code);
  return lang?.name || code;
};
