from googletrans import Translator

def translate_word(word):
    translator = Translator()
    languages = {
        "Russian": "ru",
        "German": "de",
        "French": "fr",
        "Spanish": "es",
        "Hebrew": "he",
        "Arabic": "ar",
        "Japanese": "ja"
    }

    translations = {}
    for language, code in languages.items():
        translated = translator.translate(word, dest=code)
        translations[language] = translated.text

    return translations

# Get user input
word = input("Enter an English word to translate: ")
translations = translate_word(word)

# Display the translations
print("\nTranslations:")
for language, translation in translations.items():
    print(f"{language}: {translation}")