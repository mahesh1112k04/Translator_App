from translate import Translator

# Function to display available languages
def display_languages():
    print("Available languages:")
    for lang in Translator().languages:
        print(f"- {lang}")

# Function to perform translation
def translate_text(source_language, target_language, text):
    translator = Translator(from_lang=source_language, to_lang=target_language)
    translation = translator.translate(text)
    return translation

# Get available languages and display them
display_languages()

# Get source and target languages from the user
source_lang = input("Enter the source language: ")
target_lang = input("Enter the target language: ")

# Get the text to be translated
text_to_translate = input("Enter the text to translate: ")

# Perform translation
translated_text = translate_text(source_lang, target_lang, text_to_translate)

# Display the translated text
print(f"Translated text: {translated_text}")
