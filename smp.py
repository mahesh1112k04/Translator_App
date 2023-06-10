import os
from google.cloud import translate

# Set up the translation client
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "path/to/your/credentials.json"
translate_client = translate.TranslationServiceClient()

def translate_text(text, target_language):
    # Define the translation request
    parent = translate_client.location_path("your-project-id", "global")
    response = translate_client.translate_text(
        request={
            "parent": parent,
            "contents": [text],
            "mime_type": "text/plain",
            "source_language_code": "en",
            "target_language_code": target_language,
        }
    )

    # Extract and return the translated text
    translation = response.translations[0].translated_text
    return translation

# Example usage
input_text = "Hello, how are you?"
target_language = "fr"
translated_text = translate_text(input_text, target_language)

print(f"Translation: {translated_text}")
