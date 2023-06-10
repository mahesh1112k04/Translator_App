from typing import List, Optional
from sklearn.base import BaseEstimator, TransformerMixin
import fasttext
from transformers import MarianTokenizer, MarianMTModel
import os

class EnglishTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self,
                 fasttext_model_path: str="/tmp/lid.176.bin",
                 allowed_langs: Optional[List[str]]=None,
                 target_lang: str="en"):
        
        # If the language model doesn't exist download it
        if not os.path.isfile(fasttext_model_path):
            url = 'https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin'
            r = requests.get(url, allow_redirects=True)
            open('/tmp/lid.176.bin', 'wb').write(r.content)
        
        self.fasttext_model_path = fasttext_model_path
        self.lang_model = fasttext.load_model(fasttext_model_path)
        self.allowed_langs = allowed_langs
        self.target_lang = target_lang
        self.romance_langs = {"it", "es", "fr", "pt", "oc", "ca", "rm", "wa",
                              "lld", "fur", "lij", "lmo", "gl", "lad", "an", "mwl"}
        
        if allowed_langs is None:
            self.allowed_langs = self.romance_langs.union({self.target_lang, "tr", "ar", "de", "ru"})
        else:
            self.allowed_langs = allowed_langs
        
    def get_language(self, texts: List[str]) -> List[str]:
        
        # Predict the language code for each text in texts
        langs, _ = self.lang_model.predict([x.replace("\n", " ") for x in texts])
        
        # Extract the two character language code from the predictions.
        return [x[0].split("__")[-1] for x in langs]
        
    
    def transform(self, texts: str) -> List[str]:
        
        # Get the language codes for each text in texts
        langs = self.get_language(texts)
        
        translations = []
        for text, lang in zip(texts, langs):
            # If the language is our target just add it as is without doing any prediciton.
            if lang == self.target_lang:
                translations.append(text)
            else:
                # Use the romance model if it is a romance language to avoid
                # downloading a model for every language
                if lang in self.romance_langs and self.target_lang == "en":
                    lang = "ROMANCE"
                translation_model_name = f'Helsinki-NLP/opus-mt-{lang}-{self.target_lang}'

                # Download the model and tokenizer
                model = MarianMTModel.from_pretrained(translation_model_name)
                tokenizer = MarianTokenizer.from_pretrained(translation_model_name)
                
                # Translate the text
                inputs = tokenizer([text], return_tensors="pt", padding=True)
                gen = model.generate(**inputs)
                translations.append(tokenizer.batch_decode(gen, skip_special_tokens=True)[0])
        
        return translations
