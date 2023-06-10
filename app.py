from typing import List, Optional, Set
from sklearn.base import BaseEstimator, TransformerMixin
import fasttext
from transformers import MarianTokenizer, MarianMTModel
import os
import requests

class LanguageTransformerFast(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        fasttext_model_path: str = "/tmp/lid.176.bin",
        allowed_langs: Optional[Set[str]] = None,
        target_lang: str = "en",
    ):

        self.fasttext_model_path = fasttext_model_path
        self.allowed_langs = allowed_langs
        self.target_lang = target_lang
        self.romance_langs = {
            "it",
            "es",
            "fr",
            "pt",
            "oc",
            "ca",
            "rm",
            "wa",
            "lld",
            "fur",
            "lij",
            "lmo",
            "gl",
            "lad",
            "an",
            "mwl",
        }

        if allowed_langs is None:
            self.allowed_langs = self.romance_langs.union(
                {self.target_lang, "tr", "ar", "de"}
            )
        else:
            self.allowed_langs = allowed_langs

    def get_language(self, texts: List[str]) -> List[str]:
        # If the model doesn't exist download it
        if not os.path.isfile(self.fasttext_model_path):
            url = (
                "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
            )
            r = requests.get(url, allow_redirects=True)
            open("/tmp/lid.176.bin", "wb").write(r.content)

        lang_model = fasttext.load_model(self.fasttext_model_path)

        # Predict the language code for each text in texts
        langs, _ = lang_model.predict([x.replace("\n", " ") for x in texts])

        # Extract the two character language code from the predictions.
        return [x[0].split("__")[-1] for x in langs]

    def fit(self, X, y):
        return self

    def transform(self, texts: str) -> List[str]:

        # Get the language codes for each text in texts
        langs = self.get_language(texts)

        # Zip the texts, languages, and their indecies
        # sort on the language so that all languages appear together
        text_lang_pairs = sorted(
            zip(texts, langs, range(len(langs))), key=lambda x: x[1]
        )
        model = None

        translations = []
        prev_lang = text_lang_pairs[0]
        for text, lang, idx in text_lang_pairs:
            if lang == self.target_lang or lang not in self.allowed_langs:
                translations.append((idx, text))
            else:
                # Use the romance model if it is a romance language to avoid
                # downloading a model for every language
                if lang in self.romance_langs and self.target_lang == "en":
                    lang = "ROMANCE"

                if model is None or lang != prev_lang:
                    translation_model_name = (
                        f"Helsinki-NLP/opus-mt-{lang}-{self.target_lang}"
                    )

                    # Download the model and tokenizer
                    model = MarianMTModel.from_pretrained(translation_model_name)
                    tokenizer = MarianTokenizer.from_pretrained(translation_model_name)

                # Tokenize the text
                batch = tokenizer([text], return_tensors="pt", padding=True)

                # Make sure that the tokenized text does not exceed the maximum
                # allowed size of 512
                batch["input_ids"] = batch["input_ids"][:, :512]
                batch["attention_mask"] = batch["attention_mask"][:, :512]

                gen = model.generate(**batch)
                translations.append(
                    (idx, tokenizer.batch_decode(gen, skip_special_tokens=True)[0])
                )
                prev_lang = lang

        # Reorganize the translations to match the original ordering
        return [x[1] for x in sorted(translations, key=lambda x: x[0])]
