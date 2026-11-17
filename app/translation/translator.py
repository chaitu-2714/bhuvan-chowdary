import requests
import urllib.parse
import logging

logger = logging.getLogger(__name__)

class TranslationProvider:
    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        raise NotImplementedError

class OfflineDictionaryTranslator(TranslationProvider):
    """
    Offline dictionary translator for common sign labels and simple combinations.
    """
    def __init__(self):
        # Mappings from english words to Hindi (hi) and Telugu (te)
        self.dictionary = {
            "hi": {
                "hello": "नमस्ते",
                "thanks": "धन्यवाद",
                "thank you": "धन्यवाद",
                "iloveyou": "मैं आपसे प्यार करता हूँ",
                "i love you": "मैं आपसे प्यार करता हूँ",
                "thumbs_up": "बहुत बढ़िया",
                "thumbs_down": "नापसंद",
                "peace": "शांति",
                "ok": "ठीक है",
                "stop": "रुकें",
                "point_up": "ऊपर इशारा",
                "point up": "ऊपर इशारा",
                "yes": "हाँ",
                "no": "नहीं",
                "please": "कृपया",
                "sorry": "माफ़ कीजिये",
                "help": "मदद",
                "father": "पिता",
                "mother": "माता",
                "water": "पानी",
                "food": "खाना"
            },
            "te": {
                "hello": "నమస్తే",
                "thanks": "ధన్యవాదాలు",
                "thank you": "ధన్యవాదాలు",
                "iloveyou": "నేను నిన్ను ప్రేమిస్తున్నాను",
                "i love you": "నేను నిన్ను ప్రేమిస్తున్నాను",
                "thumbs_up": "చాలా బాగుంది",
                "thumbs_down": "బాగోలేదు",
                "peace": "శాంతి",
                "ok": "సరే",
                "stop": "ఆగండి",
                "point_up": "పైకి చూపించు",
                "point up": "పైకి చూపించు",
                "yes": "అవును",
                "no": "కాదు",
                "please": "దయచేసి",
                "sorry": "క్షమించండి",
                "help": "సహాయం",
                "father": "తండ్రి",
                "mother": "తల్లి",
                "water": "మంచినీళ్లు",
                "food": "ఆహారం"
            }
        }

    def translate(self, text: str, source_lang: str = "en", target_lang: str = "hi") -> str:
        text_lower = text.lower().strip()
        
        # Check direct lookup
        lang_dict = self.dictionary.get(target_lang, {})
        if text_lower in lang_dict:
            return lang_dict[text_lower]
            
        # Try word-by-word fallback for short sentences
        words = text_lower.split()
        translated_words = []
        for word in words:
            translated_words.append(lang_dict.get(word, word))
            
        return " ".join(translated_words)

class GoogleTranslateTranslator(TranslationProvider):
    """
    Online translation provider using the Google Translate API.
    """
    def __init__(self):
        self.url = "https://translate.googleapis.com/translate_a/single"

    def translate(self, text: str, source_lang: str = "en", target_lang: str = "hi") -> str:
        if not text.strip():
            return text
            
        try:
            params = {
                "client": "gtx",
                "sl": source_lang,
                "tl": target_lang,
                "dt": "t",
                "q": text
            }
            # Set a low timeout so it fails quickly if offline, without blocking UI
            response = requests.get(self.url, params=params, timeout=3.0)
            if response.status_code == 200:
                result = response.json()
                # Google Translate returns a list: [[[translatedText, originalText, ...]]]
                translated_text = "".join([part[0] for part in result[0] if part[0]])
                return translated_text
            else:
                logger.warning(f"Google translate request failed with status: {response.status_code}")
                raise ConnectionError("Google translate service unavailable")
        except Exception as e:
            logger.warning(f"Failed to fetch online translation: {e}")
            raise

class TranslationService:
    def __init__(self, provider="online"):
        self.online_provider = GoogleTranslateTranslator()
        self.offline_provider = OfflineDictionaryTranslator()
        self.provider_mode = provider  # "online" or "offline"

    def translate(self, text: str, source_lang: str = "en", target_lang: str = "hi") -> str:
        """
        Translates text using the chosen provider.
        Automatically falls back to offline mode if online translation fails.
        """
        if target_lang == source_lang:
            return text
            
        if self.provider_mode == "online":
            try:
                return self.online_provider.translate(text, source_lang, target_lang)
            except Exception:
                logger.info("Translation falling back to offline dictionary.")
                return self.offline_provider.translate(text, source_lang, target_lang)
        else:
            return self.offline_provider.translate(text, source_lang, target_lang)
