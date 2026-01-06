"""
Translation Service - Multi-language support using NLLB-200

Handles translation between supported languages using Facebook's NLLB-200 model.
"""

import logging
from typing import Optional
import os

try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers not available. Install: pip install transformers torch")

logger = logging.getLogger(__name__)

# NLLB-200 language codes mapping
# Full list: https://github.com/facebookresearch/flores/blob/main/flores200/README.md
LANGUAGE_CODES = {
    "en": "eng_Latn",  # English
    "es": "spa_Latn",  # Spanish
    "hi": "hin_Deva",  # Hindi
    "zh": "zho_Hans",  # Chinese (Simplified)
    "ar": "arb_Arab",  # Arabic
    "pt": "por_Latn",  # Portuguese
    "fr": "fra_Latn",  # French
}

# Reverse mapping for display
LANGUAGE_NAMES = {
    "en": "English",
    "es": "Spanish",
    "hi": "Hindi",
    "zh": "Chinese",
    "ar": "Arabic",
    "pt": "Portuguese",
    "fr": "French",
}


class TranslationService:
    """Handles translation between languages using NLLB-200"""
    
    def __init__(self, model_name: str = "facebook/nllb-200-distilled-600M"):
        """
        Initialize translation service
        
        Args:
            model_name: HuggingFace model name for NLLB-200
                       Options:
                       - facebook/nllb-200-distilled-600M (smaller, faster)
                       - facebook/nllb-200-1.3B (larger, better quality)
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = None
        self._initialized = False
        
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers not available - translation will not work")
            return
        
        self._init_model()
    
    def _init_model(self):
        """Load NLLB-200 model (lazy loading)"""
        if self._initialized:
            return
        
        try:
            logger.info(f"Loading translation model: {self.model_name}")
            
            # Determine device (GPU if available, else CPU)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {self.device}")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            self._initialized = True
            logger.info("Translation model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load translation model: {e}")
            logger.warning("Translation will not be available")
            self.tokenizer = None
            self.model = None
    
    def translate(
        self,
        text: str,
        source_lang: str = "en",
        target_lang: str = "en"
    ) -> str:
        """
        Translate text from source language to target language
        
        Args:
            text: Text to translate
            source_lang: Source language code (e.g., 'en', 'es', 'hi')
            target_lang: Target language code (e.g., 'en', 'es', 'hi')
        
        Returns:
            Translated text
        """
        if not self._initialized or not self.model:
            raise RuntimeError("Translation model not loaded")
        
        if source_lang == target_lang:
            return text  # No translation needed
        
        try:
            # Get NLLB language codes
            src_code = LANGUAGE_CODES.get(source_lang)
            tgt_code = LANGUAGE_CODES.get(target_lang)
            
            if not src_code or not tgt_code:
                raise ValueError(
                    f"Unsupported language. Source: {source_lang}, Target: {target_lang}"
                )
            
            # Tokenize input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Translate
            with torch.no_grad():
                translated_tokens = self.model.generate(
                    **inputs,
                    forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(tgt_code),
                    max_length=512,
                    num_beams=3,
                    early_stopping=True
                )
            
            # Decode translation
            translated_text = self.tokenizer.decode(
                translated_tokens[0],
                skip_special_tokens=True
            )
            
            logger.debug(f"Translated: {text[:30]}... -> {translated_text[:30]}...")
            return translated_text.strip()
            
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            raise
    
    def detect_language(self, text: str) -> Optional[str]:
        """
        Detect the language of input text
        
        Note: NLLB-200 doesn't have built-in language detection.
        This is a simple heuristic - in production, use a dedicated language detection library.
        
        Args:
            text: Text to detect language for
        
        Returns:
            Detected language code or None
        """
        # Simple heuristic: check for common patterns
        # For a real implementation, use langdetect or similar
        text_lower = text.lower()
        
        # Check for non-ASCII characters that suggest non-English
        if any(ord(c) > 127 for c in text):
            # Could be Hindi, Chinese, Arabic, etc.
            # For now, return None (let Whisper detect)
            return None
        
        # Default to English if no obvious indicators
        return "en"
    
    def is_available(self) -> bool:
        """Check if translation service is available"""
        return self._initialized and self.model is not None
    
    def get_supported_languages(self) -> dict:
        """Get dictionary of supported language codes and names"""
        return LANGUAGE_NAMES.copy()


# Global instance (lazy initialization)
_translation_service = None


def get_translation_service() -> TranslationService:
    """Get or create the global TranslationService instance"""
    global _translation_service
    if _translation_service is None:
        _translation_service = TranslationService()
    return _translation_service

