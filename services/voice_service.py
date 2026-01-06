"""
Voice Service - Speech-to-Text (Whisper) and Text-to-Speech (pyttsx3)

Handles:
- Audio transcription using OpenAI Whisper
- Text-to-speech synthesis using pyttsx3
- Audio format conversion
"""

import logging
import io
import tempfile
import os
from pathlib import Path
from typing import Optional, Tuple
import wave

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    logging.warning("Whisper not available. Install: pip install openai-whisper")

try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    logging.warning("pyttsx3 not available. Install: pip install pyttsx3")

logger = logging.getLogger(__name__)


class VoiceService:
    """Handles speech-to-text and text-to-speech operations"""
    
    def __init__(self):
        self.whisper_model = None
        self.tts_engine = None
        self._init_whisper()
        self._init_tts()
    
    def _init_whisper(self):
        """Initialize Whisper model (lazy loading)"""
        if not WHISPER_AVAILABLE:
            logger.warning("Whisper not available - STT will not work")
            return
        
        try:
            # Use base model for balance of speed/accuracy
            # Options: tiny, base, small, medium, large
            logger.info("Loading Whisper model (base)...")
            self.whisper_model = whisper.load_model("base")
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            self.whisper_model = None
    
    def _init_tts(self):
        """Initialize TTS engine"""
        if not TTS_AVAILABLE:
            logger.warning("pyttsx3 not available - TTS will not work")
            return
        
        try:
            self.tts_engine = pyttsx3.init()
            # Set voice properties (optional - can be customized)
            voices = self.tts_engine.getProperty('voices')
            if voices:
                # Try to use a female voice if available, otherwise use default
                for voice in voices:
                    if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                        self.tts_engine.setProperty('voice', voice.id)
                        break
            
            # Set speech rate (words per minute) - default is usually 200
            self.tts_engine.setProperty('rate', 150)
            # Set volume (0.0 to 1.0)
            self.tts_engine.setProperty('volume', 0.9)
            logger.info("TTS engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize TTS engine: {e}")
            self.tts_engine = None
    
    def transcribe_audio(self, audio_data: bytes, language: Optional[str] = None) -> Tuple[str, Optional[str]]:
        """
        Transcribe audio to text using Whisper
        
        Args:
            audio_data: Raw audio bytes (WAV, MP3, etc.)
            language: Optional language code (e.g., 'en', 'es', 'hi')
                     If None, Whisper will auto-detect
        
        Returns:
            Tuple of (transcribed_text, detected_language)
        """
        if not self.whisper_model:
            raise RuntimeError("Whisper model not loaded")
        
        try:
            # Save audio to temporary file with proper permissions
            import stat
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            tmp_file.write(audio_data)
            tmp_path = tmp_file.name
            tmp_file.close()
            
            # Set file permissions to be readable
            os.chmod(tmp_path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
            
            try:
                # Transcribe with Whisper
                # Use fp16=False to avoid potential issues with some systems
                result = self.whisper_model.transcribe(
                    tmp_path,
                    language=language,  # None = auto-detect
                    task="transcribe",
                    fp16=False,  # Use float32 instead of float16 for compatibility
                    verbose=False
                )
                
                text = result["text"].strip()
                detected_lang = result.get("language", None)
                
                logger.debug(f"Transcribed audio: {text[:50]}... (lang: {detected_lang})")
                return text, detected_lang
                
            finally:
                # Clean up temp file
                try:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                except Exception as cleanup_error:
                    logger.warning(f"Failed to cleanup temp file: {cleanup_error}")
                    
        except FileNotFoundError as e:
            if 'ffmpeg' in str(e).lower():
                error_msg = (
                    "ffmpeg is required for audio processing but not found. "
                    "Please install ffmpeg:\n"
                    "  Ubuntu/Debian: sudo apt-get install ffmpeg\n"
                    "  macOS: brew install ffmpeg\n"
                    "  Windows: Download from https://ffmpeg.org/download.html"
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e
            raise
        except PermissionError as e:
            error_msg = (
                f"Permission denied accessing audio file. "
                f"This may be a temporary file permission issue. Error: {e}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise
    
    def text_to_speech(self, text: str, output_format: str = "wav") -> bytes:
        """
        Convert text to speech audio
        
        Args:
            text: Text to synthesize
            output_format: Output format ('wav' or 'mp3')
        
        Returns:
            Audio data as bytes
        """
        if not self.tts_engine:
            raise RuntimeError("TTS engine not initialized")
        
        try:
            # Create temporary file for output
            with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{output_format}') as tmp_file:
                tmp_path = tmp_file.name
            
            try:
                # Save speech to file
                self.tts_engine.save_to_file(text, tmp_path)
                self.tts_engine.runAndWait()
                
                # Read the generated audio file
                with open(tmp_path, 'rb') as f:
                    audio_data = f.read()
                
                logger.debug(f"Generated TTS audio: {len(audio_data)} bytes")
                return audio_data
                
            finally:
                # Clean up temp file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                    
        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
            raise
    
    def is_stt_available(self) -> bool:
        """Check if speech-to-text is available"""
        return self.whisper_model is not None
    
    def is_tts_available(self) -> bool:
        """Check if text-to-speech is available"""
        return self.tts_engine is not None


# Global instance (lazy initialization)
_voice_service = None


def get_voice_service() -> VoiceService:
    """Get or create the global VoiceService instance"""
    global _voice_service
    if _voice_service is None:
        _voice_service = VoiceService()
    return _voice_service

