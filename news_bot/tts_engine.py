"""
Text-to-Speech engine using Coqui TTS.

Converts script text to audio files for the newsletter podcast.
"""

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# =============================================================================
# TTS Configuration
# =============================================================================

@dataclass
class TTSConfig:
    """Configuration for TTS generation."""
    
    # Model settings
    model_name: str = "tts_models/en/ljspeech/tacotron2-DDC"
    vocoder_name: Optional[str] = None  # Use default vocoder for the model
    
    # Language setting (required for multilingual models like XTTS)
    language: str = "en"
    
    # Speaker settings for multi-speaker models (XTTS)
    speaker: Optional[str] = "Claribel Dervla"  # Default XTTS speaker
    speaker_wav: Optional[str] = None  # Optional: path to WAV file for voice cloning
    
    # Speed control (1.0 = normal, 1.2 = 20% faster, 0.8 = 20% slower)
    speed: float = 1.0
    
    # Output settings
    output_dir: Path = Path("audio_output")
    sample_rate: int = 22050
    
    # Processing settings
    use_cuda: bool = False  # Set to True if GPU available
    
    def __post_init__(self):
        """Ensure output_dir is a Path object."""
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
    
    @property
    def is_xtts_model(self) -> bool:
        """Check if this is an XTTS model (requires special handling)."""
        return "xtts" in self.model_name.lower()


class TTSEngineError(Exception):
    """Raised when TTS generation fails."""
    pass


class TTSEngine:
    """
    Text-to-Speech engine wrapper for Coqui TTS.
    
    Provides a clean interface for converting text to audio files.
    Handles model loading, text preprocessing, and audio generation.
    """
    
    def __init__(self, config: Optional[TTSConfig] = None):
        """
        Initialize the TTS engine.
        
        Args:
            config: TTS configuration. Uses defaults if not provided.
        """
        self.config = config or TTSConfig()
        self._tts = None
        self._initialized = False
        
        # Ensure output directory exists
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _ensure_initialized(self) -> None:
        """Lazy-load the TTS model on first use."""
        if self._initialized:
            return
        
        try:
            from TTS.api import TTS
        except ImportError:
            raise TTSEngineError(
                "Coqui TTS is not installed. Install with: pip install TTS"
            )
        
        # For PyTorch 2.6+, we need to allowlist TTS config classes for model loading
        try:
            import torch
            safe_classes = []
            
            # Try to import and add XTTS config classes
            try:
                from TTS.tts.configs.xtts_config import XttsConfig
                safe_classes.append(XttsConfig)
            except ImportError:
                pass
            
            # Add other common TTS config classes that might be pickled
            try:
                from TTS.config import BaseAudioConfig, BaseDatasetConfig, BaseTrainingConfig
                safe_classes.extend([BaseAudioConfig, BaseDatasetConfig, BaseTrainingConfig])
            except ImportError:
                pass
            
            try:
                from TTS.tts.configs.shared_configs import BaseTTSConfig, CharactersConfig
                safe_classes.extend([BaseTTSConfig, CharactersConfig])
            except ImportError:
                pass
            
            if safe_classes and hasattr(torch.serialization, 'add_safe_globals'):
                torch.serialization.add_safe_globals(safe_classes)
                logger.debug(f"Added {len(safe_classes)} TTS classes to torch safe globals")
        except Exception as e:
            # Older PyTorch versions don't have add_safe_globals
            logger.debug(f"Could not add safe globals (may not be needed): {e}")
        
        logger.info(f"Loading TTS model: {self.config.model_name}")
        start_time = datetime.now()
        
        try:
            self._tts = TTS(
                model_name=self.config.model_name,
                progress_bar=False,
                gpu=self.config.use_cuda,
            )
        except Exception as e:
            raise TTSEngineError(f"Failed to load TTS model: {e}")
        
        duration_ms = (datetime.now() - start_time).total_seconds() * 1000
        logger.info(f"TTS model loaded in {duration_ms:.0f}ms")
        
        self._initialized = True
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for optimal TTS output.
        
        Handles:
        - Normalizing whitespace
        - Converting abbreviations
        - Handling special characters
        - Managing pauses
        
        Args:
            text: Raw script text.
        
        Returns:
            Preprocessed text suitable for TTS.
        """
        # Normalize whitespace
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'\t', ' ', text)
        text = re.sub(r' +', ' ', text)
        
        # Fix escaped apostrophes and quotes (common LLM output issue)
        text = text.replace("\\'", "'")
        text = text.replace('\\"', '"')
        text = text.replace("\\`", "`")
        
        # Handle pause markers - convert "..." to comma for natural pause
        text = re.sub(r'\.{3,}', ', ', text)
        
        # Common financial abbreviations to full words
        abbreviations = {
            r'\bQ1\b': 'first quarter',
            r'\bQ2\b': 'second quarter',
            r'\bQ3\b': 'third quarter',
            r'\bQ4\b': 'fourth quarter',
            r'\bYoY\b': 'year over year',
            r'\bQoQ\b': 'quarter over quarter',
            r'\bMoM\b': 'month over month',
            r'\bIPO\b': 'I P O',
            r'\bCEO\b': 'C E O',
            r'\bCFO\b': 'C F O',
            r'\bCOO\b': 'C O O',
            r'\bM&A\b': 'M and A',
            r'\bP/E\b': 'P E ratio',
            r'\bEPS\b': 'E P S',
            r'\bGDP\b': 'G D P',
            r'\bFed\b': 'the Fed',
            r'\bSEC\b': 'S E C',
            r'\bNASDAQ\b': 'NASDAQ',
            r'\bNYSE\b': 'N Y S E',
            r'\bETF\b': 'E T F',
            r'\bESG\b': 'E S G',
            r'\bAI\b': 'A I',
            r'\bEV\b': 'E V',
            r'\bROI\b': 'R O I',
            r'\bEBITDA\b': 'EBITDA',
        }
        
        for pattern, replacement in abbreviations.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Handle currency symbols
        text = re.sub(r'\$(\d)', r'\1 dollars', text)
        text = re.sub(r'€(\d)', r'\1 euros', text)
        text = re.sub(r'£(\d)', r'\1 pounds', text)
        
        # Handle percentage symbol after numbers
        text = re.sub(r'(\d+(?:\.\d+)?)\s*%', r'\1 percent', text)
        
        # Handle large numbers with M/B/T suffixes
        text = re.sub(r'(\d+(?:\.\d+)?)\s*[Mm]\b', r'\1 million', text)
        text = re.sub(r'(\d+(?:\.\d+)?)\s*[Bb]\b', r'\1 billion', text)
        text = re.sub(r'(\d+(?:\.\d+)?)\s*[Tt]\b', r'\1 trillion', text)
        
        # Clean up any resulting double spaces
        text = re.sub(r' +', ' ', text)
        
        return text.strip()
    
    def synthesize(
        self,
        text: str,
        output_filename: Optional[str] = None,
    ) -> Path:
        """
        Synthesize speech from text and save to file.
        
        Args:
            text: Script text to convert to speech.
            output_filename: Optional filename for output. 
                           Auto-generated if not provided.
        
        Returns:
            Path to the generated audio file.
        
        Raises:
            TTSEngineError: If synthesis fails.
        """
        self._ensure_initialized()
        
        if not text or not text.strip():
            raise TTSEngineError("Cannot synthesize empty text")
        
        # Generate filename if not provided
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"broadcast_{timestamp}.wav"
        
        # Ensure .wav extension
        if not output_filename.endswith('.wav'):
            output_filename = f"{output_filename}.wav"
        
        output_path = self.config.output_dir / output_filename
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        logger.info(f"Synthesizing {len(processed_text)} characters of text...")
        start_time = datetime.now()
        
        try:
            # XTTS models require language and speaker parameters
            if self.config.is_xtts_model:
                tts_kwargs = {
                    "text": processed_text,
                    "file_path": str(output_path),
                    "language": self.config.language,
                    "speed": self.config.speed,
                }
                # Use speaker_wav for voice cloning, otherwise use named speaker
                if self.config.speaker_wav:
                    tts_kwargs["speaker_wav"] = self.config.speaker_wav
                elif self.config.speaker:
                    tts_kwargs["speaker"] = self.config.speaker
                
                self._tts.tts_to_file(**tts_kwargs)
            else:
                # Standard models (Tacotron, VITS, etc.)
                self._tts.tts_to_file(
                    text=processed_text,
                    file_path=str(output_path),
                    speed=self.config.speed,
                )
        except Exception as e:
            raise TTSEngineError(f"TTS synthesis failed: {e}")
        
        duration_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        # Get file size
        file_size_kb = output_path.stat().st_size / 1024
        
        logger.info(
            f"Audio generated: {output_path.name} "
            f"({file_size_kb:.1f} KB) in {duration_ms:.0f}ms"
        )
        
        return output_path
    
    def synthesize_to_mp3(
        self,
        text: str,
        output_filename: Optional[str] = None,
    ) -> Path:
        """
        Synthesize speech and convert to MP3 format.
        
        Generates WAV first, then converts to MP3 using pydub.
        
        Args:
            text: Script text to convert to speech.
            output_filename: Optional filename for output (without extension).
        
        Returns:
            Path to the generated MP3 file.
        
        Raises:
            TTSEngineError: If synthesis or conversion fails.
        """
        try:
            from pydub import AudioSegment
        except ImportError:
            raise TTSEngineError(
                "pydub is required for MP3 conversion. Install with: pip install pydub"
            )
        
        # Generate WAV first
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"broadcast_{timestamp}"
        
        # Remove any extension from filename
        output_filename = output_filename.rsplit('.', 1)[0]
        
        wav_path = self.synthesize(text, f"{output_filename}.wav")
        mp3_path = self.config.output_dir / f"{output_filename}.mp3"
        
        logger.info(f"Converting to MP3: {mp3_path.name}")
        start_time = datetime.now()
        
        try:
            audio = AudioSegment.from_wav(str(wav_path))
            audio.export(
                str(mp3_path),
                format="mp3",
                bitrate="192k",
            )
        except Exception as e:
            raise TTSEngineError(f"MP3 conversion failed: {e}")
        
        # Clean up WAV file
        try:
            wav_path.unlink()
        except OSError:
            logger.warning(f"Could not delete temporary WAV file: {wav_path}")
        
        duration_ms = (datetime.now() - start_time).total_seconds() * 1000
        file_size_kb = mp3_path.stat().st_size / 1024
        
        logger.info(
            f"MP3 generated: {mp3_path.name} "
            f"({file_size_kb:.1f} KB) in {duration_ms:.0f}ms"
        )
        
        return mp3_path


# =============================================================================
# Convenience Functions
# =============================================================================

def generate_audio(
    script: str,
    output_dir: Optional[Path] = None,
    output_filename: Optional[str] = None,
    use_cuda: bool = False,
) -> Path:
    """
    Generate audio from a script text.
    
    Convenience function that creates a TTSEngine and synthesizes audio.
    
    Args:
        script: The script text to convert to speech.
        output_dir: Directory for output files. Defaults to 'audio_output'.
        output_filename: Optional filename (without extension).
        use_cuda: Whether to use GPU acceleration.
    
    Returns:
        Path to the generated MP3 file.
    """
    config = TTSConfig(
        output_dir=output_dir or Path("audio_output"),
        use_cuda=use_cuda,
    )
    
    engine = TTSEngine(config)
    return engine.synthesize_to_mp3(script, output_filename)


def generate_broadcast_audio(
    script: str,
    broadcast_date: Optional[datetime] = None,
    output_dir: Optional[Path] = None,
) -> Path:
    """
    Generate broadcast audio with date-based filename.
    
    Creates an MP3 file named like: broadcast_20260117.mp3
    
    Args:
        script: The broadcast script text.
        broadcast_date: Date for filename. Defaults to today.
        output_dir: Directory for output files.
    
    Returns:
        Path to the generated MP3 file.
    """
    date = broadcast_date or datetime.now()
    filename = f"broadcast_{date.strftime('%Y%m%d')}"
    
    return generate_audio(
        script=script,
        output_dir=output_dir,
        output_filename=filename,
    )
