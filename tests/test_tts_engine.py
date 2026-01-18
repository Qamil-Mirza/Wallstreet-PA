"""Tests for TTS engine functionality."""

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
import tempfile

import pytest

from news_bot.tts_engine import (
    TTSConfig,
    TTSEngine,
    TTSEngineError,
    generate_audio,
    generate_broadcast_audio,
)


# =============================================================================
# TTSConfig Tests
# =============================================================================

class TestTTSConfig:
    """Tests for TTSConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TTSConfig()
        
        assert config.model_name == "tts_models/en/ljspeech/tacotron2-DDC"
        assert config.vocoder_name is None
        assert config.output_dir == Path("audio_output")
        assert config.sample_rate == 22050
        assert config.use_cuda is False

    def test_custom_values(self):
        """Test custom configuration values."""
        config = TTSConfig(
            model_name="tts_models/en/vctk/vits",
            output_dir=Path("/custom/path"),
            use_cuda=True,
        )
        
        assert config.model_name == "tts_models/en/vctk/vits"
        assert config.output_dir == Path("/custom/path")
        assert config.use_cuda is True

    def test_string_output_dir_converted_to_path(self):
        """Test that string output_dir is converted to Path."""
        config = TTSConfig(output_dir="/some/path")
        
        assert isinstance(config.output_dir, Path)
        assert config.output_dir == Path("/some/path")

    def test_is_xtts_model_true_for_xtts(self):
        """Test is_xtts_model returns True for XTTS models."""
        config = TTSConfig(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
        assert config.is_xtts_model is True
        
        config = TTSConfig(model_name="tts_models/en/vctk/XTTS_v1")
        assert config.is_xtts_model is True

    def test_is_xtts_model_false_for_other_models(self):
        """Test is_xtts_model returns False for non-XTTS models."""
        config = TTSConfig(model_name="tts_models/en/ljspeech/tacotron2-DDC")
        assert config.is_xtts_model is False
        
        config = TTSConfig(model_name="tts_models/en/vctk/vits")
        assert config.is_xtts_model is False

    def test_xtts_config_fields(self):
        """Test XTTS-specific configuration fields."""
        config = TTSConfig(
            model_name="tts_models/multilingual/multi-dataset/xtts_v2",
            language="en",
            speaker="Claribel Dervla",
            speaker_wav="/path/to/voice.wav",
            speed=1.2,
        )
        
        assert config.language == "en"
        assert config.speaker == "Claribel Dervla"
        assert config.speaker_wav == "/path/to/voice.wav"
        assert config.speed == 1.2

    def test_default_xtts_settings(self):
        """Test default values for XTTS settings."""
        config = TTSConfig()
        
        assert config.language == "en"
        assert config.speaker == "Claribel Dervla"
        assert config.speaker_wav is None
        assert config.speed == 1.0


# =============================================================================
# TTSEngine Preprocessing Tests
# =============================================================================

class TestTTSEnginePreprocessing:
    """Tests for TTSEngine text preprocessing."""

    @pytest.fixture
    def engine(self, tmp_path):
        """Create a TTSEngine with temp directory."""
        config = TTSConfig(output_dir=tmp_path)
        return TTSEngine(config)

    def test_preprocess_normalizes_whitespace(self, engine):
        """Test that whitespace is normalized."""
        text = "Multiple   spaces\tand\ttabs"
        result = engine.preprocess_text(text)
        
        assert "  " not in result
        assert "\t" not in result

    def test_preprocess_converts_pause_markers(self, engine):
        """Test that '...' is converted to comma for pause."""
        text = "Markets are moving... and fast"
        result = engine.preprocess_text(text)
        
        assert "..." not in result
        assert ", " in result

    def test_preprocess_expands_quarter_abbreviations(self, engine):
        """Test expansion of quarter abbreviations."""
        text = "Q1 earnings beat Q2 expectations, while Q3 and Q4 look strong."
        result = engine.preprocess_text(text)
        
        assert "first quarter" in result
        assert "second quarter" in result
        assert "third quarter" in result
        assert "fourth quarter" in result

    def test_preprocess_expands_financial_abbreviations(self, engine):
        """Test expansion of common financial abbreviations."""
        test_cases = [
            ("YoY growth", "year over year growth"),
            ("QoQ comparison", "quarter over quarter comparison"),
            ("The CEO announced", "The C E O announced"),
            ("EPS beat estimates", "E P S beat estimates"),
            ("The Fed raised rates", "the Fed raised rates"),
        ]
        
        for input_text, expected_phrase in test_cases:
            result = engine.preprocess_text(input_text)
            assert expected_phrase in result, f"Failed for: {input_text}"

    def test_preprocess_handles_currency(self, engine):
        """Test handling of currency symbols."""
        text = "Revenue hit $5 billion and €3 million in Europe."
        result = engine.preprocess_text(text)
        
        assert "$" not in result
        assert "€" not in result
        assert "dollars" in result
        assert "euros" in result

    def test_preprocess_handles_percentages(self, engine):
        """Test handling of percentage symbols."""
        text = "Growth of 25% and margins at 15.5%"
        result = engine.preprocess_text(text)
        
        assert "%" not in result
        assert "25 percent" in result
        assert "15.5 percent" in result

    def test_preprocess_handles_large_number_suffixes(self, engine):
        """Test handling of M/B/T number suffixes."""
        text = "Revenue of 5.2B and costs of 100M"
        result = engine.preprocess_text(text)
        
        assert "5.2 billion" in result
        assert "100 million" in result

    def test_preprocess_preserves_normal_text(self, engine):
        """Test that normal text is preserved."""
        text = "The market closed higher today on strong earnings."
        result = engine.preprocess_text(text)
        
        assert result == text

    def test_preprocess_fixes_escaped_apostrophes(self, engine):
        """Test that escaped apostrophes are fixed."""
        text = "Here\\'s what you need to know: the market\\'s up."
        result = engine.preprocess_text(text)
        
        assert "\\'" not in result
        assert "Here's" in result
        assert "market's" in result

    def test_preprocess_fixes_escaped_quotes(self, engine):
        """Test that escaped quotes are fixed."""
        text = 'The CEO said \\"growth is strong\\" today.'
        result = engine.preprocess_text(text)
        
        assert '\\"' not in result
        assert '"growth is strong"' in result


# =============================================================================
# TTSEngine Text Chunking Tests
# =============================================================================

class TestTTSEngineChunking:
    """Tests for TTSEngine text chunking for XTTS models."""

    @pytest.fixture
    def engine(self, tmp_path):
        """Create a TTSEngine with temp directory."""
        config = TTSConfig(output_dir=tmp_path)
        return TTSEngine(config)

    def test_chunk_text_short_text_no_split(self, engine):
        """Test that short text is not split."""
        text = "This is a short sentence."
        chunks = engine._chunk_text(text, max_chars=240)
        
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_text_exact_limit_no_split(self, engine):
        """Test that text exactly at limit is not split."""
        text = "A" * 240
        chunks = engine._chunk_text(text, max_chars=240)
        
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_text_splits_at_sentence_boundary(self, engine):
        """Test chunking prefers sentence boundaries."""
        text = "First sentence here. Second sentence is much longer and needs to fit."
        chunks = engine._chunk_text(text, max_chars=25)
        
        assert len(chunks) >= 2
        # First chunk should end at sentence boundary
        assert chunks[0].endswith(".")

    def test_chunk_text_splits_at_exclamation(self, engine):
        """Test chunking at exclamation marks."""
        text = "Exciting news! The markets are moving today."
        chunks = engine._chunk_text(text, max_chars=20)
        
        assert len(chunks) >= 2
        # Check that exclamation is preserved
        assert any("!" in chunk for chunk in chunks)

    def test_chunk_text_splits_at_question(self, engine):
        """Test chunking at question marks."""
        text = "What happened today? The market dropped significantly."
        chunks = engine._chunk_text(text, max_chars=25)
        
        assert len(chunks) >= 2

    def test_chunk_text_falls_back_to_comma(self, engine):
        """Test chunking falls back to commas when no sentence boundary."""
        text = "No sentence ends here, but we have commas, to split on"
        chunks = engine._chunk_text(text, max_chars=30)
        
        assert len(chunks) >= 2

    def test_chunk_text_falls_back_to_semicolon(self, engine):
        """Test chunking falls back to semicolons."""
        text = "First part here; second part follows; third part ends"
        chunks = engine._chunk_text(text, max_chars=25)
        
        assert len(chunks) >= 2

    def test_chunk_text_falls_back_to_space(self, engine):
        """Test chunking falls back to word boundaries."""
        text = "this text has no punctuation but lots of words to split on"
        chunks = engine._chunk_text(text, max_chars=30)
        
        assert len(chunks) >= 2
        # Chunks should be split at word boundaries
        for chunk in chunks:
            assert not chunk.startswith(" ")
            assert not chunk.endswith(" ")

    def test_chunk_text_hard_cut_no_spaces(self, engine):
        """Test hard cut when no natural split point."""
        text = "A" * 300  # No spaces or punctuation
        chunks = engine._chunk_text(text, max_chars=100)
        
        assert len(chunks) >= 3
        # First chunks should be max length
        assert len(chunks[0]) == 100

    def test_chunk_text_empty_string(self, engine):
        """Test chunking empty string returns single empty chunk."""
        chunks = engine._chunk_text("", max_chars=240)
        # Empty string returns [''] - synthesis will handle empty text validation
        assert len(chunks) <= 1

    def test_chunk_text_whitespace_only(self, engine):
        """Test chunking whitespace handles gracefully."""
        chunks = engine._chunk_text("   \n\t  ", max_chars=240)
        # Whitespace-only text is handled at synthesis level
        assert isinstance(chunks, list)

    def test_chunk_text_custom_max_chars(self, engine):
        """Test chunking with custom max_chars parameter."""
        text = "Short. Medium sentence. A longer sentence that needs splitting."
        
        chunks_50 = engine._chunk_text(text, max_chars=50)
        chunks_100 = engine._chunk_text(text, max_chars=100)
        
        # Smaller max should produce more chunks
        assert len(chunks_50) >= len(chunks_100)

    def test_chunk_text_preserves_all_content(self, engine):
        """Test that all content is preserved after chunking."""
        text = "First part. Second part. Third part with more content here."
        chunks = engine._chunk_text(text, max_chars=30)
        
        # Rejoin chunks and compare (allowing for trimmed whitespace)
        rejoined = " ".join(chunks)
        # All words should be present
        for word in text.split():
            assert word in rejoined

    def test_chunk_text_realistic_broadcast_text(self, engine):
        """Test chunking with realistic broadcast-style text."""
        text = (
            "Good morning, Wall Street! The bulls are running today as tech stocks surge. "
            "Nvidia reported earnings that crushed expectations, sending shares up fifteen percent. "
            "Meanwhile, the Fed signals potential rate cuts, boosting market sentiment. "
            "Here's what you need to know for your portfolio today."
        )
        chunks = engine._chunk_text(text, max_chars=240)
        
        # All chunks should be within limit
        for chunk in chunks:
            assert len(chunk) <= 240
        
        # Content should be preserved
        assert "Wall Street" in " ".join(chunks)
        assert "Nvidia" in " ".join(chunks)
        assert "Fed" in " ".join(chunks)


# =============================================================================
# TTSEngine Initialization Tests
# =============================================================================

class TestTTSEngineInitialization:
    """Tests for TTSEngine initialization."""

    def test_creates_output_directory(self, tmp_path):
        """Test that output directory is created on init."""
        output_dir = tmp_path / "new_audio_dir"
        config = TTSConfig(output_dir=output_dir)
        
        engine = TTSEngine(config)
        
        assert output_dir.exists()

    def test_lazy_loads_tts_model(self, tmp_path):
        """Test that TTS model is not loaded until needed."""
        config = TTSConfig(output_dir=tmp_path)
        engine = TTSEngine(config)
        
        assert engine._tts is None
        assert engine._initialized is False

    def test_import_error_on_missing_tts(self, tmp_path):
        """Test that missing TTS package raises helpful error."""
        config = TTSConfig(output_dir=tmp_path)
        engine = TTSEngine(config)
        
        with patch.dict('sys.modules', {'TTS.api': None}):
            with patch('builtins.__import__', side_effect=ImportError("No module named 'TTS'")):
                with pytest.raises(TTSEngineError) as exc_info:
                    engine._ensure_initialized()
                
                assert "Coqui TTS is not installed" in str(exc_info.value)


# =============================================================================
# TTSEngine Synthesis Tests
# =============================================================================

class TestTTSEngineSynthesis:
    """Tests for TTSEngine synthesis methods."""

    @pytest.fixture
    def mock_tts(self):
        """Create a mock TTS object."""
        mock = MagicMock()
        mock.tts_to_file = MagicMock()
        return mock

    @pytest.fixture
    def engine_with_mock_tts(self, tmp_path, mock_tts):
        """Create a TTSEngine with mocked TTS backend."""
        config = TTSConfig(output_dir=tmp_path)
        engine = TTSEngine(config)
        engine._tts = mock_tts
        engine._initialized = True
        return engine

    def test_synthesize_empty_text_raises_error(self, engine_with_mock_tts):
        """Test that empty text raises an error."""
        with pytest.raises(TTSEngineError) as exc_info:
            engine_with_mock_tts.synthesize("")
        
        assert "empty text" in str(exc_info.value).lower()

    def test_synthesize_whitespace_only_raises_error(self, engine_with_mock_tts):
        """Test that whitespace-only text raises an error."""
        with pytest.raises(TTSEngineError) as exc_info:
            engine_with_mock_tts.synthesize("   \n\t  ")
        
        assert "empty text" in str(exc_info.value).lower()

    def test_synthesize_generates_filename_when_not_provided(self, engine_with_mock_tts, tmp_path):
        """Test automatic filename generation."""
        # Create a dummy output file that accepts keyword arguments
        def create_dummy_wav(text, file_path, **kwargs):
            Path(file_path).write_bytes(b"dummy wav content")
        
        engine_with_mock_tts._tts.tts_to_file.side_effect = create_dummy_wav
        
        result = engine_with_mock_tts.synthesize("Hello world")
        
        assert result.suffix == ".wav"
        assert "broadcast_" in result.name

    def test_synthesize_uses_provided_filename(self, engine_with_mock_tts, tmp_path):
        """Test that provided filename is used."""
        def create_dummy_wav(text, file_path, **kwargs):
            Path(file_path).write_bytes(b"dummy wav content")
        
        engine_with_mock_tts._tts.tts_to_file.side_effect = create_dummy_wav
        
        result = engine_with_mock_tts.synthesize("Hello", output_filename="custom_name.wav")
        
        assert result.name == "custom_name.wav"

    def test_synthesize_adds_wav_extension(self, engine_with_mock_tts, tmp_path):
        """Test that .wav extension is added if missing."""
        def create_dummy_wav(text, file_path, **kwargs):
            Path(file_path).write_bytes(b"dummy wav content")
        
        engine_with_mock_tts._tts.tts_to_file.side_effect = create_dummy_wav
        
        result = engine_with_mock_tts.synthesize("Hello", output_filename="no_extension")
        
        assert result.name == "no_extension.wav"

    def test_synthesize_preprocesses_text(self, engine_with_mock_tts, tmp_path):
        """Test that text is preprocessed before synthesis."""
        def capture_text(text, file_path, **kwargs):
            Path(file_path).write_bytes(b"dummy")
            # Store the text that was passed
            capture_text.received_text = text
        
        engine_with_mock_tts._tts.tts_to_file.side_effect = capture_text
        
        engine_with_mock_tts.synthesize("Revenue grew 25% in Q1...")
        
        # Check that preprocessing was applied
        assert "percent" in capture_text.received_text
        assert "first quarter" in capture_text.received_text
        assert "..." not in capture_text.received_text

    def test_synthesize_returns_path_object(self, engine_with_mock_tts, tmp_path):
        """Test that synthesize returns a Path object."""
        def create_dummy_wav(text, file_path, **kwargs):
            Path(file_path).write_bytes(b"dummy")
        
        engine_with_mock_tts._tts.tts_to_file.side_effect = create_dummy_wav
        
        result = engine_with_mock_tts.synthesize("Hello")
        
        assert isinstance(result, Path)

    def test_synthesize_handles_tts_error(self, engine_with_mock_tts):
        """Test handling of TTS synthesis errors."""
        engine_with_mock_tts._tts.tts_to_file.side_effect = RuntimeError("TTS failed")
        
        with pytest.raises(TTSEngineError) as exc_info:
            engine_with_mock_tts.synthesize("Hello")
        
        assert "synthesis failed" in str(exc_info.value).lower()


# =============================================================================
# TTSEngine XTTS-Specific Tests
# =============================================================================

class TestTTSEngineXTTS:
    """Tests for XTTS-specific TTS engine behavior."""

    @pytest.fixture
    def mock_tts(self):
        """Create a mock TTS object."""
        mock = MagicMock()
        mock.tts_to_file = MagicMock()
        return mock

    @pytest.fixture
    def xtts_engine(self, tmp_path, mock_tts):
        """Create a TTSEngine configured for XTTS."""
        config = TTSConfig(
            model_name="tts_models/multilingual/multi-dataset/xtts_v2",
            output_dir=tmp_path,
            language="en",
            speaker="Claribel Dervla",
            speed=1.0,
        )
        engine = TTSEngine(config)
        engine._tts = mock_tts
        engine._initialized = True
        return engine

    def test_xtts_short_text_direct_synthesis(self, xtts_engine, tmp_path):
        """Test that short text is synthesized directly without chunking."""
        def create_dummy_wav(**kwargs):
            Path(kwargs['file_path']).write_bytes(b"RIFF" + b"\x00" * 100)
        
        xtts_engine._tts.tts_to_file.side_effect = create_dummy_wav
        
        # Text under 240 chars
        short_text = "Hello, this is a short test."
        xtts_engine.synthesize(short_text)
        
        # Should call tts_to_file once with all params
        xtts_engine._tts.tts_to_file.assert_called_once()
        call_kwargs = xtts_engine._tts.tts_to_file.call_args[1]
        
        assert "language" in call_kwargs
        assert call_kwargs["language"] == "en"

    def test_xtts_includes_speaker_param(self, xtts_engine, tmp_path):
        """Test that XTTS includes speaker parameter."""
        def create_dummy_wav(**kwargs):
            Path(kwargs['file_path']).write_bytes(b"RIFF" + b"\x00" * 100)
        
        xtts_engine._tts.tts_to_file.side_effect = create_dummy_wav
        xtts_engine.synthesize("Test text")
        
        call_kwargs = xtts_engine._tts.tts_to_file.call_args[1]
        assert call_kwargs["speaker"] == "Claribel Dervla"

    def test_xtts_speaker_wav_overrides_speaker(self, tmp_path, mock_tts):
        """Test that speaker_wav takes precedence over speaker name."""
        config = TTSConfig(
            model_name="tts_models/multilingual/multi-dataset/xtts_v2",
            output_dir=tmp_path,
            speaker="Some Speaker",
            speaker_wav="/path/to/voice.wav",
        )
        engine = TTSEngine(config)
        engine._tts = mock_tts
        engine._initialized = True
        
        def create_dummy_wav(**kwargs):
            Path(kwargs['file_path']).write_bytes(b"RIFF" + b"\x00" * 100)
        
        mock_tts.tts_to_file.side_effect = create_dummy_wav
        engine.synthesize("Test text")
        
        call_kwargs = mock_tts.tts_to_file.call_args[1]
        assert call_kwargs.get("speaker_wav") == "/path/to/voice.wav"
        assert "speaker" not in call_kwargs

    def test_xtts_speed_parameter(self, tmp_path, mock_tts):
        """Test that speed parameter is passed to XTTS."""
        config = TTSConfig(
            model_name="tts_models/multilingual/multi-dataset/xtts_v2",
            output_dir=tmp_path,
            speed=1.3,
        )
        engine = TTSEngine(config)
        engine._tts = mock_tts
        engine._initialized = True
        
        def create_dummy_wav(**kwargs):
            Path(kwargs['file_path']).write_bytes(b"RIFF" + b"\x00" * 100)
        
        mock_tts.tts_to_file.side_effect = create_dummy_wav
        engine.synthesize("Test text")
        
        call_kwargs = mock_tts.tts_to_file.call_args[1]
        assert call_kwargs["speed"] == 1.3

    def test_xtts_long_text_triggers_chunking(self, xtts_engine, tmp_path):
        """Test that long text triggers chunking for XTTS."""
        # Create a long text that exceeds 240 chars
        long_text = "This is a sentence. " * 20  # ~380 chars
        
        # Simply verify that _chunk_text would produce multiple chunks
        chunks = xtts_engine._chunk_text(xtts_engine.preprocess_text(long_text), max_chars=240)
        
        # Should have multiple chunks
        assert len(chunks) > 1
        
        # Each chunk should be within limit
        for chunk in chunks:
            assert len(chunk) <= 240

    def test_non_xtts_model_no_language_param(self, tmp_path, mock_tts):
        """Test that non-XTTS models don't get language parameter."""
        config = TTSConfig(
            model_name="tts_models/en/ljspeech/tacotron2-DDC",
            output_dir=tmp_path,
        )
        engine = TTSEngine(config)
        engine._tts = mock_tts
        engine._initialized = True
        
        def create_dummy_wav(text, file_path, **kwargs):
            Path(file_path).write_bytes(b"RIFF" + b"\x00" * 100)
        
        mock_tts.tts_to_file.side_effect = create_dummy_wav
        engine.synthesize("Test text")
        
        call_kwargs = mock_tts.tts_to_file.call_args[1]
        assert "language" not in call_kwargs
        assert "speaker" not in call_kwargs


# =============================================================================
# TTSEngine Chunk Synthesis Tests
# =============================================================================

class TestTTSEngineSynthesizeChunks:
    """Tests for _synthesize_chunks method."""

    @pytest.fixture
    def mock_tts(self):
        """Create a mock TTS object."""
        mock = MagicMock()
        return mock

    @pytest.fixture
    def xtts_engine(self, tmp_path, mock_tts):
        """Create a TTSEngine configured for XTTS."""
        config = TTSConfig(
            model_name="tts_models/multilingual/multi-dataset/xtts_v2",
            output_dir=tmp_path,
        )
        engine = TTSEngine(config)
        engine._tts = mock_tts
        engine._initialized = True
        return engine

    def test_synthesize_chunks_passes_xtts_params(self, xtts_engine, tmp_path, mock_tts):
        """Test that XTTS parameters are passed for each chunk via _synthesize_chunks."""
        # Patch pydub at import level for the method
        import pydub
        original_audio_segment = pydub.AudioSegment
        
        chunks = ["Test chunk one.", "Test chunk two."]
        
        def create_dummy_wav(**kwargs):
            Path(kwargs['file_path']).write_bytes(b"RIFF" + b"\x00" * 100)
        
        mock_tts.tts_to_file.side_effect = create_dummy_wav
        
        # Create mock AudioSegment
        mock_combined = MagicMock()
        mock_audio = MagicMock()
        
        class MockAudioSegment:
            @staticmethod
            def empty():
                return mock_combined
            
            @staticmethod
            def from_wav(path):
                return mock_audio
        
        mock_combined.__iadd__ = MagicMock(return_value=mock_combined)
        
        try:
            pydub.AudioSegment = MockAudioSegment
            output_path = tmp_path / "output.wav"
            xtts_engine._synthesize_chunks(chunks, output_path)
            
            # Should call TTS twice
            assert mock_tts.tts_to_file.call_count == 2
            
            # Check parameters
            call_kwargs = mock_tts.tts_to_file.call_args[1]
            assert call_kwargs.get("language") == "en"
            assert "speed" in call_kwargs
        finally:
            pydub.AudioSegment = original_audio_segment

    def test_synthesize_chunks_temp_file_naming(self, xtts_engine, tmp_path, mock_tts):
        """Test that temp files are named correctly during chunking."""
        chunks = ["First.", "Second.", "Third."]
        
        file_paths = []
        def capture_paths(**kwargs):
            file_paths.append(kwargs['file_path'])
            Path(kwargs['file_path']).write_bytes(b"RIFF" + b"\x00" * 100)
        
        mock_tts.tts_to_file.side_effect = capture_paths
        
        import pydub
        original_audio_segment = pydub.AudioSegment
        
        mock_combined = MagicMock()
        
        class MockAudioSegment:
            @staticmethod
            def empty():
                return mock_combined
            
            @staticmethod
            def from_wav(path):
                return MagicMock()
        
        mock_combined.__iadd__ = MagicMock(return_value=mock_combined)
        
        try:
            pydub.AudioSegment = MockAudioSegment
            output_path = tmp_path / "output.wav"
            xtts_engine._synthesize_chunks(chunks, output_path)
            
            # Check temp file naming pattern
            assert any("_temp_chunk_0" in p for p in file_paths)
            assert any("_temp_chunk_1" in p for p in file_paths)
            assert any("_temp_chunk_2" in p for p in file_paths)
        finally:
            pydub.AudioSegment = original_audio_segment


# =============================================================================
# TTSEngine MP3 Conversion Tests
# =============================================================================

class TestTTSEngineMP3Conversion:
    """Tests for TTSEngine MP3 conversion."""

    @pytest.fixture
    def mock_tts(self):
        """Create a mock TTS object."""
        mock = MagicMock()
        return mock

    @pytest.fixture
    def engine_with_mock_tts(self, tmp_path, mock_tts):
        """Create a TTSEngine with mocked TTS backend."""
        config = TTSConfig(output_dir=tmp_path)
        engine = TTSEngine(config)
        engine._tts = mock_tts
        engine._initialized = True
        return engine

    def test_synthesize_to_mp3_requires_pydub(self, engine_with_mock_tts):
        """Test that missing pydub raises helpful error."""
        with patch.dict('sys.modules', {'pydub': None}):
            with patch('builtins.__import__', side_effect=ImportError("No module")):
                with pytest.raises(TTSEngineError) as exc_info:
                    engine_with_mock_tts.synthesize_to_mp3("Hello")
                
                assert "pydub" in str(exc_info.value).lower()

    def test_synthesize_to_mp3_creates_mp3_file(self, engine_with_mock_tts, tmp_path):
        """Test that MP3 file is created."""
        def create_dummy_wav(text, file_path, **kwargs):
            Path(file_path).write_bytes(b"RIFF" + b"\x00" * 100)
        
        engine_with_mock_tts._tts.tts_to_file.side_effect = create_dummy_wav
        
        # Create a mock pydub module
        mock_pydub = MagicMock()
        mock_audio_segment = MagicMock()
        mock_pydub.AudioSegment = mock_audio_segment
        
        # Mock the audio segment instance returned by from_wav
        mock_audio_instance = MagicMock()
        mock_audio_segment.from_wav.return_value = mock_audio_instance
        
        def create_mp3(path, format, bitrate):
            Path(path).write_bytes(b"ID3" + b"\x00" * 50)
        mock_audio_instance.export.side_effect = create_mp3
        
        with patch.dict('sys.modules', {'pydub': mock_pydub}):
            result = engine_with_mock_tts.synthesize_to_mp3("Hello", "test_output")
        
        assert result.suffix == ".mp3"
        assert "test_output" in result.name

    def test_synthesize_to_mp3_removes_temp_wav(self, engine_with_mock_tts, tmp_path):
        """Test that temporary WAV file is cleaned up."""
        def create_dummy_wav(text, file_path, **kwargs):
            Path(file_path).write_bytes(b"RIFF" + b"\x00" * 100)
        
        engine_with_mock_tts._tts.tts_to_file.side_effect = create_dummy_wav
        
        # Create a mock pydub module
        mock_pydub = MagicMock()
        mock_audio_segment = MagicMock()
        mock_pydub.AudioSegment = mock_audio_segment
        
        mock_audio_instance = MagicMock()
        mock_audio_segment.from_wav.return_value = mock_audio_instance
        
        def create_mp3(path, format, bitrate):
            Path(path).write_bytes(b"ID3" + b"\x00" * 50)
        mock_audio_instance.export.side_effect = create_mp3
        
        with patch.dict('sys.modules', {'pydub': mock_pydub}):
            engine_with_mock_tts.synthesize_to_mp3("Hello", "test_output")
        
        # WAV should be deleted (check if any remain in tmp_path)
        wav_files = list(tmp_path.glob("*.wav"))
        assert len(wav_files) == 0


# =============================================================================
# Convenience Function Tests
# =============================================================================

class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_generate_audio_creates_engine(self, tmp_path):
        """Test that generate_audio creates and uses a TTSEngine."""
        with patch.object(TTSEngine, 'synthesize_to_mp3') as mock_synth:
            mock_synth.return_value = tmp_path / "output.mp3"
            
            with patch.object(TTSEngine, '_ensure_initialized'):
                result = generate_audio(
                    script="Hello world",
                    output_dir=tmp_path,
                )
            
            mock_synth.assert_called_once()

    def test_generate_audio_passes_parameters(self, tmp_path):
        """Test that generate_audio passes parameters correctly."""
        with patch.object(TTSEngine, 'synthesize_to_mp3') as mock_synth:
            mock_synth.return_value = tmp_path / "custom.mp3"
            
            with patch.object(TTSEngine, '_ensure_initialized'):
                generate_audio(
                    script="Test script",
                    output_dir=tmp_path,
                    output_filename="custom",
                )
            
            mock_synth.assert_called_with("Test script", "custom")

    def test_generate_broadcast_audio_uses_date_filename(self, tmp_path):
        """Test that generate_broadcast_audio creates date-based filename."""
        test_date = datetime(2026, 1, 17, 10, 30, 0)
        
        with patch.object(TTSEngine, 'synthesize_to_mp3') as mock_synth:
            mock_synth.return_value = tmp_path / "broadcast_20260117.mp3"
            
            with patch.object(TTSEngine, '_ensure_initialized'):
                generate_broadcast_audio(
                    script="Test",
                    broadcast_date=test_date,
                    output_dir=tmp_path,
                )
            
            # Check that filename contains date
            call_args = mock_synth.call_args
            assert "20260117" in call_args[0][1]

    def test_generate_broadcast_audio_defaults_to_today(self, tmp_path):
        """Test that generate_broadcast_audio defaults to today's date."""
        with patch.object(TTSEngine, 'synthesize_to_mp3') as mock_synth:
            mock_synth.return_value = tmp_path / "broadcast.mp3"
            
            with patch.object(TTSEngine, '_ensure_initialized'):
                with patch('news_bot.tts_engine.datetime') as mock_dt:
                    mock_now = datetime(2026, 3, 15)
                    mock_dt.now.return_value = mock_now
                    mock_dt.strftime = datetime.strftime
                    
                    generate_broadcast_audio(
                        script="Test",
                        output_dir=tmp_path,
                    )
                
                call_args = mock_synth.call_args
                assert "20260315" in call_args[0][1]


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

class TestEdgeCasesAndErrors:
    """Tests for edge cases and error handling."""

    def test_very_long_text_is_handled(self, tmp_path):
        """Test that very long text doesn't cause issues."""
        config = TTSConfig(output_dir=tmp_path)
        engine = TTSEngine(config)
        
        # Test preprocessing of long text
        long_text = "Markets are moving. " * 1000
        result = engine.preprocess_text(long_text)
        
        assert len(result) > 0
        assert "Markets are moving" in result

    def test_special_characters_in_text(self, tmp_path):
        """Test handling of special characters."""
        config = TTSConfig(output_dir=tmp_path)
        engine = TTSEngine(config)
        
        text = "Apple (AAPL) rose 5% on news & rumors—great stuff!"
        result = engine.preprocess_text(text)
        
        # Should handle without error
        assert "Apple" in result
        assert "percent" in result

    def test_unicode_text(self, tmp_path):
        """Test handling of unicode characters."""
        config = TTSConfig(output_dir=tmp_path)
        engine = TTSEngine(config)
        
        text = "Revenue: €500M • Growth: ↑15%"
        result = engine.preprocess_text(text)
        
        # Should handle unicode
        assert "euros" in result or "500" in result

    def test_empty_config_output_dir_uses_default(self):
        """Test that empty output_dir uses default."""
        config = TTSConfig()
        assert config.output_dir == Path("audio_output")

    def test_multiple_abbreviations_in_same_sentence(self, tmp_path):
        """Test handling multiple abbreviations together."""
        config = TTSConfig(output_dir=tmp_path)
        engine = TTSEngine(config)
        
        text = "The CEO announced Q1 EPS of $2.50, up 20% YoY."
        result = engine.preprocess_text(text)
        
        assert "C E O" in result
        assert "first quarter" in result
        assert "E P S" in result
        assert "year over year" in result


# =============================================================================
# Speed Configuration Tests
# =============================================================================

class TestSpeedConfiguration:
    """Tests for TTS speed configuration."""

    def test_default_speed_is_normal(self):
        """Test that default speed is 1.0 (normal)."""
        config = TTSConfig()
        assert config.speed == 1.0

    def test_faster_speed_config(self):
        """Test configuring faster speech."""
        config = TTSConfig(speed=1.3)
        assert config.speed == 1.3

    def test_slower_speed_config(self):
        """Test configuring slower speech."""
        config = TTSConfig(speed=0.8)
        assert config.speed == 0.8

    def test_speed_passed_to_non_xtts_synthesis(self, tmp_path):
        """Test that speed is passed to non-XTTS synthesis."""
        mock_tts = MagicMock()
        config = TTSConfig(
            model_name="tts_models/en/ljspeech/tacotron2-DDC",
            output_dir=tmp_path,
            speed=1.2,
        )
        engine = TTSEngine(config)
        engine._tts = mock_tts
        engine._initialized = True
        
        def create_dummy_wav(text, file_path, **kwargs):
            Path(file_path).write_bytes(b"RIFF" + b"\x00" * 100)
        
        mock_tts.tts_to_file.side_effect = create_dummy_wav
        engine.synthesize("Test")
        
        call_kwargs = mock_tts.tts_to_file.call_args[1]
        assert call_kwargs["speed"] == 1.2


# =============================================================================
# Integration Tests
# =============================================================================

class TestTTSEngineIntegration:
    """Integration-style tests for complete TTS flows."""

    def test_full_synthesis_flow_standard_model(self, tmp_path):
        """Test complete synthesis flow with standard model."""
        mock_tts = MagicMock()
        config = TTSConfig(
            model_name="tts_models/en/ljspeech/tacotron2-DDC",
            output_dir=tmp_path,
        )
        engine = TTSEngine(config)
        engine._tts = mock_tts
        engine._initialized = True
        
        def create_wav(text, file_path, **kwargs):
            Path(file_path).write_bytes(b"RIFF" + b"\x00" * 100)
        
        mock_tts.tts_to_file.side_effect = create_wav
        
        result = engine.synthesize(
            "Good morning Wall Street! The CEO announced Q1 earnings of 5 billion dollars.",
            output_filename="test_broadcast"
        )
        
        assert result.exists()
        assert result.suffix == ".wav"
        
        # Verify preprocessing was applied - check the keyword arg 'text'
        call_kwargs = mock_tts.tts_to_file.call_args[1]
        synthesized_text = call_kwargs.get("text", "")
        assert "C E O" in synthesized_text
        assert "first quarter" in synthesized_text
        assert "5 billion dollars" in synthesized_text

    def test_full_synthesis_flow_xtts_with_chunking(self, tmp_path):
        """Test complete XTTS synthesis flow produces correct chunks."""
        config = TTSConfig(
            model_name="tts_models/multilingual/multi-dataset/xtts_v2",
            output_dir=tmp_path,
            language="en",
            speaker="Claribel Dervla",
        )
        engine = TTSEngine(config)
        
        # Long broadcast script
        long_script = """
        Good morning Wall Street! It's a big day for tech stocks. 
        Nvidia reported earnings that crushed expectations with revenue up 265 percent year over year.
        The CEO announced a new partnership that's sending shares higher.
        Meanwhile Apple unveiled its latest buyback program worth 110 billion dollars.
        Tesla delivered fewer vehicles than expected but the stock is holding steady.
        That's your market update for today folks!
        """
        
        # Test that chunking works correctly for XTTS
        processed = engine.preprocess_text(long_script)
        chunks = engine._chunk_text(processed, max_chars=240)
        
        # Should have multiple chunks
        assert len(chunks) > 1
        
        # Each chunk should be within the limit
        for chunk in chunks:
            assert len(chunk) <= 240
        
        # Content should be preserved
        rejoined = " ".join(chunks)
        assert "Wall Street" in rejoined
        assert "Nvidia" in rejoined
        assert "Tesla" in rejoined

    def test_mp3_conversion_integration(self, tmp_path):
        """Test complete MP3 conversion flow."""
        mock_tts = MagicMock()
        config = TTSConfig(output_dir=tmp_path)
        engine = TTSEngine(config)
        engine._tts = mock_tts
        engine._initialized = True
        
        def create_wav(text, file_path, **kwargs):
            Path(file_path).write_bytes(b"RIFF" + b"\x00" * 100)
        
        mock_tts.tts_to_file.side_effect = create_wav
        
        # Mock pydub for MP3 conversion
        mock_pydub = MagicMock()
        mock_audio = MagicMock()
        mock_pydub.AudioSegment.from_wav.return_value = mock_audio
        
        def create_mp3(path, format, bitrate):
            Path(path).write_bytes(b"ID3" + b"\x00" * 50)
        mock_audio.export.side_effect = create_mp3
        
        with patch.dict('sys.modules', {'pydub': mock_pydub}):
            result = engine.synthesize_to_mp3("Test broadcast", "output")
        
        assert result.suffix == ".mp3"
        assert result.name == "output.mp3"
        # WAV should be cleaned up
        wav_path = tmp_path / "output.wav"
        assert not wav_path.exists()
