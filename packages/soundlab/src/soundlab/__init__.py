"""
SoundLab - Production-ready music processing library.

Features:
- Stem separation using Demucs
- Audio-to-MIDI transcription using Basic Pitch
- Audio effects processing using Pedalboard
- Comprehensive audio analysis
- Voice generation (TTS and SVC)

Example
-------
>>> import soundlab
>>>
>>> # Separate stems
>>> from soundlab.separation import StemSeparator
>>> separator = StemSeparator()
>>> result = separator.separate("song.mp3", "output/")
>>>
>>> # Analyze audio
>>> from soundlab.analysis import analyze_audio
>>> analysis = analyze_audio("song.mp3")
>>> print(analysis.summary)
"""

from soundlab._version import __version__, __version_info__

# Core exports
from soundlab.core import (
    AudioFormat,
    AudioMetadata,
    AudioSegment,
    SoundLabConfig,
    SoundLabError,
    get_config,
)

# I/O exports
from soundlab.io import (
    load_audio,
    save_audio,
    get_audio_metadata,
    load_midi,
    save_midi,
)

# Separation exports
from soundlab.separation import (
    StemSeparator,
    SeparationConfig,
    DemucsModel,
)

# Transcription exports
from soundlab.transcription import (
    MIDITranscriber,
    TranscriptionConfig,
)

# Analysis exports
from soundlab.analysis import (
    analyze_audio,
    detect_tempo,
    detect_key,
    measure_loudness,
)

# Effects exports
from soundlab.effects import (
    EffectsChain,
    CompressorConfig,
    ReverbConfig,
)

__all__ = [
    # Version
    "__version__",
    "__version_info__",
    # Core
    "AudioFormat",
    "AudioMetadata",
    "AudioSegment",
    "SoundLabConfig",
    "SoundLabError",
    "get_config",
    # I/O
    "load_audio",
    "save_audio",
    "get_audio_metadata",
    "load_midi",
    "save_midi",
    # Separation
    "StemSeparator",
    "SeparationConfig",
    "DemucsModel",
    # Transcription
    "MIDITranscriber",
    "TranscriptionConfig",
    # Analysis
    "analyze_audio",
    "detect_tempo",
    "detect_key",
    "measure_loudness",
    # Effects
    "EffectsChain",
    "CompressorConfig",
    "ReverbConfig",
]
