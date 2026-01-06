# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

### Changed

### Deprecated

### Removed

### Fixed

### Security

## [0.1.0] - 2026-01-06

### Added

#### Core Features
- **Stem Separation**: Audio source separation using Facebook Research's Demucs models
  - htdemucs: 4-stem separation (vocals, drums, bass, other)
  - htdemucs_ft: Fine-tuned variant for improved quality
  - htdemucs_6s: 6-stem separation (vocals, drums, bass, guitar, piano, other)
  - Automatic model downloading and caching

- **Audio-to-MIDI Transcription**: Polyphonic pitch detection with Spotify's Basic Pitch
  - Convert audio to MIDI files with note onset detection
  - Support for multiple instruments and polyphonic content
  - Configurable sensitivity and pitch thresholds

- **Audio Analysis**: Comprehensive audio feature extraction
  - BPM (tempo) detection with beat tracking
  - Musical key detection (pitch class profile analysis)
  - Loudness analysis (RMS, peak, LUFS-style measurements)
  - Spectral analysis (centroid, rolloff, bandwidth, flatness)
  - Onset detection for rhythm and transient analysis

- **Effects Chain**: Professional audio processing tools
  - **Dynamics**: Compressor, limiter, gate, expander
  - **Equalization**: Parametric EQ with multiple filter types
  - **Time-based**: Reverb, delay, chorus, flanger
  - Chain multiple effects with automatic parameter validation

- **Voice Generation** (Optional): Text-to-speech and voice conversion
  - TTS integration for speech synthesis
  - So-VITS-SVC support for voice style conversion
  - Optional dependency for flexibility

#### Infrastructure
- **GPU Acceleration**: Automatic CUDA detection with CPU fallback
  - Seamless GPU/CPU switching based on availability
  - Optimized performance for both platforms

- **CLI Interface**: Command-line tool for batch processing
  - Process multiple files with glob patterns
  - Configurable output formats and quality settings
  - Progress tracking and error handling

- **Gradio Notebook Interface**: Interactive web UI for experimentation
  - User-friendly interface for all features
  - Real-time audio preview and playback
  - Jupyter/Colab notebook integration

#### Quality Assurance
- **Comprehensive Test Suite**: Full test coverage for all modules
  - Unit tests for core functionality
  - Integration tests for end-to-end workflows
  - Mock-based testing for external dependencies

- **Documentation**: Complete project documentation
  - API reference for all modules
  - Usage examples and tutorials
  - Contributing guidelines and development setup
