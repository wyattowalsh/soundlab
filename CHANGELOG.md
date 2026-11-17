# Changelog

All notable changes to SoundLab will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- Stem files are named from the Demucs model's source order. A float32 stem request is not saved as 24-bit PCM.
- Stereo duration uses the time axis. Reading `StemResult.instrumental` no longer writes a file; vocals-only exposes `no_vocals`.
- Drum MIDI notes are written on channel 9. Basic Pitch `predict` receives the ICASSP model path and a millisecond note length.
- The studio notebook compiles, passes the uploaded song into separation, and installs this package from `packages/soundlab`.
- `analyze_audio` keeps onset times. Loudness uses the declared `pyloudnorm` dependency.
- `SOUNDLAB_SEED` seeds Python, NumPy, and Torch. `force_gpu` reports a missing CUDA device instead of selecting CPU.
- Studio defaults leave Drive, resampling, channel matching, peak normalization, excerpts, silence gating, MIDI quantization, and loudness export off until selected.
- `soundlab tts` calls `TTSGenerator` when the voice extra imports.
- Reverb wet and dry settings reach pedalboard as `wet_level` and `dry_level`. High-pass and low-pass Q is applied by an RBJ biquad inside `EffectsChain`. Asking pedalboard itself for a non-default Q raises, and any other plugin parameter the backend cannot store raises instead of being ignored.
- `soundlab effects` can apply an explicit `--gain-db` and `--highpass-hz`. With neither option, it copies the audio.
- CREPE uses `onset_thresh` as the onset delta and `frame_thresh` as the confidence gate. `DrumTranscriptionConfig` reaches the drum transcriber, including its pitches and velocity scale.
- Key detection on silence or an undefined correlation returns a confidence inside 0–1 instead of failing validation. A Basic Pitch MIDI file that exists but cannot be read raises that read error.
- Stem files follow `SeparationConfig.output_format`. WAV can stay float. FLAC and MP3 reject a float32 request, and MP3 uses `mp3_bitrate`. The studio separation cell no longer passes `mp3_bitrate=None` for WAV.
- Studio export and QA cells read optional results through `globals()`, and mono stem export imports `AudioSegment` at runtime.

### Changed

- Docs name the workspace root `soundlab-workspace` and state that `uv sync` without `--all-packages` does not install the library. Workspace sync is `uv sync --all-packages`, with `--extra notebook` when the notebook extra is required. A local library install that leaves the `all` extra unselected is `uv pip install -e "packages/soundlab[notebook]"`.
- Extras are documented on `packages/soundlab`. Git installs use `#subdirectory=packages/soundlab`.
- Status text matches the tree: the package is Beta, PyPI publication is disabled, the runtime version is `0.1.0` (`soundlab.__version__`), and package metadata is `1.0.0`. The PyPI project named `soundlab` is a different project.
- Quickstart and package README examples use `AudioSegment.duration_seconds`, `save_audio(segment, path)`, `EffectsChain.add` with `process_array` or `process(input_path, output_path)`, `get_device(mode)`, and `TTSGenerator.generate(TTSConfig)`.
- The extension guide points at `EffectConfig.to_plugin()` and `SeparatorBackend.separate(mix_wav, output_dir)`.
- The Colab guide uses the studio notebook controls, including `target_sample_rate`, `normalize_audio`, `normalization_level`, `separation_model`, `segment_length`, and `install_from` values `github_main` and `github_dev`.
- Install docs no longer tell readers to resolve a lockfile. This repository has no `uv.lock`.

The `[1.0.0]` section below is the historical note from 2026-01-09, including its production-status wording. Current status is this Unreleased section.

## [1.0.0] - 2026-01-09

### Added

#### Stem Separation
- **Vocal Isolation**: New `--vocals-only` CLI flag for stem separation that outputs vocals and combined instrumental stems
- **On-demand Instrumental**: `StemResult.instrumental` property computes instrumental mix lazily by summing drums, bass, and other stems
- **Demucs Integration**: Support for multiple Demucs models (`htdemucs`, `htdemucs_ft`, `htdemucs_6s`, `mdx_extra`, `mdx_extra_q`)
- **6-stem Separation**: Extended stem separation with piano and guitar stems via `htdemucs_6s` model

#### Transcription
- **Drum Transcription Backend**: `DrumTranscriber` for onset-based drum-to-MIDI conversion using spectral centroid classification
- **CREPE Transcription Backend**: `CREPETranscriber` for Python 3.12+ melodic transcription with accurate pitch estimation
- **DrumTranscriptionConfig**: Optimized configuration preset for drum transcription with General MIDI drum mapping
- **Backend Auto-Selection**: Transcription backends automatically select best available option based on Python version
- **Piano Roll Visualization**: `render_piano_roll()` function for visualizing transcription results

#### Audio Analysis
- **Tempo Detection**: `detect_tempo()` for BPM estimation with confidence scoring
- **Key Detection**: `detect_key()` for musical key and mode detection
- **Loudness Measurement**: `measure_loudness()` for integrated loudness (LUFS) analysis
- **Spectral Analysis**: `analyze_spectral()` for spectral centroid, bandwidth, and rolloff
- **Onset Detection**: `detect_onsets()` for transient/attack detection with timestamps
- **Combined Analysis**: `analyze_audio()` convenience function for comprehensive analysis

#### Effects Processing
- **Effects Chain**: Composable `EffectsChain` for serial audio processing
- **Dynamics**: Compressor, Limiter, Gate, and Gain effects
- **EQ/Filters**: Highpass and Lowpass filter effects
- **Time-based**: Reverb and Delay effects
- **Creative**: Chorus, Phaser, and Distortion effects
- **Pedalboard Integration**: Built on Spotify's Pedalboard for high-performance DSP

#### Voice Module
- **Text-to-Speech**: `TTSGenerator` with Coqui TTS integration
- **Voice Conversion**: `VoiceConverter` (SVC) interface for voice transformation
- **Configuration Models**: `TTSConfig` and `SVCConfig` for voice processing settings

#### Pipeline Infrastructure
- **Candidate Plans**: Multi-path pipeline execution with `build_candidate_plans()`
- **Quality Assurance**: `QAEvaluator` for automated stem and MIDI quality scoring
- **Checkpointing**: Stage-based checkpoints for resumable processing
- **Post-processing**: `StemPostProcessor` and `MidiPostProcessor` for cleanup operations
- **Run Management**: `init_run()`, `run_paths()`, and artifact tracking

#### I/O and Export
- **Audio I/O**: `load_audio()` and `save_audio()` with format detection
- **MIDI I/O**: `load_midi()` and `save_midi()` for MIDI file operations
- **Multi-format Export**: `export_audio()` supporting WAV, MP3, FLAC, OGG formats
- **Metadata Extraction**: `get_audio_metadata()` for audio file information

#### Utilities
- **Progress Callbacks**: `TqdmProgressCallback` for operation progress tracking
- **Retry Decorators**: `gpu_retry()` and `io_retry()` with configurable backoff
- **Device Management**: `get_device()` for automatic CPU/CUDA selection
- **Structured Logging**: `configure_logging()` with Loguru integration

#### CLI
- **Separation Command**: `soundlab separate` with model and device options
- **Transcription Command**: `soundlab transcribe` with threshold parameters
- **Analysis Command**: `soundlab analyze` with optional JSON output
- **Effects Command**: `soundlab effects` for audio processing

### Changed
- Version bumped to 1.0.0 (production release)
- Development status changed from Beta to Production/Stable
- Minimum Python version set to 3.12

### Fixed
- `StemResult.instrumental` no longer returns None (was stub implementation that returned None)
- Proper array length handling in instrumental stem mixing

### Notes
- Basic Pitch transcription requires Python <3.12 (upstream TensorFlow limitation)
- CREPE backend available for Python 3.12+ users as melodic transcription alternative
- Drum transcription works on all supported Python versions (no TensorFlow dependency)
- Voice and visualization extras install from this repository: `uv pip install -e "packages/soundlab[voice]"` and `uv pip install -e "packages/soundlab[visualization]"`. Do not `pip install soundlab`.

## [0.1.0] - 2026-01-08

### Added
- Initial beta release
- Demucs-based stem separation with 4-stem and 6-stem models
- Basic Pitch MIDI transcription (Python <3.12 only)
- Audio analysis: tempo, key, loudness, spectral features
- Effects chain with dynamics, EQ, and time-based effects
- MIDI I/O with note event handling
- Audio I/O with multi-format support
- CLI interface for core operations
- Pydantic configuration models
- Type hints throughout codebase
