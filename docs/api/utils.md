# Utils

Utility functions for GPU management, logging, progress tracking, and retry logic.

## Overview

The `soundlab.utils` module provides essential utilities that support the core functionality of SoundLab. These utilities handle cross-cutting concerns like hardware detection, logging, progress tracking, and error recovery.

## Features

- **GPU Management**: Automatic GPU detection and memory monitoring
- **Logging**: Structured logging with configurable levels
- **Progress Tracking**: Callbacks and tracking for long-running operations
- **Retry Logic**: Automatic retry with exponential backoff
- **Device Selection**: Intelligent CPU/GPU selection with fallback

## Key Components

### GPU Utilities
- `get_device()` - Detect available compute device
- `get_gpu_memory()` - Monitor GPU memory usage
- `clear_gpu_cache()` - Clear GPU memory cache
- `is_gpu_available()` - Check GPU availability

### Logging
- `setup_logging()` - Configure logging system
- `get_logger()` - Get logger instance
- `log_performance()` - Performance metric logging

### Progress Tracking
- `ProgressCallback` - Callback protocol for progress updates
- `ProgressTracker` - Track operation progress
- `ConsoleProgressBar` - Console-based progress bar

### Retry Logic
- `retry_with_backoff()` - Decorator for automatic retry
- `RetryConfig` - Retry configuration options

## Usage Examples

### GPU Management

```python
from soundlab.utils import get_device, get_gpu_memory, is_gpu_available

# Check GPU availability
if is_gpu_available():
    print("GPU is available!")
else:
    print("No GPU detected, using CPU")

# Get device
device = get_device()
print(f"Using device: {device}")  # 'cuda' or 'cpu'

# Monitor GPU memory
if device == 'cuda':
    memory = get_gpu_memory()
    print(f"GPU Memory:")
    print(f"  Total: {memory.total_gb:.2f} GB")
    print(f"  Used: {memory.used_gb:.2f} GB")
    print(f"  Free: {memory.free_gb:.2f} GB")
    print(f"  Usage: {memory.usage_percent:.1f}%")
```

### GPU Memory Management

```python
from soundlab.utils import get_gpu_memory, clear_gpu_cache

def check_memory_and_clear():
    """Monitor and manage GPU memory."""
    memory = get_gpu_memory()

    # Check if running low on memory
    if memory.usage_percent > 90:
        print(f"⚠️  High GPU memory usage: {memory.usage_percent:.1f}%")
        print("Clearing GPU cache...")
        clear_gpu_cache()

        # Check again
        memory = get_gpu_memory()
        print(f"After clearing: {memory.usage_percent:.1f}%")

# Use in processing loop
for i, audio_file in enumerate(audio_files):
    process_audio(audio_file)

    # Clear cache every 10 files
    if i % 10 == 0:
        check_memory_and_clear()
```

### Device Selection

```python
from soundlab.utils import get_device
from soundlab.core import SoundLabConfig

# Force CPU mode
config = SoundLabConfig(device="cpu")
device = get_device(config)
print(f"Forced device: {device}")  # Always 'cpu'

# Automatic selection (default)
config = SoundLabConfig(device="auto")
device = get_device(config)
print(f"Auto-selected: {device}")  # 'cuda' if available, else 'cpu'

# Force GPU (raises error if unavailable)
try:
    config = SoundLabConfig(device="cuda")
    device = get_device(config)
except RuntimeError as e:
    print(f"GPU not available: {e}")
    # Fallback to CPU
    config.device = "cpu"
    device = get_device(config)
```

### Logging Setup

```python
from soundlab.utils import setup_logging, get_logger

# Setup logging
setup_logging(
    level="INFO",           # DEBUG, INFO, WARNING, ERROR, CRITICAL
    log_file="soundlab.log", # Optional log file
    format="detailed"       # 'simple' or 'detailed'
)

# Get logger
logger = get_logger(__name__)

# Use logger
logger.info("Starting audio processing")
logger.debug(f"Processing file: {filename}")
logger.warning("GPU memory is running low")
logger.error(f"Failed to load model: {error}")
```

### Custom Logging

```python
from soundlab.utils import get_logger
import time

logger = get_logger(__name__)

def process_with_logging(audio_path):
    """Process audio with detailed logging."""
    logger.info(f"Processing: {audio_path}")
    start_time = time.time()

    try:
        # Processing steps
        logger.debug("Loading audio...")
        audio = load_audio(audio_path)

        logger.debug("Applying effects...")
        processed = apply_effects(audio)

        logger.debug("Saving output...")
        save_audio("output.wav", processed)

        elapsed = time.time() - start_time
        logger.info(f"Completed in {elapsed:.1f}s")

    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        raise
```

### Progress Tracking

```python
from soundlab.utils import ProgressCallback

def my_progress_callback(progress: float, message: str):
    """Custom progress callback."""
    percentage = progress * 100
    print(f"[{percentage:5.1f}%] {message}")

# Use with operations that support progress callbacks
from soundlab.separation import StemSeparator

separator = StemSeparator()
result = separator.separate(
    "song.mp3",
    "output/",
    progress_callback=my_progress_callback
)
```

### Progress Bar

```python
from soundlab.utils import ConsoleProgressBar

# Create progress bar
progress_bar = ConsoleProgressBar(
    total=100,
    description="Processing",
    show_percentage=True,
    show_time=True
)

# Update progress
for i in range(100):
    # Do work
    time.sleep(0.1)

    # Update
    progress_bar.update(i + 1, message=f"Step {i+1}")

progress_bar.complete()
```

### Progress Tracker

```python
from soundlab.utils import ProgressTracker

# Track multi-stage operation
tracker = ProgressTracker(stages=[
    ("Loading", 0.1),      # 10% of total
    ("Processing", 0.7),   # 70% of total
    ("Saving", 0.2)        # 20% of total
])

# Stage 1: Loading
tracker.start_stage("Loading")
for i in range(10):
    # Load chunk
    tracker.update_stage_progress(i / 10)
    print(f"Overall progress: {tracker.overall_progress:.1%}")

# Stage 2: Processing
tracker.start_stage("Processing")
for i in range(100):
    # Process
    tracker.update_stage_progress(i / 100)

# Stage 3: Saving
tracker.start_stage("Saving")
# Save...
tracker.update_stage_progress(1.0)

tracker.complete()
```

### Retry Logic

```python
from soundlab.utils import retry_with_backoff, RetryConfig

# Use as decorator
@retry_with_backoff(
    max_attempts=3,
    initial_delay=1.0,
    max_delay=10.0,
    exponential_base=2.0
)
def download_model(model_url):
    """Download with automatic retry on failure."""
    response = requests.get(model_url)
    response.raise_for_status()
    return response.content

# Use with config
retry_config = RetryConfig(
    max_attempts=5,
    initial_delay=0.5,
    max_delay=30.0,
    exponential_base=2.0,
    retry_on_exceptions=[ConnectionError, TimeoutError]
)

@retry_with_backoff(config=retry_config)
def api_request(endpoint):
    """API request with retry."""
    return make_request(endpoint)
```

### Custom Retry Handler

```python
from soundlab.utils import retry_with_backoff
from soundlab.core import ModelLoadError
import time

def on_retry(attempt, delay, exception):
    """Called on each retry attempt."""
    print(f"Attempt {attempt} failed: {exception}")
    print(f"Retrying in {delay:.1f}s...")

@retry_with_backoff(
    max_attempts=3,
    initial_delay=1.0,
    on_retry=on_retry,
    retry_on_exceptions=[ModelLoadError, ConnectionError]
)
def load_model_with_retry(model_path):
    """Load model with retry logic."""
    return load_model(model_path)

# Use
try:
    model = load_model_with_retry("model.pth")
except Exception as e:
    print(f"Failed after all retries: {e}")
```

### Performance Logging

```python
from soundlab.utils import log_performance, get_logger
import time

logger = get_logger(__name__)

@log_performance(logger)
def process_audio(audio_path):
    """Function with automatic performance logging."""
    # Processing...
    time.sleep(2)
    return "result"

# Logs: "process_audio completed in 2.00s"
result = process_audio("song.wav")
```

### Resource Monitoring

```python
from soundlab.utils import get_gpu_memory, get_logger
import psutil

logger = get_logger(__name__)

def log_system_resources():
    """Log system resource usage."""
    # CPU
    cpu_percent = psutil.cpu_percent(interval=1)
    logger.info(f"CPU usage: {cpu_percent}%")

    # RAM
    ram = psutil.virtual_memory()
    logger.info(f"RAM usage: {ram.percent}% ({ram.used / 1e9:.1f}GB / {ram.total / 1e9:.1f}GB)")

    # GPU
    try:
        gpu_memory = get_gpu_memory()
        logger.info(f"GPU memory: {gpu_memory.usage_percent:.1f}% ({gpu_memory.used_gb:.1f}GB / {gpu_memory.total_gb:.1f}GB)")
    except:
        logger.info("GPU not available")

# Use in processing pipeline
log_system_resources()
process_large_file("audio.wav")
log_system_resources()
```

### Batch Processing with Progress

```python
from soundlab.utils import ProgressTracker, get_logger
from pathlib import Path

logger = get_logger(__name__)

def batch_process_with_progress(input_dir, output_dir):
    """Process multiple files with progress tracking."""
    files = list(Path(input_dir).glob("*.wav"))
    tracker = ProgressTracker(total_steps=len(files))

    logger.info(f"Processing {len(files)} files...")

    for i, file in enumerate(files):
        logger.debug(f"Processing {file.name}...")

        # Process file
        process_file(file, output_dir)

        # Update progress
        tracker.update(i + 1, message=f"Processed {file.name}")
        logger.info(f"Progress: {tracker.overall_progress:.1%}")

    tracker.complete()
    logger.info("Batch processing complete!")

# Use
batch_process_with_progress("input/", "output/")
```

### Context Manager for GPU

```python
from soundlab.utils import clear_gpu_cache, get_gpu_memory
from contextlib import contextmanager

@contextmanager
def gpu_memory_manager():
    """Context manager for GPU memory management."""
    # Before
    initial_memory = get_gpu_memory()
    print(f"Initial GPU memory: {initial_memory.used_gb:.2f}GB")

    try:
        yield
    finally:
        # After
        clear_gpu_cache()
        final_memory = get_gpu_memory()
        print(f"Final GPU memory: {final_memory.used_gb:.2f}GB")
        print(f"Memory freed: {initial_memory.used_gb - final_memory.used_gb:.2f}GB")

# Use
with gpu_memory_manager():
    # Process heavy operation
    result = process_large_audio("huge_file.wav")
```

## Configuration

### Logging Levels
- **DEBUG**: Detailed diagnostic information
- **INFO**: General informational messages
- **WARNING**: Warning messages for non-critical issues
- **ERROR**: Error messages for serious problems
- **CRITICAL**: Critical errors that may cause failure

### Retry Configuration
- **max_attempts**: Maximum retry attempts (default: 3)
- **initial_delay**: Initial delay in seconds (default: 1.0)
- **max_delay**: Maximum delay in seconds (default: 60.0)
- **exponential_base**: Base for exponential backoff (default: 2.0)
- **retry_on_exceptions**: List of exception types to retry

### GPU Memory
- Automatic detection of CUDA availability
- Memory usage tracking in GB and percentage
- Cache clearing for memory management

## Best Practices

### GPU Usage
1. Always check GPU availability before using
2. Monitor memory usage for long-running operations
3. Clear cache between processing batches
4. Use CPU fallback for compatibility

### Logging
1. Use appropriate log levels (DEBUG for development, INFO for production)
2. Include context in log messages
3. Use structured logging for better parsing
4. Rotate log files to prevent disk space issues

### Progress Tracking
1. Provide meaningful progress messages
2. Update progress at reasonable intervals
3. Include time estimates when possible
4. Handle errors gracefully in callbacks

### Retry Logic
1. Use exponential backoff for network requests
2. Limit retry attempts to prevent infinite loops
3. Log retry attempts for debugging
4. Specify which exceptions should trigger retries

## API Reference

::: soundlab.utils
    options:
      show_source: true
      members: true
      heading_level: 3
      show_root_heading: true
      show_root_toc_entry: true
      show_bases: true
      show_inheritance_diagram: false
      group_by_category: true
      members_order: source
