# Vocal Dataset Builder (VDB-Pipeline)

An automated pipeline designed to extract vocals from music files, perform advanced preprocessing, and build high-quality datasets for SVC/AI training.

## 1. Key Features
- **Source Separation**: High-quality lead vocal extraction using **Ensemble Presets** (e.g., `karaoke` with `avg_wave` algorithm).
- **Dereverberation**: Optional final stage to remove reverb and echo, ensuring "Dry" vocals for optimal training.
- **Audio Slicing**: Automatic slicing based on silence thresholds (retains clips of 2s to 10s).
- **Format Normalization**: Standardizes format to **44.1kHz, Mono, PCM-16 WAV**.
- **Dataset Splitting**: Automatically distributes files into Train and Validation (exactly 10 clips) folders.

## 2. Anaconda Environment Setup

```bash
# 1. Create and activate a new virtual environment
conda create -n vdb_env python=3.10 -y
conda activate vdb_env

# 2. Install FFmpeg (required for audio decoding)
conda install -c conda ffmpeg -y

# 3. Install project dependencies
pip install -r requirements.txt
```

## 3. Usage
### Execution
```bash
python src/main.py --input_dir "./path/to/raw_songs" --output_dir "./path/to/output_dataset"
```

## 4. Directory Structure
- `src/`: Core source code (Separation, Dereverb, Slicing, Splitting modules).
- `config/`: Pipeline configuration files (`default_config.yaml`).
- `tmp/`: Temporary directory for intermediate processed files (automatically created).

## 5. Configuration (`config/default_config.yaml`)
### `separator`
- `use_preset`: Toggle between Ensemble Presets (True) and Manual Model selection (False).
- `ensemble_preset`: Chosen preset (e.g., `karaoke` for lead vocals).
- `enable_dereverb`: Toggle the final Dereverberation stage.
- `dereverb_model_filename`: Model used for removing reverb.

### `slicer`
- `threshold`: Silence detection threshold (dB).
- `min_length`: Minimum clip length in milliseconds (default: 2000ms).
- `sr`: Target sampling rate (44100Hz recommended).

### `splitter`
- `val_size`: Number of clips for the Validation set (fixed at 10).
- `seed`: Random seed for deterministic dataset splitting.
