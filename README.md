# Vocal Dataset Builder (VDB-Pipeline)

An automated pipeline designed to extract vocals from music files, slice them according to specific standards, and build a dataset for training.

## 1. Key Features
- **Source Separation**: High-quality vocal separation using the `MelBandRoFormer` model.
- **Audio Slicing**: Automatic slicing based on silence thresholds (retains clips of 2s or longer).
- **Format Normalization**: Standardizes format to 44.1kHz, Mono, PCM-16 WAV.
- **Dataset Splitting**: Automatically distributes files into Train and Validation (fixed at 10 clips) folders.

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
python src/main.py --input_dir "./raw" --output_dir "./dataset" --config "config/default_config.yaml"
```

## 4. Directory Structure
- `src/`: Core source code (Separation, Slicing, Splitting modules).
- `config/`: Pipeline configuration files (`default_config.yaml`).
- `tmp/`: Temporary directory for intermediate processed files (automatically created).

## 5. Configuration (`config/default_config.yaml`)
- `separator`: Model filename and output format settings.
- `slicer`: Silence threshold (dB), minimum length (ms), sampling rate, etc.
- `splitter`: Validation set size and random seed settings.
