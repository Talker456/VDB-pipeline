import os
import argparse
import yaml
import sys

# Add current directory to sys.path to allow imports when running as a script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.separator import VocalSeparator
    from src.slicer_wrapper import AudioSlicerWrapper
    from src.splitter import DatasetSplitter
except ImportError:
    from separator import VocalSeparator
    from slicer_wrapper import AudioSlicerWrapper
    from splitter import DatasetSplitter

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Vocal Dataset Builder Pipeline")
    parser.add_argument("--input_dir", required=True, help="Directory containing raw song files")
    parser.add_argument("--output_dir", required=True, help="Directory to save final dataset")
    parser.add_argument("--config", default="config/default_config.yaml", help="Path to config file")
    parser.add_argument("--temp_dir", default="tmp", help="Temporary directory for intermediate files")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"❌ Config file not found: {args.config}")
        return

    config = load_config(args.config)
    
    # Setup paths
    vocals_tmp = os.path.join(args.temp_dir, "vocals")
    sliced_tmp = os.path.join(args.temp_dir, "sliced")
    # Output structure as per requirement: train/audio/ and val/audio/
    train_out = os.path.join(args.output_dir, "train", "audio")
    val_out = os.path.join(args.output_dir, "val", "audio")
    
    print("🚀 Starting VDB-Pipeline...")
    
    # 1. Source Separation
    sep_cfg = config['separator']
    separator = VocalSeparator(
        output_dir=vocals_tmp,
        config=sep_cfg
    )
    vocal_files = separator.separate_files(args.input_dir)
    
    if not vocal_files:
        print("Stopping pipeline: No vocal files generated.")
        return

    # 2. Slicing & Normalization
    sli_cfg = config['slicer']
    slicer = AudioSlicerWrapper(
        sr=sli_cfg['sr'],
        threshold=sli_cfg['threshold'],
        min_length=sli_cfg['min_length'],
        min_interval=sli_cfg['min_interval'],
        hop_size=sli_cfg['hop_size'],
        mono=sli_cfg['mono']
    )
    clip_count, _ = slicer.slice_files(vocal_files, sliced_tmp)
    
    if clip_count == 0:
        print("Stopping pipeline: No audio clips generated.")
        return

    # 3. Dataset Stratification
    spl_cfg = config['splitter']
    splitter = DatasetSplitter(
        val_size=spl_cfg['val_size'],
        seed=spl_cfg['seed']
    )
    splitter.split(sliced_tmp, train_out, val_out)
    
    print("\n✅ All stages completed successfully!")

if __name__ == "__main__":
    main()
