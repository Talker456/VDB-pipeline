import os
import argparse
import yaml
import sys
import shutil

# Add current directory to sys.path
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

def run_experiment():
    parser = argparse.ArgumentParser(description="Synchronized VDB-Pipeline for Reverb Comparison Experiment")
    parser.add_argument("--input_dir", required=True, help="Directory containing raw song files")
    parser.add_argument("--output_base", required=True, help="Base directory for synchronized datasets")
    parser.add_argument("--config", default="config/default_config.yaml", help="Path to config file")
    parser.add_argument("--temp_dir", default="tmp_exp", help="Temporary directory")
    
    args = parser.parse_args()
    config = load_config(args.config)
    
    # 0. Setup Paths
    vocals_tmp = os.path.join(args.temp_dir, "vocals")
    dry_sliced_tmp = os.path.join(args.temp_dir, "sliced_dry")
    wet_sliced_tmp = os.path.join(args.temp_dir, "sliced_wet")
    
    dry_output = os.path.join(args.output_base, "dry")
    wet_output = os.path.join(args.output_base, "reverb")
    
    # 1. Dual Separation (Get Dry and Wet pairs)
    print("🚀 Step 1: Dual Source Separation...")
    sep_cfg = config['separator']
    # Force dereverb for the master run
    sep_cfg['enable_dereverb'] = True 
    
    separator = VocalSeparator(output_dir=vocals_tmp, config=sep_cfg)
    dry_files, pair_map = separator.separate_files(args.input_dir, return_pairs=True)
    
    if not dry_files:
        print("❌ No vocals extracted.")
        return

    # 2. Master Slicing (Dry Tracks)
    print("\n🚀 Step 2: Master Slicing (using Dry tracks)...")
    sli_cfg = config['slicer']
    slicer = AudioSlicerWrapper(
        sr=sli_cfg['sr'],
        threshold=sli_cfg['threshold'],
        min_length=sli_cfg['min_length'],
        min_interval=sli_cfg['min_interval'],
        hop_size=sli_cfg['hop_size'],
        mono=sli_cfg['mono']
    )
    
    dry_clip_count, metadata = slicer.slice_files(dry_files, dry_sliced_tmp)
    
    # 3. Slave Slicing (Wet Tracks using Dry metadata)
    print("\n🚀 Step 3: Slave Slicing (applying metadata to Reverb tracks)...")
    wet_files = list(pair_map.values())
    wet_clip_count = slicer.apply_metadata_to_files(metadata, wet_files, wet_sliced_tmp, pair_map)
    
    if dry_clip_count != wet_clip_count:
        print(f"⚠️ Warning: Clip count mismatch! Dry: {dry_clip_count}, Wet: {wet_clip_count}")

    # 4. Synchronized Splitting
    print("\n🚀 Step 4: Synchronized Dataset Splitting...")
    spl_cfg = config['splitter']
    splitter = DatasetSplitter(val_size=spl_cfg['val_size'], seed=spl_cfg['seed'])
    
    # Split Dry first
    print("--- Splitting DRY Dataset ---")
    splitter.split(dry_sliced_tmp, os.path.join(dry_output, "train", "audio"), os.path.join(dry_output, "val", "audio"))
    
    # We need to ensure Wet is split EXACTLY like Dry.
    # Since we used the same prefix logic and indices, we can map filenames.
    print("\n--- Syncing REVERB Dataset split with DRY ---")
    
    # Helper to find corresponding wet file for a dry file
    # Dry: SongA_(Vocals)_..._(noreverb)_..._clip_0001.wav
    # Wet: SongA_(Vocals)_..._clip_0001.wav
    
    def sync_split(src_sliced_dir, ref_split_dir, target_split_dir):
        os.makedirs(target_split_dir, exist_ok=True)
        ref_files = os.listdir(ref_split_dir)
        src_files = os.listdir(src_sliced_dir)
        
        # Create a mapping based on the clip ID and a portion of the filename
        # A more robust way: since we know the pair_map and the clip_id
        count = 0
        for ref_f in ref_files:
            # Extract song base and clip id from dry filename
            # e.g., "Song_A_(Vocals)_..._clip_0001.wav"
            # We look for a file in src_files that matches the end "_clip_XXXX.wav" 
            # and shares a significant prefix before the dereverb tags.
            
            # Simple approach: find file in src_files that has same clip index and belongs to the same song
            # We can use the fact that we know which dry_path produced which wet_path
            
            for dry_path, wet_path in pair_map.items():
                dry_base = os.path.splitext(os.path.basename(dry_path))[0].replace(" ", "_")
                wet_base = os.path.splitext(os.path.basename(wet_path))[0].replace(" ", "_")
                
                if ref_f.startswith(dry_base):
                    clip_suffix = ref_f[len(dry_base):] # e.g. "_clip_0001.wav"
                    target_f = wet_base + clip_suffix
                    if target_f in src_files:
                        shutil.copy(os.path.join(src_sliced_dir, target_f), os.path.join(target_split_dir, target_f))
                        count += 1
                        break
        return count

    # Split Wet by mirroring Dry's structure
    for subset in ["train", "audio"], ["val", "audio"]:
        ref_dir = os.path.join(dry_output, *subset)
        target_dir = os.path.join(wet_output, *subset)
        if os.path.exists(target_dir): shutil.rmtree(target_dir)
        
        synced_count = sync_split(wet_sliced_tmp, ref_dir, target_dir)
        print(f" - Synced {synced_count} clips to {target_dir}")

    print("\n✅ Synchronized Experiment Datasets Ready!")
    print(f"📍 Dry Dataset: {dry_output}")
    print(f"📍 Reverb Dataset: {wet_output}")

if __name__ == "__main__":
    run_experiment()
