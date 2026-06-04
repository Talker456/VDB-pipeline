import os
import glob
from audio_separator.separator import Separator

class VocalSeparator:
    def __init__(self, model_filename, output_dir, output_format='WAV', lead_model_filename=None):
        self.model_filename = model_filename
        self.lead_model_filename = lead_model_filename
        self.output_dir = output_dir
        self.output_format = output_format
        os.makedirs(self.output_dir, exist_ok=True)
        # Initialize with output_dir
        self.separator = Separator(output_dir=self.output_dir, output_format=self.output_format)

    def separate_files(self, input_dir, enable_lead_separation=False):
        audio_extensions = ["*.mp3", "*.wav", "*.flac", "*.m4a"]
        files_to_process = []
        for ext in audio_extensions:
            files_to_process.extend(glob.glob(os.path.join(input_dir, ext)))
        
        if not files_to_process:
            print(f"❌ No audio files found in '{input_dir}'.")
            return []

        print(f"🎵 Starting separation for {len(files_to_process)} files...")
        
        # Stage 1: Basic Vocal Separation
        self.separator.load_model(self.model_filename)
        
        vocal_files = []
        for file_path in files_to_process:
            print(f"▶ Stage 1 (Vocals): {os.path.basename(file_path)}")
            output_files = self.separator.separate(file_path)
            
            for out_file in output_files:
                if "(Vocals)" in out_file:
                    vocal_files.append(os.path.join(self.output_dir, out_file))
                    
        # Stage 2: Lead/Backing Separation (Optional)
        if enable_lead_separation and self.lead_model_filename:
            print(f"▶ Stage 2 (Lead Vocal Filtering)...")
            self.separator.load_model(self.lead_model_filename)
            lead_files = []
            for vf_path in vocal_files:
                print(f"  - Extracting lead vocal: {os.path.basename(vf_path)}")
                # Separate the vocal stem again
                output_files = self.separator.separate(vf_path)
                
                # In UVR_MDXNET_KARA_2, lead is usually (Vocals) and backing is (Instrumental) 
                # or vice-versa depending on the specific model mapping.
                # However, audio-separator usually labels them based on the model's internal stems.
                # For KARA_2, (Vocals) is typically the lead.
                for out_file in output_files:
                    if "(Vocals)" in out_file:
                        lead_files.append(os.path.join(self.output_dir, out_file))
            
            print("✅ Lead vocal filtering complete!")
            return lead_files

        print("✅ Vocal extraction complete!")
        return vocal_files
