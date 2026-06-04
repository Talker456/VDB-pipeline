import os
import glob
from audio_separator.separator import Separator

class VocalSeparator:
    def __init__(self, output_dir, config):
        self.output_dir = os.path.abspath(output_dir)
        self.config = config
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize Separator with either preset or manual settings
        if self.config.get('use_preset'):
            self.separator = Separator(
                output_dir=self.output_dir,
                ensemble_preset=self.config.get('ensemble_preset'),
                ensemble_algorithm=self.config.get('ensemble_algorithm'),
                output_format=self.config.get('output_format', 'WAV')
            )
        else:
            self.separator = Separator(
                output_dir=self.output_dir, 
                output_format=self.config.get('output_format', 'WAV')
            )

    def separate_files(self, input_dir):
        audio_extensions = ["*.mp3", "*.wav", "*.flac", "*.m4a"]
        files_to_process = []
        for ext in audio_extensions:
            # Convert input file paths to absolute paths
            for f in glob.glob(os.path.join(input_dir, ext)):
                files_to_process.append(os.path.abspath(f))
        
        if not files_to_process:
            print(f"❌ No audio files found in '{input_dir}'.")
            return []

        print(f"🎵 Starting separation for {len(files_to_process)} files...")
        
        initial_results = []

        if self.config.get('use_preset'):
            preset_name = self.config.get('ensemble_preset')
            print(f"▶ Using Ensemble Preset: {preset_name}")
            self.separator.load_model()
            
            for file_path in files_to_process:
                print(f"▶ Separating (Preset): {os.path.basename(file_path)}")
                output_files = self.separator.separate(file_path)
                
                for out_file in output_files:
                    if "(Lead Vocals)" in out_file or "(Vocals)" in out_file:
                        # Join with absolute output_dir to ensure valid path
                        initial_results.append(os.path.join(self.output_dir, out_file))
            
            print(f"✅ Ensemble separation complete!")
        
        else:
            # Stage 1: Basic Vocal Separation (Manual Mode)
            self.separator.load_model(self.config.get('model_filename'))
            
            vocal_files = []
            for file_path in files_to_process:
                print(f"▶ Stage 1 (Vocals): {os.path.basename(file_path)}")
                output_files = self.separator.separate(file_path)
                
                for out_file in output_files:
                    if "(Vocals)" in out_file:
                        vocal_files.append(os.path.join(self.output_dir, out_file))
                        
            # Stage 2: Lead/Backing Separation (Optional)
            enable_lead = self.config.get('enable_lead_separation', False)
            lead_model = self.config.get('lead_model_filename')
            
            if enable_lead and lead_model:
                print(f"▶ Stage 2 (Lead Vocal Filtering)...")
                self.separator.load_model(lead_model)
                for vf_path in vocal_files:
                    print(f"  - Extracting lead vocal: {os.path.basename(vf_path)}")
                    output_files = self.separator.separate(vf_path)
                    for out_file in output_files:
                        if "(Vocals)" in out_file or "(Lead Vocals)" in out_file:
                            initial_results.append(os.path.join(self.output_dir, out_file))
                
                print("✅ Lead vocal filtering complete!")
            else:
                initial_results = vocal_files
                print("✅ Vocal extraction complete!")

        # Final Stage: Dereverberation (Optional)
        if self.config.get('enable_dereverb') and self.config.get('dereverb_model_filename'):
            print(f"▶ Final Stage: Dereverberation (Removing Reverb)...")
            self.separator.load_model(self.config.get('dereverb_model_filename'))
            dry_results = []
            for res_path in initial_results:
                print(f"  - Removing reverb from: {os.path.basename(res_path)}")
                # Ensure input path to separate is absolute
                output_files = self.separator.separate(os.path.abspath(res_path))
                
                for out_file in output_files:
                    dry_results.append(os.path.join(self.output_dir, out_file))
            
            print("✅ Dereverberation complete!")
            return dry_results

        return initial_results
