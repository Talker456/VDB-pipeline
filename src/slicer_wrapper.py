import os
import librosa
import soundfile as sf
# Assuming we run from project root and 'src' is a package
try:
    from src.slicer import Slicer
except ImportError:
    from slicer import Slicer

class AudioSlicerWrapper:
    def __init__(self, sr=44100, threshold=-40, min_length=2000, min_interval=300, hop_size=10, mono=True):
        self.sr = sr
        self.threshold = threshold
        self.min_length = min_length
        self.min_interval = min_interval
        self.hop_size = hop_size
        self.mono = mono
        self.slicer = Slicer(
            sr=self.sr,
            threshold=self.threshold,
            min_length=self.min_length,
            min_interval=self.min_interval,
            hop_size=self.hop_size
        )

    def slice_files(self, vocal_files, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        if not vocal_files:
            print("❌ No vocal files to slice.")
            return 0

        print(f"✂️ Slicing {len(vocal_files)} vocal files and converting to {self.sr}Hz...")
        clip_count = 0
        
        # Calculate min samples based on min_length (ms)
        min_samples = int(self.sr * (self.min_length / 1000.0))

        for vf_path in vocal_files:
            # Load and resample
            audio, _ = librosa.load(vf_path, sr=self.sr, mono=self.mono)
            
            # Slice
            chunks = self.slicer.slice(audio)
            base_name = os.path.splitext(os.path.basename(vf_path))[0].replace(" ", "_")

            for i, chunk in enumerate(chunks):
                if len(chunk) >= min_samples:
                    out_name = f"{base_name}_clip_{i:04d}.wav"
                    out_path = os.path.join(output_dir, out_name)
                    sf.write(out_path, chunk, self.sr, subtype='PCM_16')
                    clip_count += 1
        
        print(f"✅ Slicing complete! Generated {clip_count} high-quality clips in '{output_dir}'.")
        return clip_count
