import unittest
import os
import sys
import numpy as np
import librosa
import soundfile as sf
import shutil

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.slicer import Slicer
from src.slicer_wrapper import AudioSlicerWrapper
from src.separator import VocalSeparator

class TestExperimentSync(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = "test_sync_tmp"
        self.output_dir = "test_sync_out"
        os.makedirs(self.tmp_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create a dummy audio signal (10 seconds, 44.1kHz)
        self.sr = 44100
        t = np.linspace(0, 10, self.sr * 10)
        # Signal with some sound and some silence
        self.signal = np.sin(2 * np.pi * 440 * t)
        self.signal[self.sr*2:self.sr*4] = 0 # 2 seconds of silence
        self.signal[self.sr*6:self.sr*8] = 0 # another 2 seconds of silence
        
        self.test_audio_path = os.path.join(self.tmp_dir, "test_audio.wav")
        sf.write(self.test_audio_path, self.signal, self.sr)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)
        shutil.rmtree(self.output_dir)

    def test_slicer_indices_return(self):
        """Check if Slicer.slice returns correct indices mapping to the audio length."""
        slicer = Slicer(sr=self.sr, threshold=-40, min_length=500, min_interval=100, hop_size=10)
        chunks, indices = slicer.slice(self.signal)
        
        self.assertEqual(len(chunks), len(indices))
        for i, (start, end) in enumerate(indices):
            # Verify the length of the chunk matches the index range
            self.assertEqual(len(chunks[i]), end - start)
            # Basic sanity check: indices should be within signal range
            self.assertGreaterEqual(start, 0)
            self.assertLessEqual(end, len(self.signal))

    def test_synchronized_slicing_workflow(self):
        """Verify that applying master metadata to a second file yields identical segments."""
        wrapper = AudioSlicerWrapper(sr=self.sr, threshold=-40, min_length=500, min_interval=100)
        
        # 1. Master Slicing (Dry)
        dry_out = os.path.join(self.output_dir, "dry")
        count, metadata = wrapper.slice_files([self.test_audio_path], dry_out)
        
        self.assertGreater(count, 0)
        self.assertIn(self.test_audio_path, metadata)
        
        # 2. Slave Slicing (Wet)
        wet_out = os.path.join(self.output_dir, "wet")
        # In this test, we use the same file as 'wet' to check for 100% identity
        pair_map = {self.test_audio_path: self.test_audio_path}
        
        slave_count = wrapper.apply_metadata_to_files(metadata, [self.test_audio_path], wet_out, pair_map)
        
        self.assertEqual(count, slave_count)
        
        # 3. Verify files are identical in length and count
        dry_files = sorted(os.listdir(dry_out))
        wet_files = sorted(os.listdir(wet_out))
        
        self.assertEqual(len(dry_files), len(wet_files))
        
        for d_f, w_f in zip(dry_files, wet_files):
            d_audio, _ = librosa.load(os.path.join(dry_out, d_f), sr=self.sr)
            w_audio, _ = librosa.load(os.path.join(wet_out, w_f), sr=self.sr)
            self.assertEqual(len(d_audio), len(w_audio))
            np.testing.assert_array_almost_equal(d_audio, w_audio)

    def test_separator_pair_mapping(self):
        """Check if separator returns the correct dry-wet mapping when return_pairs=True."""
        config = {
            'use_preset': False,
            'model_filename': 'dummy.ckpt',
            'enable_dereverb': True,
            'dereverb_model_filename': 'dereverb.ckpt'
        }
        
        # We need to mock the Separator class inside separator.py
        from unittest.mock import patch, MagicMock
        
        with patch('src.separator.Separator') as MockSep:
            mock_inst = MockSep.return_value
            # Use a side_effect for glob to return the file only for .wav extension
            def side_effect_glob(pattern):
                if pattern.endswith('*.wav'):
                    return ['song1.wav']
                return []
                
            with patch('glob.glob', side_effect=side_effect_glob):
                # Mock separate calls: 1. Vocal separation, 2. Dereverb
                mock_inst.separate.side_effect = [
                    ['song1_(Vocals).wav'],
                    ['song1_(Vocals)_(No Reverb).wav', 'song1_(Vocals)_(Reverb).wav']
                ]
                
                vsep = VocalSeparator(self.output_dir, config)
                dry_files, pair_map = vsep.separate_files("dummy_input", return_pairs=True)
                
                # Check results
                self.assertEqual(len(dry_files), 1)
                dry_path = os.path.join(os.path.abspath(self.output_dir), 'song1_(Vocals)_(No Reverb).wav')
                wet_path = os.path.join(os.path.abspath(self.output_dir), 'song1_(Vocals).wav')
                
                self.assertIn(dry_path, dry_files)
                self.assertEqual(pair_map[dry_path], wet_path)

if __name__ == '__main__':
    unittest.main()
