import unittest
from unittest.mock import MagicMock, patch
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.separator import VocalSeparator

class TestVocalSeparator(unittest.TestCase):
    def setUp(self):
        self.output_dir = "test_output"
        # Ensure output dir doesn't actually need to exist for mock
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def tearDown(self):
        if os.path.exists(self.output_dir):
            import shutil
            shutil.rmtree(self.output_dir)

    @patch('src.separator.Separator')
    def test_preset_mode_initialization(self, MockSeparator):
        """Verify that Preset Mode correctly configures the Separator."""
        config = {
            'use_preset': True,
            'ensemble_preset': 'karaoke',
            'ensemble_algorithm': 'avg_wave',
            'output_format': 'WAV'
        }
        
        VocalSeparator(self.output_dir, config)
        
        # Check if Separator was initialized with absolute path
        MockSeparator.assert_called_with(
            output_dir=os.path.abspath(self.output_dir),
            ensemble_preset='karaoke',
            ensemble_algorithm='avg_wave',
            output_format='WAV'
        )

    @patch('src.separator.Separator')
    def test_manual_mode_initialization(self, MockSeparator):
        """Verify that Manual Mode correctly configures the Separator without presets."""
        config = {
            'use_preset': False,
            'model_filename': 'test_model.ckpt',
            'output_format': 'FLAC'
        }
        
        VocalSeparator(self.output_dir, config)
        
        # Check if Separator was initialized with absolute path
        MockSeparator.assert_called_with(
            output_dir=os.path.abspath(self.output_dir),
            output_format='FLAC'
        )

    @patch('src.separator.Separator')
    def test_manual_mode_separation_flow(self, MockSeparator):
        """Verify that Manual Mode still follows the 2-stage separation logic if enabled."""
        config = {
            'use_preset': False,
            'model_filename': 'stage1_model.ckpt',
            'enable_lead_separation': True,
            'lead_model_filename': 'stage2_model.ckpt'
        }
        
        # Setup mock behavior
        mock_sep_instance = MockSeparator.return_value
        
        # Mock glob to return one file only when searching for .mp3
        def side_effect_glob(pattern):
            if pattern.endswith('*.mp3'):
                return ['input.mp3']
            return []

        with patch('glob.glob', side_effect=side_effect_glob):
            # Mock separate to return a list of files
            mock_sep_instance.separate.side_effect = [
                ['input_(Vocals).wav'], # Stage 1 output
                ['input_(Vocals)_(Lead Vocals).wav'] # Stage 2 output
            ]
            
            vocal_sep = VocalSeparator(self.output_dir, config)
            results = vocal_sep.separate_files('some_input_dir')
            
            # Verify separate was called twice with absolute path
            abs_input = os.path.abspath('input.mp3')
            self.assertEqual(mock_sep_instance.separate.call_count, 2)
            mock_sep_instance.separate.assert_any_call(abs_input)
            
            # Verify final result contains the filtered file with absolute path
            abs_output_dir = os.path.abspath(self.output_dir)
            expected_file = os.path.join(abs_output_dir, 'input_(Vocals)_(Lead Vocals).wav')
            self.assertIn(expected_file, results)

    @patch('src.separator.Separator')
    def test_preset_mode_separation_flow(self, MockSeparator):
        """Verify that Preset Mode only runs a single stage of separation."""
        config = {
            'use_preset': True,
            'ensemble_preset': 'karaoke'
        }
        
        mock_sep_instance = MockSeparator.return_value
        
        def side_effect_glob(pattern):
            if pattern.endswith('*.mp3'):
                return ['input.mp3']
            return []

        with patch('glob.glob', side_effect=side_effect_glob):
            mock_sep_instance.separate.return_value = ['input_(Lead Vocals).wav']
            
            vocal_sep = VocalSeparator(self.output_dir, config)
            results = vocal_sep.separate_files('some_input_dir')
            
            # Verify separate was called once per input file with absolute path
            abs_input = os.path.abspath('input.mp3')
            mock_sep_instance.separate.assert_called_once_with(abs_input)
            
            abs_output_dir = os.path.abspath(self.output_dir)
            expected_file = os.path.join(abs_output_dir, 'input_(Lead Vocals).wav')
            self.assertIn(expected_file, results)

    @patch('src.separator.Separator')
    def test_dereverb_chaining_with_preset(self, MockSeparator):
        """Verify that Dereverb runs after Preset separation."""
        config = {
            'use_preset': True,
            'ensemble_preset': 'karaoke',
            'enable_dereverb': True,
            'dereverb_model_filename': 'dereverb_model.ckpt'
        }
        
        mock_sep_instance = MockSeparator.return_value
        
        def side_effect_glob(pattern):
            if pattern.endswith('*.mp3'):
                return ['input.mp3']
            return []

        with patch('glob.glob', side_effect=side_effect_glob):
            # 1st call: Preset separation, 2nd call: Dereverb
            mock_sep_instance.separate.side_effect = [
                ['input_(Lead Vocals).wav'],
                ['input_(Lead Vocals)_(No Reverb).wav']
            ]
            
            vocal_sep = VocalSeparator(self.output_dir, config)
            results = vocal_sep.separate_files('some_input_dir')
            
            # Verify final result contains the dry file with absolute path
            abs_output_dir = os.path.abspath(self.output_dir)
            expected_file = os.path.join(abs_output_dir, 'input_(Lead Vocals)_(No Reverb).wav')
            self.assertIn(expected_file, results)

    @patch('src.separator.Separator')
    def test_dereverb_skipping_when_disabled(self, MockSeparator):
        """Verify that Dereverb is skipped when enable_dereverb is False."""
        config = {
            'use_preset': True,
            'ensemble_preset': 'karaoke',
            'enable_dereverb': False
        }
        
        mock_sep_instance = MockSeparator.return_value
        
        def side_effect_glob(pattern):
            if pattern.endswith('*.mp3'):
                return ['input.mp3']
            return []

        with patch('glob.glob', side_effect=side_effect_glob):
            mock_sep_instance.separate.return_value = ['input_(Lead Vocals).wav']
            
            vocal_sep = VocalSeparator(self.output_dir, config)
            results = vocal_sep.separate_files('some_input_dir')
            
            # load_model only called once for preset
            self.assertEqual(mock_sep_instance.load_model.call_count, 1)
            self.assertEqual(mock_sep_instance.separate.call_count, 1)
            self.assertIn("(Lead Vocals)", results[0])
            self.assertNotIn("(No Reverb)", results[0])

if __name__ == '__main__':
    unittest.main()
