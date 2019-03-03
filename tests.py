import unittest
from utils import Feats
from utils import SimulateRaw
from utils import PreProcess
from utils import CreateModel
from utils import TrainTestVal
from utils import LoadMuseData
from utils import FeatureEngineer


class ExampleTest(unittest.TestCase):
    """
    Our basic test class
    """

    def test_addition(self):
        """
        The actual test.
        Any method which starts with ``test_`` will considered as a test case.
        """
        res = 2 + 2
        self.assertEqual(res, 4)

    def test_feats(self):
        """
        Testing utils import.
        """
        feats = Feats()
        self.assertEqual(feats.num_classes, 2)

    def test_example_muse(self):
        """
        Testing example Muse code.
        """

        # Load Data
        raw = LoadMuseData(subs=[101, 102], nsesh=2, data_dir='visual/cueing')

        # Pre-Process EEG Data
        epochs = PreProcess(raw=raw, event_id={'LeftCue': 1, 'RightCue': 2})

        # Engineer Features for Model
        feats = FeatureEngineer(epochs=epochs)

        # Create Model
        model, _ = CreateModel(feats=feats)

        # Train with validation, then Test
        model, data = TrainTestVal(model=model,
                                   feats=feats,
                                   train_epochs=1,
                                   show_plots=False)

        self.assertLess(data['acc'], 1)


    def test_simulate_raw(self):
        """
        Testing simulated data pipeline.
        """
        # Simulate Data
        raw,event_id = SimulateRaw(amp1=50, amp2=60, freq=1.)

        # Pre-Process EEG Data
        epochs = PreProcess(raw,event_id)

        # Engineer Features for Model
        feats = FeatureEngineer(epochs)

        # Create Model
        model, _ = CreateModel(feats, units=[16,16])

        # Train with validation, then Test
        model, data = TrainTestVal(model,feats, 
                    train_epochs=1,show_plots=False)

        self.assertLess(data['acc'], 1)
    
    def test_frequencydomain_complex(self):
        """
        Testing simulated data pipeline.
        """
        # Simulate Data
        raw,event_id = SimulateRaw(amp1=50, amp2=60, freq=1.)

        # Pre-Process EEG Data
        epochs = PreProcess(raw,event_id)

        # Engineer Features for Model
        feats = FeatureEngineer(epochs,frequency_domain=True,
                                include_phase=True)

        # Create Model
        model, _ = CreateModel(feats, units=[16,16])

        # Train with validation, then Test
        model, data = TrainTestVal(model,feats, 
                    train_epochs=1,show_plots=False)

        self.assertLess(data['acc'], 1)

if __name__ == '__main__':
    unittest.main()