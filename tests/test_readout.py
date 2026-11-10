import unittest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from visual_snn.readout import scores, objective, prefix_scores, improves
from visual_snn.model import ImageSRNN
from visual_snn.encoders import RateEncoder
from visual_snn.engine import classifier_pass


class ReadoutTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        torch.set_num_threads(1)

    def test_membrane_objective_and_scores_agree(self):
        m = torch.randn(3, 2, 4, requires_grad=True)
        s = torch.zeros_like(m)
        y = torch.tensor([1, 2])
        self.assertTrue(torch.equal(objective(s, m, y, 'membrane'), F.cross_entropy(m.mean(0), y)))
        self.assertTrue(torch.equal(prefix_scores(s, m, 'membrane')[-1], scores(s, m, 'membrane')))

    def test_spike_objective_and_scores_agree(self):
        s = torch.randn(3, 2, 4, requires_grad=True)
        m = torch.zeros_like(s)
        y = torch.tensor([1, 2])
        self.assertTrue(torch.equal(objective(s, m, y, 'spike'), F.cross_entropy(s.sum(0), y)))
        self.assertTrue(torch.equal(prefix_scores(s, m, 'spike')[-1], scores(s, m, 'spike')))

    def test_spike_loss_gradient_flows_when_output_is_silent(self):
        model = ImageSRNN(16, 3, hidden=8)
        with torch.no_grad():
            model.fc2.weight.zero_()
            model.fc2.bias.zero_()
            model.lif2.recurrent.weight.zero_()
            model.lif2.recurrent.bias.zero_()
        s, m, _ = model(torch.ones(4, 2, 1, 4, 4))
        self.assertEqual(s.sum().item(), 0)
        objective(s, m, torch.tensor([1, 2]), 'spike').backward()
        self.assertGreater(model.fc2.bias.grad.abs().sum().item(), 0)

    def test_legacy_objective_preserved(self):
        m, s = torch.randn(3, 2, 4), torch.zeros(3, 2, 4)
        y = torch.tensor([1, 2])
        expected = sum(F.cross_entropy(step, y) for step in m)
        self.assertTrue(torch.equal(objective(s, m, y, 'paper'), expected))
        self.assertTrue(torch.equal(scores(s, m, 'paper'), s.sum(0)))

    def test_checkpoint_selection_follows_selected_metric(self):
        old = {'accuracy': .37, 'classifier_loss': 7.4}
        new = {'accuracy': .22, 'classifier_loss': 6.7}
        self.assertFalse(improves(new, old, 'membrane'))
        self.assertFalse(improves(new, old, 'spike'))
        self.assertTrue(improves(new, old, 'paper'))
        self.assertTrue(improves({'accuracy': .37, 'classifier_loss': 7.3}, old, 'spike'))

    def test_silent_outputs_do_not_override_membrane_prediction(self):
        class SilentModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc2 = nn.Linear(1, 3)
            def forward(self, events):
                m = torch.tensor([[0., 3., 1.], [0., 1., 3.]])[None].repeat(2, 1, 1)
                z = torch.zeros_like(m)
                return z, m, {key: torch.tensor(0.) for key in
                              ['hidden_spikes', 'output_spikes', 'synaptic_events_proxy']}
        loader = DataLoader(TensorDataset(torch.zeros(2, 1, 1, 1), torch.tensor([1, 2])), batch_size=2)
        c = {'device': 'cpu', 'steps': 2, 'eval_seed': 1, 'readout': 'membrane'}
        result = classifier_pass(SilentModel(), RateEncoder(2), loader, c)
        self.assertEqual(result['accuracy'], 1)
        self.assertEqual(result['spike_readout_accuracy'], 0)
        self.assertEqual(result['no_output_spike_fraction'], 1)
        self.assertEqual(result['accuracy_by_timestep'], [1, 1])
        self.assertEqual(sum(map(sum, result['confusion_matrix'])), 2)


if __name__ == '__main__':
    unittest.main()
