import unittest
import torch
from visual_snn.encoders import LastImageEncoder, RateEncoder, align_features
from visual_snn.losses import encoder_objective
from visual_snn.model import ImageSRNN
from visual_snn.engine import classifier_loss
from visual_snn.data import make_loaders


class ExtensionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        torch.set_num_threads(1)

    def test_soft_pretraining_reaches_features_and_thresholds(self):
        enc = LastImageEncoder(1, 4, steps=3, hidden=8)
        x = torch.rand(4, 1, 4, 4)
        p, flat, z1, z2 = enc.soft(x)
        loss, *_ = encoder_objective(p, flat, z1, z2)
        loss.backward()
        for parameter in [enc.linear1.weight, enc.linear2.weight, enc.threshold]:
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_binary_events_and_temporal_layout(self):
        enc = LastImageEncoder(1, 4, steps=2, hidden=8).eval()
        with torch.no_grad():
            enc.linear1.weight.zero_()
            enc.linear2.weight.zero_()
            enc.linear2.bias.fill_(0.5)
            enc.threshold[0].fill_(0.1)
            enc.threshold[1].fill_(0.9)
        events = enc(torch.rand(3, 1, 4, 4))
        self.assertEqual(tuple(events.shape), (2, 3, 1, 4, 4))
        self.assertTrue((events[0] == 1).all())
        self.assertTrue((events[1] == 0).all())

    def test_frozen_threshold_control_trains_features_only(self):
        enc = LastImageEncoder(1, 4, steps=3, hidden=8, learn_thresholds=False)
        before = enc.threshold.detach().clone()
        p, x, z1, z2 = enc.soft(torch.rand(4, 1, 4, 4))
        loss, *_ = encoder_objective(p, x, z1, z2)
        loss.backward()
        torch.optim.AdamW([p for p in enc.parameters() if p.requires_grad]).step()
        self.assertTrue(torch.equal(enc.threshold, before))
        self.assertIsNone(enc.threshold.grad)

    def test_repository_omission_is_explicit(self):
        p, x = torch.rand(3, 2, 16), torch.rand(2, 16)
        z1, z2 = torch.rand_like(x), torch.rand_like(x)
        first = encoder_objective(p, x, z1, z2, information_terms='repository')[0]
        second = encoder_objective(p, x, 2*z1, z2, information_terms='repository')[0]
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.isfinite(encoder_objective(p, x, z1, z2)[0]))

    def test_recurrent_state_resets_and_gradients(self):
        model = ImageSRNN(16, 3, hidden=8)
        events = RateEncoder(4)(torch.rand(2, 1, 4, 4))
        out, mem, stats = model(events)
        other, other_mem, _ = model(events)
        self.assertTrue(torch.equal(out, other))
        self.assertTrue(torch.equal(mem, other_mem))
        classifier_loss(mem, torch.tensor([0, 2])).backward()
        self.assertTrue(torch.isfinite(model.fc1.weight.grad).all())
        self.assertGreater(model.fc1.weight.grad.abs().sum().item(), 0)
        self.assertGreaterEqual(stats['synaptic_events_proxy'].item(), 0)

    def test_dimension_alignment(self):
        for n in [8, 15, 16, 30]:
            z = torch.rand(2, n, requires_grad=True)
            a = align_features(z, 16)
            self.assertEqual(tuple(a.shape), (2, 16))
            a.sum().backward()
            self.assertIsNotNone(z.grad)

    def test_batch_of_one(self):
        enc = LastImageEncoder(1, 4, hidden=8)
        p, *_ = enc.soft(torch.rand(1, 1, 4, 4))
        self.assertTrue(torch.isfinite(p).all())

    def test_data_split_independence(self):
        c = dict(dataset='mnist', split_seed=42, val_fraction=.2, seed=0, batch_size=8, limit=None)
        _, _, split = make_loaders(c, fake=True)
        _, _, other = make_loaders({**c, 'seed': 3}, fake=True)
        self.assertEqual(split, other)
        self.assertFalse(set(split['train_indices']) & set(split['val_indices']))


if __name__ == '__main__':
    unittest.main()
