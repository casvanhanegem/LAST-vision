"""Two recurrent LIF layers, replacing two classes with 10 or 100 classes."""
import torch
from torch import nn
import snntorch as snn
from snntorch import surrogate


class ImageSRNN(nn.Module):
    def __init__(self, input_count, classes, hidden=500, beta=0.99):
        super().__init__()
        self.fc1 = nn.Linear(input_count, hidden)
        self.fc2 = nn.Linear(hidden, classes)
        self.lif1 = snn.RLeaky(beta=beta, linear_features=hidden,
                               spike_grad=surrogate.atan(), reset_mechanism='subtract')
        self.lif2 = snn.RLeaky(beta=beta, linear_features=classes,
                               spike_grad=surrogate.atan(), reset_mechanism='subtract')

    def forward(self, events):
        x = events.flatten(2)  # T,B,N: preserve actual temporal order.
        batch = x.shape[1]
        s1 = x.new_zeros(batch, self.fc1.out_features)
        s2 = x.new_zeros(batch, self.fc2.out_features)
        m1, m2 = torch.zeros_like(s1), torch.zeros_like(s2)
        spike_trace, mem_trace = [], []
        hidden_count = output_count = recurrent_ops = x.new_zeros(())
        for current in x:
            # Recurrent input uses the previous timestep's spikes.
            recurrent_ops = recurrent_ops + s1.detach().sum() * s1.shape[1]
            recurrent_ops = recurrent_ops + s2.detach().sum() * s2.shape[1]
            s1, m1 = self.lif1(self.fc1(current), s1, m1)
            s2, m2 = self.lif2(self.fc2(s1), s2, m2)
            spike_trace.append(s2)
            mem_trace.append(m2)
            hidden_count = hidden_count + s1.detach().sum()
            output_count = output_count + s2.detach().sum()
        return torch.stack(spike_trace), torch.stack(mem_trace), {
            'hidden_spikes': hidden_count, 'output_spikes': output_count,
            'synaptic_events_proxy': x.detach().sum() * self.fc1.out_features
                + hidden_count * self.fc2.out_features + recurrent_ops}


def make_model(c):
    channels, size, classes = (1, 28, 10) if c['dataset'] == 'mnist' else (3, 32, 100)
    return ImageSRNN(channels * size * size, classes, c['classifier_hidden'], c['beta'])
