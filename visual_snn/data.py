"""Static MNIST/CIFAR-100, kept in [0,1] for intensity-based encoding."""
import hashlib
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset


def make_loaders(config, fake=False):
    dataset = config['dataset']
    channels, size, classes = (1, 28, 10) if dataset == 'mnist' else (3, 32, 100)
    if fake:
        generator = torch.Generator().manual_seed(config['split_seed'])
        def samples(n):
            return TensorDataset(torch.rand(n, channels, size, size, generator=generator),
                                 torch.randint(classes, (n,), generator=generator))
        base, test = samples(48), samples(16)
    else:
        from torchvision import datasets, transforms
        constructor = datasets.MNIST if dataset == 'mnist' else datasets.CIFAR100
        transform = transforms.ToTensor()
        base = constructor(config['data_dir'], train=True, download=config['download'], transform=transform)
        test = constructor(config['data_dir'], train=False, download=config['download'], transform=transform)
    permutation = torch.randperm(len(base), generator=torch.Generator().manual_seed(config['split_seed']))
    nval = max(1, round(len(base) * config['val_fraction']))
    val_indices = permutation[:nval].tolist()
    train_indices = permutation[nval:].tolist()
    train, val = Subset(base, train_indices), Subset(base, val_indices)
    subsets = []
    for subset in (train, val, test):
        if config.get('limit'):
            subset = Subset(subset, range(min(config['limit'], len(subset))))
        subsets.append(subset)
    loaders = []
    for i, subset in enumerate(subsets):
        loaders.append(DataLoader(subset, batch_size=config['batch_size'], shuffle=i == 0,
                                  num_workers=0, generator=torch.Generator().manual_seed(config['seed'])))
    split = {'train_indices': train_indices, 'val_indices': val_indices,
             'sha256': hashlib.sha256(permutation.numpy().tobytes()).hexdigest(),
             'fake_data': fake, 'limit': config.get('limit')}
    return loaders, (channels, size, classes), split
