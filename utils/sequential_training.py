import gc
import torch
from torch.utils.data import DataLoader


def cleanup_resources(device=None):
    gc.collect()
    if device is not None and 'cuda' in str(device):
        torch.cuda.empty_cache()


class SequentialDomainRunner:
    """Manages DataLoader lifecycle across domains.

    Ensures only one domain's DataLoaders exist at any time by destroying
    all loaders and releasing resources when a domain completes.

    Usage:
        runner = SequentialDomainRunner(device, generator, worker_init_fn)
        for domain in domains:
            with runner.domain_context():
                loader = runner.create_loader(dataset, ...)
                # train, evaluate ...
            # all loaders for this domain destroyed, resources cleaned
    """

    def __init__(self, device=None, generator=None, worker_init_fn=None):
        self.device = device
        self.generator = generator
        self.worker_init_fn = worker_init_fn
        self._loaders = []

    def create_loader(self, dataset, batch_size, shuffle, num_workers,
                      pin_memory, drop_last=False):
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory,
            drop_last=drop_last,
            generator=self.generator,
            worker_init_fn=self.worker_init_fn,
        )
        self._loaders.append(loader)
        return loader

    def register_loaders(self, loaders):
        if isinstance(loaders, (list, tuple)):
            self._loaders.extend(loaders)
        else:
            self._loaders.append(loaders)

    def destroy_loaders(self):
        for loader in self._loaders:
            if loader is not None:
                del loader
        self._loaders.clear()
        cleanup_resources(self.device)

    def domain_context(self):
        return _DomainContext(self)


class _DomainContext:
    def __init__(self, runner):
        self.runner = runner

    def __enter__(self):
        return self.runner

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.runner.destroy_loaders()
