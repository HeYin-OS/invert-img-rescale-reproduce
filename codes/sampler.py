import math
import torch
from torch.utils.data.sampler import Sampler
import torch.distributed


class IterSampler(Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, ratio=100):
        super().__init__()
        if num_replicas is None:
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.num_samples = int(math.ceil(len(self.dataset) * ratio / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.rank = rank
        self.epoch = 0

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.epoch)
        indices = torch.randperm(self.total_size, generator=generator).tolist()
        dsize = len(self.dataset)
        indices = [v % dsize for v in indices]
        indices = indices[self.rank:self.total_size:self.num_replicas]
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
