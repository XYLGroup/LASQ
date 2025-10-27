import os
import torch
import torch.utils.data
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from datasets.data_augment import PairCompose, PairToTensor, PairRandomHorizontalFilp, PairResize, PairNormalize
from torch.utils.data.distributed import DistributedSampler

class LLdataset:
    def __init__(self, config):
        self.config = config

    def get_loaders(self, world_size=1, local_rank=0):
        train_dataset = AllWeatherDataset(self.config.data.data_dir,
                                          patch_size=self.config.data.patch_size,
                                          filelist='{}_train.txt'.format(self.config.data.train_dataset),
                                          train=True)
        val_dataset = AllWeatherDataset(self.config.data.data_dir,
                                        patch_size=self.config.data.patch_size,
                                        filelist='{}_val.txt'.format(self.config.data.val_dataset),
                                        train=False)

        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=local_rank,
            shuffle=True
        ) if world_size > 1 else None

        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=local_rank,
            shuffle=False
        ) if world_size > 1 else None

        train_batch_size = self.config.training.batch_size // world_size if world_size > 1 else self.config.training.batch_size

        train_loader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=self.config.data.num_workers,
            pin_memory=True,
            drop_last=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.sampling.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=self.config.data.num_workers,
            pin_memory=True
        )

        return train_loader, val_loader


class AllWeatherDataset(torch.utils.data.Dataset):
    def __init__(self, dir, patch_size, filelist=None, train=True):
        super().__init__()

        self.dir = dir
        self.file_list = filelist
        self.train_list = os.path.join(dir, self.file_list)
        with open(self.train_list) as f:
            contents = f.readlines()
            input_names = [i.strip() for i in contents]

        self.input_names = input_names
        self.patch_size = patch_size

        if train:
            self.transforms = PairCompose([
                PairRandomHorizontalFilp(),
                PairToTensor()
            ])
        else:
            self.transforms = PairCompose([
                PairToTensor()
            ])



    def get_images(self, index):
        input_name = self.input_names[index].replace('\n', '')

        low_img_name, high_light_name = input_name.split(' ')[0], input_name.split(' ')[1]

        img_id = low_img_name.split('/')[-1]
        low_img, high_light_name_img = Image.open(low_img_name).convert("RGB"), Image.open(high_light_name).convert("RGB")

        low_img, high_light_name_img = \
            self.transforms(low_img, high_light_name_img)

        return torch.cat([low_img, high_light_name_img], dim=0), img_id

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)





