import random
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torchvision.transforms import Resize as TResize
import PIL
# from PIL import Image


class PairRandomCrop(transforms.RandomCrop):

    def __call__(self, image, label):

        if self.padding is not None:
            image = F.pad(image, self.padding, self.fill, self.padding_mode)
            label = F.pad(label, self.padding, self.fill, self.padding_mode)

        # pad the width if needed
        if self.pad_if_needed and image.size[0] < self.size[1]:
            image = F.pad(image, (self.size[1] - image.size[0], 0), self.fill, self.padding_mode)
            label = F.pad(label, (self.size[1] - label.size[0], 0), self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and image.size[1] < self.size[0]:
            image = F.pad(image, (0, self.size[0] - image.size[1]), self.fill, self.padding_mode)
            label = F.pad(label, (0, self.size[0] - image.size[1]), self.fill, self.padding_mode)

        i, j, h, w = self.get_params(image, self.size)

        return F.crop(image, i, j, h, w), F.crop(label, i, j, h, w)


class PairCompose(transforms.Compose):
    def __call__(self, *images):
        for t in self.transforms:
            images = t(*images)
        return images



class PairRandomHorizontalFilp(transforms.RandomHorizontalFlip):
    def __call__(self, *imgs):
        if random.random() < self.p:
            return tuple(F.hflip(img) for img in imgs)
        return imgs


class PairRandomVerticalFlip(transforms.RandomVerticalFlip):
    def __call__(self, img, label):
        if random.random() < self.p:
            return F.vflip(img), F.vflip(label)
        return img, label


class PairToTensor(transforms.ToTensor):
    def __call__(self, *pics):
        return tuple(F.to_tensor(pic) for pic in pics)



class PairResize:
    def __init__(self, size):
        self.resize = transforms.Resize(size)

    def __call__(self, img, label):
        return self.resize(img), self.resize(label)


class PairNormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img, label):
        if isinstance(img, PIL.Image.Image):
            img = F.to_tensor(img)

        if isinstance(label, PIL.Image.Image):
            label = F.to_tensor(label)

        img = F.normalize(img, mean=self.mean, std=self.std)
        label = F.normalize(label, mean=self.mean, std=self.std)

        return img, label
