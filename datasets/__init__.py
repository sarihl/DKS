# Adapted from https://github.com/dongliangcao/Self-Supervised-Multimodal-Shape-Matching/blob/main/datasets/__init__.py
import importlib
from copy import deepcopy
from os import path as osp
import sys

import torchvision.transforms as transforms

from utils import scandir
from utils.registry import DATASET_REGISTRY

__all__ = ['build_dataset']

# automatically scan and import dataset modules for registry
# scan all the files under the data folder with '_dataset' in file names
data_folder = osp.dirname(osp.abspath(__file__))
dataset_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(data_folder, recursive=True) if
                     v.endswith('_dataset.py')]
# import all the dataset modules
_dataset_modules = [importlib.import_module(f'datasets.{file_name}') for file_name in dataset_filenames]


def build_transform(transform_opt):
    if transform_opt is None:
        return None
    transform = []
    normalize = None  # normalize can only be applied in torch.Tensor

    for name, params in transform_opt.items():
        if name == 'CenterCrop':
            transform += [transforms.CenterCrop(**params)]
        elif name == 'RandomCrop':
            transform += [transforms.RandomCrop(**params)]
        elif name == 'Resize':
            transform += [transforms.Resize(**params)]
        elif name == 'RandomResizedCrop':
            transform += [transforms.RandomResizedCrop(**params)]
        elif name == 'RandomHorizontalFlip':
            transform += [transforms.RandomHorizontalFlip(**params)]
        elif name == 'RandomVerticalFlip':
            transform += [transforms.RandomVerticalFlip(**params)]
        elif name == 'FiveCrop':
            transform += [transforms.FiveCrop(**params)]
        elif name == 'Pad':
            transform += [transforms.Pad(**params)]
        elif name == 'RandomAffine':
            transform += [transforms.RandomAffine(**params)]
        elif name == 'ColorJitter':
            transform += [transforms.ColorJitter(**params)]
        elif name == 'Grayscale':
            transform += [transforms.Grayscale(**params)]
        elif name == 'Normalize':
            normalize = params
        else:
            raise ValueError(f'The transform {name} is currently not support!')

    # convert PIL.Image to torch.Tensor
    transform += [transforms.ToTensor()]
    if normalize:
        transform += [transforms.Normalize(**normalize)]

    return transforms.Compose(transform)


def build_dataset(dataset_opt):
    """Build dataset from options.

    Args:
        dataset_opt (dict): Configuration for dataset. It must contain:
            type (str): dataset type.
    Return:
        dataset (pytorch_lightning.dataset): dataset built by opt.
    """
    dataset_opt = deepcopy(dataset_opt)
    dataset_type = dataset_opt.pop('type')
    # build transform
    if 'transform' in dataset_opt:
        dataset_opt['transform'] = build_transform(dataset_opt['transform'])
    # build dataset
    dataset = DATASET_REGISTRY.get(dataset_type)(**dataset_opt)
    print(f'dataset [{dataset.__class__.__name__}] is built.', file=sys.stderr)  # logging
    return dataset
