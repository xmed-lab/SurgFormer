import torch.utils.data

from .cholec80 import build as build_cholec
from .mvp import build as build_mvp


def build_dataset(image_set, args):
    if args.dataset_file == 'mvp':
        return build_mvp(image_set, args)
    if args.dataset_file == 'cholec80':
        return build_cholec(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')
