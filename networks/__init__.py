# Adapted from https://github.com/dongliangcao/Self-Supervised-Multimodal-Shape-Matching/blob/main/networks/__init__.py
import importlib
import os.path as osp
import sys
from utils import scandir
from utils.registry import NETWORK_REGISTRY

__all__ = ['build_network']

# automatically scan and import network modules for registry
# scan all the files under the 'networks' folder and collect files ending with
# '_network.py'
network_folder = osp.dirname(osp.abspath(__file__))
network_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(network_folder) if v.endswith('_network.py')]
# import all the network modules
_network_modules = [importlib.import_module(f'networks.{file_name}') for file_name in network_filenames]


def build_network(opt):
    """Build network from options.

    Args:
        opt (dict): Configuration. It must contain:
            type (str): Network type.

    Returns:
        network (nn.Module): network built by opt.
    """
    network_type = opt.pop('type')
    network = NETWORK_REGISTRY.get(network_type)(**opt)
    print(f'Network [{network.__class__.__name__}] is created.', file=sys.stderr)  # logging
    return network
