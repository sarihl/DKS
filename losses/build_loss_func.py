from utils.registry import LOSS_REGISTRY
import sys


def build_loss(opt):
    """Build loss from options.

    Args:
        opt (dict): Configuration. It must contain:
            type (str): loss type.
            name (str): loss name for logging.
            coef (float): loss coefficient. Default: 1.0.

    Returns:
        loss (nn.Module): loss built by opt.
    """
    loss_type = opt.pop('type')
    loss = LOSS_REGISTRY.get(loss_type)(opt)
    print(f'loss [{loss.__class__.__name__}] is created.', file=sys.stderr)  # logging
    return loss
