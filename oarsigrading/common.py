import torch.distributed as dist
import logging
import functools


def get_dist_info():
    initialized = dist.is_initialized()
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def master_only(func):
    """
    Implementation by Artur Kuzin
    Args:
        func:
    Returns:
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rank, _ = get_dist_info()
        if rank == 0:
            return func(*args, **kwargs)

    return wrapper


class Logger:
    @staticmethod
    @master_only
    def log_info(whattolog, where_from=None):
        if where_from is None:
            where_from = __name__
        logger = logging.getLogger(where_from)
        logger.info(whattolog)

    @staticmethod
    @master_only
    def log_error(whattolog, where_from=None):
        if where_from is None:
            where_from = __name__
        logger = logging.getLogger(where_from)
        logger.error(whattolog)