import functools
import os

def is_main_process():
    return int(os.environ.get("SLURM_PROCID", 0)) == 0


#def only_on_main_process(func: Callable[P, R] | None = None,):
#    if callable(func):
#
#        @functools.wraps(func)
#        def wrapper(*args: P.args, **kwargs: P.kwargs):
#            if is_main_process():
#                return func(*args, **kwargs)
#            return None
#
#        return wrapper
