import os
import uuid
from argparse import Namespace

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    print("WARNING: Tensorboard not found. Not logging progress")
    TENSORBOARD_FOUND = False


def prepare_output_and_logger(args, use_tensorboard = True):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND and use_tensorboard:
        tb_writer = SummaryWriter(args.model_path)
    else:
        if use_tensorboard:
            print("Tensorboard not available: not logging progress. ")
        else:
            print("Not logging progress using Tensorboard. ")
    return tb_writer