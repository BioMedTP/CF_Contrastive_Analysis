import random
import sys

sys.path =  ['.'] + sys.path

import torch
from arguments import training_arguments
from utils.common_utils import printer, setup_seed


if __name__ == "__main__":
    config = training_arguments.load_config()
    setup_seed(config.exp.seed)

    printer(config)
    if config.train.train_runner == "fse_inverter":
        from runners.training_runners_inverter import training_runners
    elif config.train.train_runner == "fse_editor_cs" or "fse_editor_cs1s2":
        from runners.training_runners_cs_encoder import training_runners
    
    trainer = training_runners[config.train.train_runner](config)
    trainer.setup()
    trainer.run()
