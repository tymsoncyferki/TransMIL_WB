import argparse
from pathlib import Path
import numpy as np
import glob

from datasets import DataInterface
from models import ModelInterface
from utils.utils import *

# pytorch_lightning
import pytorch_lightning as pl
from pytorch_lightning import Trainer

#--->Setting parameters
def make_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', default='train', type=str)
    parser.add_argument('--config', default='Camelyon/TransMIL.yaml',type=str)
    parser.add_argument('--gpus', default = [])
    parser.add_argument('--fold', default = 0)
    args = parser.parse_args()
    return args

#---->main
def main(cfg):

    #---->Initialize seed
    pl.seed_everything(cfg.General.seed)

    #---->load loggers
    cfg.load_loggers = load_loggers(cfg)

    #---->load callbacks
    cfg.callbacks = load_callbacks(cfg)

    #---->Define Data 
    DataInterface_dict = {'train_batch_size': cfg.Data.train_dataloader.batch_size,
                'train_num_workers': cfg.Data.train_dataloader.num_workers,
                'test_batch_size': cfg.Data.test_dataloader.batch_size,
                'test_num_workers': cfg.Data.test_dataloader.num_workers,
                'dataset_name': cfg.Data.dataset_name,
                'dataset_cfg': cfg.Data,}
    dm = DataInterface(**DataInterface_dict)

    #---->Define Model
    ModelInterface_dict = {'model': cfg.Model,
                            'loss': cfg.Loss,
                            'optimizer': cfg.Optimizer,
                            'data': cfg.Data,
                            'log': cfg.log_path
                            }
    model = ModelInterface(**ModelInterface_dict)
    
    #---->Instantiate Trainer
    trainer = Trainer(
        num_sanity_val_steps=0, 
        logger=cfg.load_loggers,
        callbacks=cfg.callbacks,
        max_epochs=cfg.General.epochs,
        gpus=cfg.General.gpus,
        amp_level=cfg.General.amp_level,
        precision=cfg.General.precision,
        accumulate_grad_batches=cfg.General.grad_acc,
        deterministic=True,
        check_val_every_n_epoch=1,
    )

    #---->train or test
    if 'train' in cfg.General.server:
        trainer.fit(model = model, datamodule = dm)
    else:
        model_paths = list(cfg.log_path.glob('*.ckpt'))
        model_paths = [str(model_path) for model_path in model_paths if 'epoch' in str(model_path)]
        print(model_paths)
        for path in model_paths:
            print(path)
            new_model = model.load_from_checkpoint(checkpoint_path=path, cfg=cfg)
            trainer.test(model=new_model, datamodule=dm)

def test(cfg, name, loss, **kwargs):
    import json
    import os
    import time 
    import shutil
    import sys
    import warnings
    from addict import Dict
    
    warnings.simplefilter("ignore")

    base_dir = "logs\Bisque\TransMIL"
    f = open('nul', 'w')

    cfg.Optimizer = Dict(kwargs)
    cfg.Loss = Dict({'base_loss': loss})
    opts = {"loss": cfg.Loss, "optimizer": cfg.Optimizer}
    print(opts)
    cfg.config = args.config
    cfg.General.gpus = args.gpus
    cfg.General.server = args.stage
    cfg.Data.fold = args.fold
    
    cfg.General.server = "train"
    print("Train started.")
    sys.stdout = f
    # sys.stder = f
    t0 = time.time()
    main(cfg)
    opts["time_train"] = time.time() - t0
    sys.stdout = sys.__stdout__
    print(f"Train finished. Took {opts['time_train']}")
    
    cfg.General.server = "test"
    print("Test started.")
    sys.stdout = f
    t0 = time.time()
    main(cfg)
    opts["time_test"] = time.time() - t0
    sys.stdout = sys.__stdout__
    sys.stder = sys.__stderr__
    print(f"Test finished. Took {opts['time_test']}")
    
    f.close()
    
    os.makedirs(os.path.join(base_dir, name), exist_ok=True)
    os.rename(os.path.join(base_dir, "fold0", "metrics.csv"), os.path.join(base_dir, name, "metrics.csv"))
    os.rename(os.path.join(base_dir, "fold0", "result.csv"), os.path.join(base_dir, name, "result.csv"))
    with open(os.path.join(base_dir, name, "cfg.json"), 'w', encoding='utf-8') as f:
        json.dump(opts, f, ensure_ascii=False, indent=4)
    shutil.rmtree(os.path.join(base_dir, "fold0"))

if __name__ == "__main__":
    args = make_parse()
    cfg = read_yaml(args.config)
    
    for loss in (
        "CrossEntropyLoss",
        # "MSELoss",
        # "SmoothL1Loss",
        # "focal",
    ):
        for opts in (
            {"opt": "lookahead_radam", "lr": 0.001, "weight_decay": 0.0001, "opt_betas": (0.9, 0.999), "opt_eps": 1e-08},
            {"opt": "lookahead_radam", "lr": 0.01, "weight_decay": 0.0001, "opt_betas": (0.9, 0.999), "opt_eps": 1e-08},
            {"opt": "lookahead_novograd", "lr": 0.001, "weight_decay": 0.0001, "opt_betas": (0.9, 0.999), "opt_eps": 1e-08},
            {"opt": "lookahead_novograd", "lr": 0.01, "weight_decay": 0.0001, "opt_betas": (0.9, 0.999), "opt_eps": 1e-08},
            {"opt": "lookahead_adadelta", "lr": 0.001, "weight_decay": 0.0001, "opt_eps": 1e-08},
            {"opt": "lookahead_adadelta", "lr": 0.01, "weight_decay": 0.0001, "opt_eps": 1e-08},
            {"opt": "lookahead_sgdp", "lr": 0.001, "momentum":0.9, "weight_decay": 0.0001, "opt_eps": 1e-08},
        ):
            name = loss.lower() + "_" + opts["opt"] + "_" + str(opts["lr"])
            print(f"Starting for name {name}")
            test(cfg.copy(), name, loss, **opts)

# if __name__ == '__main__':  

#     args = make_parse()
#     cfg = read_yaml(args.config)

#     #---->update
#     cfg.config = args.config
#     cfg.General.gpus = args.gpus
#     cfg.General.server = args.stage
#     cfg.Data.fold = args.fold
#     #---->main
#     main(cfg)
 