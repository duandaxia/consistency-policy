"""
Usage:
python eval.py --checkpoint data/image/pusht/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt -o data/pusht_eval_output
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import pathlib
import click
import hydra
import torch
import dill
import wandb
import json
from diffusion_policy.workspace.base_workspace import BaseWorkspace
import torch
from torch import nn
from diffusion_policy.dataset.base_dataset import BaseImageDataset, LinearNormalizer
from consistency_policy.utils import get_policy, rmat_to_quat, rot6d_to_rmat
# from consistency_policy.policy_wrapper import PolicyWrapperRobomimic
import numpy as np

@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output_dir', required=True)
@click.option('-d', '--device', default='cuda:0')
def main(checkpoint, output_dir, device):
    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    print("Available keys in payload:", cfg.keys())

    policy = get_policy(checkpoint)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    policy.to(device)
    policy.eval()

    # dataset_dir="/proj/rep-learning-robotics/users/x_yufdu/daxia/diffusion/realtime_dp/diffusion_policy/data/robomimic/datasets"
    # task_name = cfg.task.task_name
    # if cfg.name == "train_tedi_unet_hybrid":
    #     file_name = "image_abs.hdf5"
    # else:
    #     file_name = "low_dim_abs.hdf5"
    # dataset_path = os.path.join(dataset_dir, task_name, "ph",file_name)
    # print(dataset_path)
    # cfg.task.dataset.dataset_path = dataset_path
    # cfg.task.dataset= dataset_path
    # cfg.task.env_runner.dataset_path = dataset_path
    
    # workspace: BaseWorkspace = cls(cfg, output_dir=output_dir)
    # workspace.load_payload(payload, exclude_keys="optimizer", include_keys=None)
    
    # get policy from workspace
    # policy = workspace.model
    # if cfg.training.use_ema:
    #     policy = workspace.ema_model
    

    
    # run eval
    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=output_dir)
    runner_log = env_runner.run(policy)
    
    # dump log to json
    json_log = dict()
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        else:
            json_log[key] = value
    out_path = os.path.join(output_dir, 'eval_log.json')
    json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)

if __name__ == '__main__':
    main()
