import argparse
import json
import numpy as np
import os
from pathlib import Path
import lightning as L
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import timm.optim.optim_factory as optim_factory
from pathlib import Path
from lightning.pytorch.callbacks import ModelCheckpoint, DeviceStatsMonitor, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch import seed_everything
import time
import utils.misc as misc
from utils.misc import NativeScalerWithGradNormCount as NativeScaler
import swin_mae
from utils.engine_pretrain import train_one_epoch
from utils.data import GeoWebDataset, UsualTransform


seed_everything(222)


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=48, type=int)
    parser.add_argument('--max_steps', default=400, type=int)
    parser.add_argument('--resume', default='', type=str,
                        help='path to resume checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--data_path', type=str)  # fill in the dataset path here
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--model', default='swin_tiny', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--in_chans', default=4, type=int,
                        help='images input channels')
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)
    # optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_steps', type=int, default=10, metavar='N',
                        help='epochs to warmup LR')
    # other parameters
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='gpu',
                        help='device to use for training / testing')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.set_defaults(pin_mem=True)
    return parser


def cosine_scheduler_step_based(base_value, final_value, warmup_iters, total_iters, start_warmup_value=0):
    warmup_schedule = np.array([])
    if warmup_iters > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)
    iters = np.arange(total_iters - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == total_iters
    return schedule


# Define the Lightning module
class LitModel(L.LightningModule):
    def __init__(self, model, args):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.automatic_optimization = False
        self.model = model
        self.weight_decay = args.weight_decay
        self.lr = args.lr
        self.lr_sched = cosine_scheduler_step_based(args.lr, args.min_lr, args.warmup_steps, args.max_steps)

    def configure_optimizers(self):
        param_groups = optim_factory.param_groups_weight_decay(self.model, self.weight_decay)
        optimizer = torch.optim.AdamW(param_groups, lr=self.lr, betas=(0.99, 0.995))  # default betas=(0.9. 0.95)
        return optimizer

    def training_step(self, batch):
        opt = self.optimizers()
        # update learning rate
        for i, param_group in enumerate(opt.param_groups):
            param_group['lr'] = self.lr_sched[self.global_step]
        # forward pass
        loss, _, _ = self.model(batch)
        self.log("train_loss", loss)
        # opt and backward pass
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()
        return loss


def main(args):
    # Data and transforms
    transform_train = UsualTransform(args.input_size)

    dataset_train = GeoWebDataset(root=args.data_path,
                                  n_bands=args.in_chans,
                                  augmentations=transform_train,
                                  num_nodes=1,
                                  num_shards=1,
                                  imgs_per_shard=3)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True
    )

    # Models
    model = swin_mae.__dict__[args.model](in_chans=args.in_chans, norm_pix_loss=args.norm_pix_loss,
                                          mask_ratio=args.mask_ratio)

    # compiled_mae = torch.compile(model)
    masked_autoencoder = LitModel(model, args)

    # Callbacks
    checkpoint_callback = ModelCheckpoint(dirpath=args.output_dir,
                                          every_n_train_steps=int(args.max_steps/5),
                                          save_last=True)
    lr_callback = LearningRateMonitor(logging_interval="step")

    # Logger
    logger = TensorBoardLogger(save_dir=args.output_dir,
                               name="")

    # Trainer and Training
    trainer = L.Trainer(accelerator=args.device,
                        max_steps=args.max_steps,
                        log_every_n_steps=10,
                        default_root_dir=args.output_dir,
                        # profiler="simple",
                        enable_progress_bar=True,
                        logger=logger,
                        # precision="bf16-mixed",
                        callbacks=[checkpoint_callback, lr_callback])

    print("beginning the training.")
    start = time.time()
    if not args.resume:
        trainer.fit(model=masked_autoencoder, train_dataloaders=data_loader_train)
    else:
        trainer.fit(model=masked_autoencoder, train_dataloaders=data_loader_train, ckpt_path=args.resume)
    end = time.time()
    print(f"training completed. Elapsed time {end - start} seconds.")


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
