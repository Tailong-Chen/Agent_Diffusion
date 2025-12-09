import argparse
import datetime
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from util.crop import center_crop_arr
from tiff_dataset import TiffStackDataset, MultiTiffDataset
from tiff_dataset_grayscale import TiffStackDatasetNormalized
import util.misc as misc

import copy
from engine_jit import train_one_epoch, evaluate

# Import the new DenoiserUNet
from denoiser_unet import DenoiserUNet as Denoiser


def get_args_parser():
    parser = argparse.ArgumentParser('JiT-UNet', add_help=False)

    # architecture
    parser.add_argument('--model', default='UNet', type=str, metavar='MODEL',
                        help='Name of the model to train (UNet)')
    parser.add_argument('--img_size', default=256, type=int, help='Image size')
    parser.add_argument('--in_channels', default=1, type=int, help='Input channels (1 for grayscale TIFF, 3 for RGB)')
    parser.add_argument('--out_channels', default=1, type=int, help='Output channels')
    
    # UNet specific args (optional, can rely on defaults in DenoiserUNet)
    parser.add_argument('--attn_dropout', type=float, default=0.0, help='Attention dropout rate')
    parser.add_argument('--proj_dropout', type=float, default=0.0, help='Projection dropout rate')

    # training
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='Epochs to warm up LR')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='Learning rate (absolute)')
    parser.add_argument('--blr', type=float, default=1e-4, metavar='LR',
                        help='Base learning rate')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='Minimum LR')
    parser.add_argument('--lr_schedule', type=str, default='constant',
                        help='Learning rate schedule')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay')
    parser.add_argument('--ema_decay1', type=float, default=0.9999,
                        help='The first ema to track')
    parser.add_argument('--ema_decay2', type=float, default=0.9996,
                        help='The second ema to track')
    parser.add_argument('--P_mean', default=-0.8, type=float)
    parser.add_argument('--P_std', default=0.8, type=float)
    parser.add_argument('--noise_scale', default=1.0, type=float)
    parser.add_argument('--t_eps', default=5e-2, type=float)
    parser.add_argument('--label_drop_prob', default=0.1, type=float)

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='Starting epoch')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # sampling
    parser.add_argument('--sampling_method', default='heun', type=str,
                        help='ODE samping method')
    parser.add_argument('--num_sampling_steps', default=50, type=int,
                        help='Sampling steps')
    parser.add_argument('--cfg', default=1.0, type=float,
                        help='Classifier-free guidance factor')
    parser.add_argument('--interval_min', default=0.0, type=float,
                        help='CFG interval min')
    parser.add_argument('--interval_max', default=1.0, type=float,
                        help='CFG interval max')
    parser.add_argument('--num_images', default=16, type=int,
                        help='Number of images to generate during evaluation')
    parser.add_argument('--eval_freq', type=int, default=10,
                        help='Frequency (in epochs) for evaluation')
    parser.add_argument('--online_eval', action='store_true')
    parser.add_argument('--evaluate_gen', action='store_true')
    parser.add_argument('--gen_bsz', type=int, default=16,
                        help='Generation batch size')
    
    # gradient accumulation
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # dataset
    parser.add_argument('--data_path', default='./Data', type=str,
                        help='Path to the dataset')
    parser.add_argument('--class_num', default=1, type=int)
    parser.add_argument('--use_tiff', action='store_true', default=True,
                        help='Use TIFF stack dataset')
    parser.add_argument('--tiff_file', default='mt.tif', type=str,
                        help='TIFF filename in data_path')
    parser.add_argument('--use_normalized_tiff', action='store_true', default=False,
                        help='Use normalized TIFF dataset')
    parser.add_argument('--normalize_per_image', action='store_true', default=True,
                        help='Normalize each image independently')

    # checkpointing
    parser.add_argument('--output_dir', default='./output_jit_unet',
                        help='Directory to save outputs')
    parser.add_argument('--resume', default='',
                        help='Folder that contains checkpoint to resume from')
    parser.add_argument('--save_last_freq', type=int, default=5,
                        help='Frequency (in epochs) to save checkpoints')
    parser.add_argument('--log_freq', default=100, type=int)
    parser.add_argument('--device', default='cuda',
                        help='Device to use')

    # distributed training
    parser.add_argument('--world_size', default=1, type=int,
                        help='Number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='URL used to set up distributed training')

    return parser


def main(args):
    misc.init_distributed_mode(args)
    print('Job directory:', os.path.dirname(os.path.realpath(__file__)))
    print("Arguments:\n{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # Set seeds for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    # Set up TensorBoard logging (only on main process)
    if global_rank == 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.output_dir)
    else:
        log_writer = None

    # Data augmentation transforms
    # Define a custom RandomRotate90 transform
    class RandomRotate90:
        def __call__(self, img):
            k = np.random.randint(0, 4)
            if k == 0: return img
            return img.rotate(k * 90)

    # Transform pipeline (Cropping is now handled by Dataset if structure map is used)
    # But we still need a fallback crop in case dataset didn't crop (e.g. image too small or no structure map)
    # However, if dataset cropped, RandomCrop will just take the whole thing if size matches.
    
    transform_train = transforms.Compose([
        # transforms.RandomCrop(args.img_size), # Removed: Dataset handles cropping
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Lambda(lambda img: RandomRotate90()(img)),
        transforms.PILToTensor()
    ])

    # Choose dataset based on use_tiff flag
    if args.use_tiff:
        tiff_path = os.path.join(args.data_path, args.tiff_file)
        if os.path.isfile(tiff_path):
            # Single TIFF file
            if args.use_normalized_tiff:
                # Check for structure map
                structure_map_path = os.path.join(args.data_path, 'mt_skeletons.npy')
                if not os.path.exists(structure_map_path):
                    structure_map_path = None
                    print("Warning: Structure map not found, using random cropping without structure awareness.")
                
                # Use normalized dataset for 16-bit images
                dataset_train = TiffStackDatasetNormalized(
                    tiff_path, 
                    transform=transform_train,
                    normalize_per_image=args.normalize_per_image,
                    structure_map_path=structure_map_path,
                    crop_size=args.img_size
                )
                print(f"Using normalized TIFF dataset: {tiff_path}")
                if structure_map_path:
                    print(f"Using structure-aware cropping with map: {structure_map_path}")
                
                # Assuming normalized dataset returns 1 channel if grayscale
                # We can check dataset properties if needed, but usually it's 1 channel for microscopy
                args.in_channels = 1
                args.out_channels = 1
            else:
                # Standard TIFF dataset
                dataset_train = TiffStackDataset(tiff_path, transform=transform_train)
                print(f"Using single TIFF file: {tiff_path}")
                # TiffStackDataset might return RGB if PIL converts it, or 1 channel
                # Let's assume 1 channel for safety with microscopy data
                args.in_channels = 1
                args.out_channels = 1
        elif os.path.isdir(args.data_path):
            # Multiple TIFF files in directory
            dataset_train = MultiTiffDataset(args.data_path, transform=transform_train)
            print(f"Using multiple TIFF files from: {args.data_path}")
            args.in_channels = 1
            args.out_channels = 1
        else:
            raise ValueError(f"TIFF path not found: {tiff_path}")
        
        # Update class_num based on dataset
        args.class_num = dataset_train.num_classes
    else:
        # Original ImageNet dataset
        dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
        args.in_channels = 3
        args.out_channels = 3
    
    print(dataset_train)

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    print("Sampler_train =", sampler_train)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True
    )

    torch._dynamo.config.cache_size_limit = 128
    torch._dynamo.config.optimize_ddp = False

    # Create denoiser (UNet based)
    model = Denoiser(args)

    print("Model =", model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters: {:.6f}M".format(n_params / 1e6))

    model.to(device)

    eff_batch_size = args.batch_size * misc.get_world_size()
    if args.lr is None:  # only base_lr (blr) is specified
        args.lr = args.blr * eff_batch_size / 256

    print("Base lr: {:.2e}".format(args.lr * 256 / eff_batch_size))
    print("Actual lr: {:.2e}".format(args.lr))
    print("Effective batch size: %d" % eff_batch_size)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
    model_without_ddp = model.module

    # Set up optimizer with weight decay adjustment for bias and norm layers
    param_groups = misc.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)

    # Resume from checkpoint if provided
    checkpoint_path = os.path.join(args.resume, "checkpoint-last.pth") if args.resume else None
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model_without_ddp.load_state_dict(checkpoint['model'])

        ema_state_dict1 = checkpoint['model_ema1']
        ema_state_dict2 = checkpoint['model_ema2']
        model_without_ddp.ema_params1 = [ema_state_dict1[name].cuda() for name, _ in model_without_ddp.named_parameters()]
        model_without_ddp.ema_params2 = [ema_state_dict2[name].cuda() for name, _ in model_without_ddp.named_parameters()]
        print("Resumed checkpoint from", args.resume)

        if 'optimizer' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            print("Loaded optimizer & scaler state!")
        del checkpoint
    else:
        model_without_ddp.ema_params1 = copy.deepcopy(list(model_without_ddp.parameters()))
        model_without_ddp.ema_params2 = copy.deepcopy(list(model_without_ddp.parameters()))
        print("Training from scratch")

    # Evaluate generation
    if args.evaluate_gen:
        print("Evaluating checkpoint at {} epoch".format(args.start_epoch))
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            with torch.no_grad():
                evaluate(model_without_ddp, args, 0, batch_size=args.gen_bsz, log_writer=log_writer)
        return

    # Training loop
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(model, model_without_ddp, data_loader_train, optimizer, device, epoch, log_writer=log_writer, args=args)

        # Save checkpoint periodically
        if epoch % args.save_last_freq == 0 or epoch + 1 == args.epochs:
            misc.save_model(
                args=args,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                epoch=epoch,
                epoch_name="last"
            )

        if epoch % 1000 == 0 and epoch > 0:
            misc.save_model(
                args=args,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                epoch=epoch
            )

        # Perform online evaluation at specified intervals
        if args.online_eval and (epoch % args.eval_freq == 0 or epoch + 1 == args.epochs):
            torch.cuda.empty_cache()
            with torch.no_grad():
                evaluate(model_without_ddp, args, epoch, batch_size=args.gen_bsz, log_writer=log_writer)
            torch.cuda.empty_cache()

        if misc.is_main_process() and log_writer is not None:
            log_writer.flush()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time:', total_time_str)


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
