import argparse
import time
import os
import random
import numpy as np
import torch


def get_parser():
    parser = argparse.ArgumentParser(description='3D Contrastive Anomaly Detection')
    parser.add_argument('--config', type=str, default='config/config_PA3AD_train.py', help='config file')
    parser.add_argument('--gpu_id', type=str, default='0', help='gpu id')
    parser.add_argument('--manual_seed', type=int, default=42, help='random seed for reproducibility')
    parser.add_argument('--enable_deterministic', action='store_true', default=True, help='enable deterministic mode')
    parser.add_argument('--task', type=str, default='train', help='task type: train or test')

    parser.add_argument('--dataset', type=str, default='AnomalyShapeNet', 
                       choices=['AnomalyShapeNet', 'Real3D'], help='dataset name')
    parser.add_argument('--category', type=str, default='vase0', 
                       help='category name (AnomalyShapeNet: vase0, bottle, car... / Real3D: airplane, candybar, car...)')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size for contrastive learning')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--step_epoch', type=int, default=20, help='learning rate decay step')
    parser.add_argument('--save_freq', type=int, default=100, help='save frequency')

    parser.add_argument('--contrast_temperature', type=float, default=0.1, help='temperature for contrastive loss')
    parser.add_argument('--contrast_weight', type=float, default=0.1, help='weight for contrastive loss')
    parser.add_argument('--classification_weight', type=float, default=0.1, help='weight for classification loss')
    parser.add_argument('--feature_diff_weight', type=float, default=0.05, help='weight for feature difference loss')

    # Network
    parser.add_argument('--backbone_arch', type=str, default='MinkUNet34TransformerLocalGlobal', help='backbone architecture')
    parser.add_argument('--in_channels', type=int, default=3, help='input channels')
    parser.add_argument('--out_channels', type=int, default=32, help='output channels')
    parser.add_argument('--use_shared_backbone', action='store_true', help='use shared backbone instead of dual backbone')

    # Data
    parser.add_argument('--voxel_size', type=float, default=0.05, help='voxel size')
    parser.add_argument('--mask_num', type=int, default=50, help='number of masks (AnomalyShapeNet: 50, Real3D: 64)')
    parser.add_argument('--data_repeat', type=int, default=100, help='data repeat times')
    parser.add_argument('--num_works', type=int, default=4, help='number of workers')

    # Optimizer
    parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer type')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')

    # LR scheduler
    parser.add_argument('--scheduler_type', type=str, default='cosine', help='scheduler type')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='warmup epochs')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='minimum learning rate')
    parser.add_argument('--warmup_start_lr', type=float, default=1e-5, help='warmup start learning rate')

    # Loss
    parser.add_argument('--focal_alpha', type=float, default=0.75, help='focal loss alpha')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='focal loss gamma')
    parser.add_argument('--use_focal_loss', action='store_true', help='use focal loss')
    parser.add_argument('--hard_negative_mining', action='store_true', help='use hard negative mining')
    parser.add_argument('--negative_ratio', type=float, default=3.0, help='negative to positive ratio')

    # Logging
    parser.add_argument('--logpath', type=str, default='./log/contrastive_exp/', help='log path')
    parser.add_argument('--pretrain', type=str, default='', help='pretrain model path')

    args = parser.parse_args()

    if args.enable_deterministic:
        set_deterministic_config(args.manual_seed)

    # Create log directory
    current_time = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    args.logpath = os.path.join(args.logpath, f'{args.category}_{current_time}/')

    # Ensure logpath ends with '/' for checkpoint function
    if not args.logpath.endswith('/'):
        args.logpath += '/'

    print(f"Training config summary:")
    print(f"  Seed: {args.manual_seed}")
    print(f"  Deterministic: {args.enable_deterministic}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Category: {args.category}")
    print(f"  Backbone: {args.backbone_arch}")
    print(f"  Shared Backbone: {args.use_shared_backbone}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Mask Num: {args.mask_num}")
    print(f"  LR: {args.lr}")

    return args


def set_deterministic_config(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)

    print("Deterministic config applied.")
    return True
