#!/usr/bin/env python3

"""
PA3AD Evaluation Script
"""

import os, sys
import time
import random
import torch
import torch.utils.data
import numpy as np
import torch.optim as optim
from math import cos, pi
from sklearn.metrics import roc_auc_score, average_precision_score
import logging
from tqdm import tqdm
import MinkowskiEngine as ME


# Add project root to sys path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
sys.path.append(current_dir)

from tools.log import get_logger
from config.config_PA3AD_eval import get_parser
from network.PA3AD import (
    SharedPA3ADNet, PA3ADNet, 
    concat_eval_fn
)


def verify_transformer_modules(model, backbone_arch):
    """Verify that Transformer modules are correctly loaded."""
    print("\n" + "="*60)
    print("Verifying Transformer modules...")
    print("="*60)

    transformer_modules = []
    attention_modules = []

    def find_modules(module, prefix=""):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name

            if 'transformer' in name.lower() or 'attention' in name.lower():
                if hasattr(child, 'forward'):
                    if 'transformer' in name.lower():
                        transformer_modules.append(full_name)
                    if 'attention' in name.lower():
                        attention_modules.append(full_name)

            find_modules(child, full_name)

    find_modules(model)

    print(f"Backbone arch: {backbone_arch}")
    print(f"Transformer modules found: {len(transformer_modules)}")
    print(f"Attention modules found: {len(attention_modules)}")

    if transformer_modules:
        print("\nTransformer modules:")
        for i, module in enumerate(transformer_modules, 1):
            print(f"  {i}. {module}")

    if attention_modules:
        print("\nAttention modules:")
        for i, module in enumerate(attention_modules, 1):
            print(f"  {i}. {module}")

    is_transformer = 'transformer' in backbone_arch.lower()
    has_transformer_modules = len(transformer_modules) > 0 or len(attention_modules) > 0

    if is_transformer and has_transformer_modules:
        print("\nTransformer architecture verified successfully!")
        return True
    elif is_transformer and not has_transformer_modules:
        print("\nWARNING: Transformer architecture specified but no modules found!")
        return False
    else:
        print("\nUsing traditional architecture (no Transformer)")
        return True


def load_checkpoint(model, checkpoint_path, logger):
    logger.info(f'Loading model weights: {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path)

    if 'model' in checkpoint:
        model_dict = checkpoint['model']

        import os
        file_size = os.path.getsize(checkpoint_path) / (1024 * 1024)
        logger.info(f'Model file size: {file_size:.1f}MB')
    else:
        model_dict = checkpoint
        logger.info('Direct model weights format')

    # Handle module. prefix
    for k, v in model_dict.items():
        if 'module.' in k:
            model_dict = {k[len('module.'):]: v for k, v in model_dict.items()}
        break

    missing_keys, unexpected_keys = model.load_state_dict(model_dict, strict=False)

    if missing_keys:
        logger.warning(f'Missing keys: {missing_keys[:3]}...' if len(missing_keys) > 3 else f'Missing keys: {missing_keys}')
    if unexpected_keys:
        logger.warning(f'Unexpected keys: {unexpected_keys[:3]}...' if len(unexpected_keys) > 3 else f'Unexpected keys: {unexpected_keys}')

    logger.info('Model weights loaded successfully')
    return model


def analyze_defect_performance(pred_masks, gt_masks, sample_names, logger):
    """Analyze performance for different defect types."""
    defect_performance = {}

    defect_types = ['positive', 'bulge', 'concavity', 'scratch', 'crack', 'hole', 'bending']

    for defect_type in defect_types:
        type_indices = []
        for i, name in enumerate(sample_names):
            if defect_type in name:
                type_indices.append(i)

        if len(type_indices) > 0:
            type_pred = []
            type_gt = []

            for idx in type_indices:
                if idx < len(pred_masks) and idx < len(gt_masks):
                    type_pred.extend(pred_masks[idx])
                    type_gt.extend(gt_masks[idx])

            if len(type_pred) > 0 and len(set(type_gt)) > 1:
                try:
                    auc_score = roc_auc_score(type_gt, type_pred)
                    ap_score = average_precision_score(type_gt, type_pred)
                    defect_performance[defect_type] = {
                        'auc_roc': auc_score,
                        'auc_pr': ap_score,
                        'sample_count': len(type_indices),
                        'point_count': len(type_pred)
                    }
                except Exception as e:
                    logger.warning(f"Failed to compute metrics for {defect_type}: {e}")

    return defect_performance


def PA3AD_eval(cfg):
    logger = get_logger(cfg)
    logger.info('Starting PA3AD evaluation...')
    logger.info(cfg)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model
    if hasattr(cfg, 'use_shared_backbone') and cfg.use_shared_backbone:
        model = SharedPA3ADNet(cfg.in_channels, cfg.out_channels, arch=cfg.backbone_arch)
        logger.info('Using shared backbone architecture')
    else:
        model = PA3ADNet(cfg.in_channels, cfg.out_channels, arch=cfg.backbone_arch)
        logger.info('Using dual backbone architecture')

    model = model.to(device)
    model.eval()

    # Load checkpoint
    if os.path.isabs(cfg.checkpoint_name) or cfg.checkpoint_name.startswith('./'):
        checkpoint_path = cfg.checkpoint_name
    else:
        checkpoint_path = os.path.join(cfg.logpath, cfg.checkpoint_name)

    if not os.path.exists(checkpoint_path):
        logger.error(f'Checkpoint file not found: {checkpoint_path}')
        return

    model = load_checkpoint(model, checkpoint_path, logger)

    # Create dataset
    if cfg.dataset == 'Real3D':
        from datasets.Real3D.dataset_preprocess import ContrastiveDataset
        dataset = ContrastiveDataset(cfg)
    elif cfg.dataset == 'AnomalyShapeNet':
        from datasets.AnomalyShapeNet.dataset_preprocess import ContrastiveDataset
        dataset = ContrastiveDataset(cfg)
    else:
        logger.error(f'Unsupported dataset: {cfg.dataset}')
        return

    logger.info(f'Test samples: {len(dataset.test_file_list)}')

    # Create deterministic test data loader
    test_set = list(range(len(dataset.test_file_list)))

    deterministic_test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=1,
        collate_fn=dataset.testMerge,
        num_workers=0,
        shuffle=False,
        sampler=None,
        drop_last=False,
        pin_memory=False,
        worker_init_fn=None,
        persistent_workers=False,
        generator=None
    )

    # Set GT mask path
    if cfg.dataset == 'AnomalyShapeNet':
        gt_mask_path = f'datasets/AnomalyShapeNet/dataset/pcd/{cfg.category}/GT/'
        tag = 'positive'
    elif cfg.dataset == 'Real3D':
        gt_mask_path = f'datasets/Real3D/Real3D-AD-PCD/{cfg.category}/gt/'
        tag = 'good'
    else:
        logger.error(f'Unsupported dataset: {cfg.dataset}')
        return

    def deterministic_eval_fn(batch, model):
        return concat_eval_fn(batch, model)

    # Evaluation
    label_score = []
    gt_masks = []
    pred_masks = []

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(deterministic_test_loader, desc='Evaluating')):
            torch.cuda.empty_cache()
            sample_name = batch['fn'][0].split('/')[-1].split('.')[0]

            # Load GT mask
            if tag in sample_name:
                gt_masks.append(np.zeros(batch['xyz_original'].shape[0]))
            else:
                try:
                    if cfg.dataset == 'AnomalyShapeNet':
                        gt_mask = np.loadtxt(gt_mask_path + sample_name + '.txt', delimiter=',')[:, 3:].squeeze(1)
                    elif cfg.dataset == 'Real3D':
                        gt_mask = np.loadtxt(gt_mask_path + sample_name + '.txt')[:, 3:].squeeze(1)
                    gt_masks.append(gt_mask)
                except Exception as e:
                    logger.warning(f'Failed to load GT file {sample_name}: {e}')
                    gt_masks.append(np.zeros(batch['xyz_original'].shape[0]))

            score, pred_offset = deterministic_eval_fn(batch, model)
            if score is None:
                continue

            # Point-level anomaly score
            pred_mask = pred_offset.detach().cpu().abs().sum(dim=-1).numpy()
            pred_masks.append(pred_mask)

            # Sample-level labels and scores
            if 'labels' in batch:
                label_score += list(zip(batch['labels'].numpy().tolist(), [score]))

    # Compute Object-level AUC
    if label_score:
        labels, scores = zip(*label_score)
        labels = np.array(labels)
        scores = np.array(scores)

        # Normalize sample-level scores
        if np.max(scores) > np.min(scores):
            scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))

        object_auc_roc = roc_auc_score(labels, scores)
        object_auc_pr = average_precision_score(labels, scores)

        # Compute Point-level AUC
        point_pred = np.concatenate(pred_masks, axis=0)
        point_gt = np.concatenate(gt_masks, axis=0)

        # Normalize point-level predictions
        if np.max(point_pred) > np.min(point_pred):
            point_pred = (point_pred - np.min(point_pred)) / (np.max(point_pred) - np.min(point_pred))

        point_auc_roc = roc_auc_score(point_gt, point_pred)
        point_auc_pr = average_precision_score(point_gt, point_pred)

        # Average metrics
        average_auc_roc = (object_auc_roc + point_auc_roc) / 2
        average_auc_pr = (object_auc_pr + point_auc_pr) / 2

        # Print results
        logger.info('Evaluation results:')
        logger.info(f'Object AUC-ROC: {object_auc_roc:.4f}, Point AUC-ROC: {point_auc_roc:.4f}')
        logger.info(f'Object AUC-PR: {object_auc_pr:.4f}, Point AUC-PR: {point_auc_pr:.4f}')
        logger.info(f'Average AUC-ROC: {average_auc_roc:.4f}, Average AUC-PR: {average_auc_pr:.4f}')

        # Save results
        results = {
            'object_auc_roc': object_auc_roc,
            'point_auc_roc': point_auc_roc,
            'object_auc_pr': object_auc_pr,
            'point_auc_pr': point_auc_pr,
            'average_auc_roc': average_auc_roc,
            'average_auc_pr': average_auc_pr
        }

        result_path = os.path.join(cfg.logpath, 'eval_results.txt')
        with open(result_path, 'w') as f:
            for k, v in results.items():
                f.write(f'{k}: {v:.4f}\n')
        logger.info(f'Results saved to: {result_path}')
    else:
        logger.warning('No label data found')


if __name__ == '__main__':

    cfg = get_parser()

    if not hasattr(cfg, 'checkpoint_name'):
        cfg.checkpoint_name = 'best_model.pth'

    cfg.use_shared_backbone = getattr(cfg, 'use_shared_backbone', False)

    if not hasattr(cfg, 'backbone_arch') or cfg.backbone_arch == 'MinkUNet34C':
        cfg.backbone_arch = 'MinkUNet34TransformerLocalGlobal'
        print(f"Using MinkUNet34TransformerLocalGlobal: {cfg.backbone_arch}")

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu_id

    print(f"Applying deterministic config (seed: {cfg.manual_seed})")

    import random
    import numpy as np
    random.seed(cfg.manual_seed)
    np.random.seed(cfg.manual_seed)

    torch.manual_seed(cfg.manual_seed)
    torch.cuda.manual_seed(cfg.manual_seed)
    torch.cuda.manual_seed_all(cfg.manual_seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(cfg.manual_seed)

    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

    PA3AD_eval(cfg)
