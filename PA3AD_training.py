import os, sys
import time
import random
import torch
import torch.utils.data
import numpy as np
import torch.optim as optim
from math import cos, pi
from tensorboardX import SummaryWriter
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

import tools.log as log
from config.config_PA3AD_train import get_parser
import MinkowskiEngine as ME
from network.PA3AD import concat_model_fn as model_fn, concat_eval_fn as eval_fn


def cosine_lr_after_step(optimizer, base_lr, epoch, step_epoch, total_epochs, clip=1e-6):
    if epoch < step_epoch:
        lr = base_lr
    else:
        lr = clip + 0.5 * (base_lr - clip) * (1 + cos(pi * ((epoch - step_epoch) / (total_epochs - step_epoch))))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_epoch_advanced(train_loader, model, model_fn, optimizer, epoch, max_batch_iter, cfg=None):
    model.train()

    iter_time = log.AverageMeter()
    batch_time = log.AverageMeter()
    start_time = time.time()
    data_time = log.AverageMeter()

    total_loss_meter = log.AverageMeter()
    norm_loss_meter = log.AverageMeter()
    dir_loss_meter = log.AverageMeter()
    l2_loss_meter = log.AverageMeter()

    end = time.time()
    for i, batch in enumerate(train_loader):
        data_time.update(time.time() - end)

        if (i + 1) % 5 == 0:
            torch.cuda.empty_cache()

        loss, pred, visual_dict, meter_dict = model_fn(batch, model, cfg)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss_meter.update(meter_dict['total_loss'][0], meter_dict['total_loss'][1])
        norm_loss_meter.update(meter_dict['offset_norm_loss'][0], meter_dict['offset_norm_loss'][1])
        dir_loss_meter.update(meter_dict['offset_dir_loss'][0], meter_dict['offset_dir_loss'][1])
        l2_loss_meter.update(meter_dict['l2_loss'][0], meter_dict['l2_loss'][1])

        iter_time.update(time.time() - end)

        if (i + 1) % 1 == 0:
            remain_iter = max_batch_iter * (cfg.epochs - epoch) + (max_batch_iter - i)
            remain_time = remain_iter * iter_time.avg
            t_m, t_s = divmod(remain_time, 60)
            t_h, t_m = divmod(t_m, 60)
            remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

            lr = optimizer.param_groups[0]['lr']

            print('epoch: {}/{} iter: {}/{} total: {:.4f}({:.4f}) norm: {:.4f} dir: {:.4f} l2: {:.4f} '
                  'lr: {:.6f} data: {:.2f}({:.2f}) iter: {:.2f}({:.2f}) remain: {}'.format(
                      epoch, cfg.epochs, i + 1, max_batch_iter,
                      total_loss_meter.val, total_loss_meter.avg,
                      norm_loss_meter.val, dir_loss_meter.val, l2_loss_meter.val,
                      lr, data_time.val, data_time.avg, iter_time.val, iter_time.avg, remain_time))

        end = time.time()

    batch_time.update(time.time() - start_time)

    if 'logger' in globals():
        logger.info('epoch: {}/{}, total: {:.4f}, norm: {:.4f}, dir: {:.4f}, l2: {:.4f}, time: {}s'.format(
            epoch, cfg.epochs, total_loss_meter.avg, norm_loss_meter.avg, dir_loss_meter.avg, l2_loss_meter.avg, batch_time.val))


def test_epoch_advanced(test_loader, model, eval_fn, epoch, cfg=None):
    model.eval()

    # Set GT mask path
    if cfg.dataset == 'AnomalyShapeNet':
        gt_mask_path = f'datasets/AnomalyShapeNet/dataset/pcd/{cfg.category}/GT/'
        tag = 'positive'
    elif cfg.dataset == 'Real3D':
        gt_mask_path = f'datasets/Real3D/Real3D-AD-PCD/{cfg.category}/gt/'
        tag = 'good'
    else:
        logger.error(f'Unsupported dataset: {cfg.dataset}')
        return None

    label_score = []
    gt_masks = []
    pred_masks = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating'):
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
                    if 'logger' in globals():
                        logger.warning(f'Failed to load GT file {sample_name}: {e}')
                    gt_masks.append(np.zeros(batch['xyz_original'].shape[0]))

            # Model prediction
            score, pred_offset = eval_fn(batch, model)
            pred_mask = pred_offset.detach().cpu().abs().sum(dim=-1).numpy()
            pred_masks.append(pred_mask)

            if 'labels' in batch:
                label_score += list(zip(batch['labels'].numpy().tolist(), [score]))

    # Compute metrics
    if label_score:
        labels, scores = zip(*label_score)
        labels = np.array(labels)
        scores = np.array(scores)

        if np.max(scores) > np.min(scores):
            scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))

        object_auc_roc = roc_auc_score(labels, scores)
        object_auc_pr = average_precision_score(labels, scores)

        point_pred = np.concatenate(pred_masks, axis=0)
        point_gt = np.concatenate(gt_masks, axis=0)

        if np.max(point_pred) > np.min(point_pred):
            point_pred = (point_pred - np.min(point_pred)) / (np.max(point_pred) - np.min(point_pred))

        point_auc_roc = roc_auc_score(point_gt, point_pred)
        point_auc_pr = average_precision_score(point_gt, point_pred)

        avg_auc_roc = (object_auc_roc + point_auc_roc) / 2
        avg_auc_pr = (object_auc_pr + point_auc_pr) / 2

        if 'logger' in globals():
            logger.info(f'Object AUC-ROC: {object_auc_roc:.4f}, Point AUC-ROC: {point_auc_roc:.4f}')
            logger.info(f'Average AUC-ROC: {avg_auc_roc:.4f}, Average AUC-PR: {avg_auc_pr:.4f}')

        if 'writer' in globals():
            writer.add_scalar('test/object_auc_roc', object_auc_roc, epoch)
            writer.add_scalar('test/point_auc_roc', point_auc_roc, epoch)
            writer.add_scalar('test/average_auc_roc', avg_auc_roc, epoch)

        return {
            'object_auc_roc': object_auc_roc,
            'point_auc_roc': point_auc_roc,
            'object_auc_pr': object_auc_pr,
            'point_auc_pr': point_auc_pr,
            'avg_auc_roc': avg_auc_roc,
            'avg_auc_pr': avg_auc_pr
        }
    else:
        if 'logger' in globals():
            logger.warning('No label data found')
        return None


def PA3AD_training(cfgs):
    global cfg
    cfg = cfgs

    # Enhanced anomaly generation config (currently disabled, using traditional normal displacement)
    cfg.use_enhanced_anomaly = False
    cfg.anomaly_region_size = 0.1
    cfg.anomaly_intensity_range = [0.06, 0.12]

    # Logger and TensorBoard
    global logger
    from tools.log import get_logger
    logger = get_logger(cfg)
    logger.info(cfg)

    global writer
    writer = SummaryWriter(cfg.logpath)

    logger.info(f'=> Creating model, Backbone: {cfg.backbone_arch}')

    # Select model
    if hasattr(cfg, 'use_shared_backbone') and cfg.use_shared_backbone:
        from network.PA3AD import SharedPA3ADNet as net
        logger.info('Using shared backbone')
    else:
        from network.PA3AD import PA3ADNet as net
        logger.info('Using dual backbone')

    use_cuda = torch.cuda.is_available()
    assert use_cuda
    model = net(cfg.in_channels, cfg.out_channels, arch=cfg.backbone_arch)
    model = model.cuda()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'#Total params: {total_params:,}, #Trainable: {trainable_params:,}')

    # Optimizer
    if cfg.optimizer == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                             lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'AdamW':
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=cfg.lr, betas=(0.9, 0.999), weight_decay=cfg.weight_decay)

    # Load dataset
    if cfg.dataset == 'AnomalyShapeNet':
        from datasets.AnomalyShapeNet.dataset_preprocess import ContrastiveDataset
        dataset = ContrastiveDataset(cfg)
    elif cfg.dataset == 'Real3D':
        from datasets.Real3D.dataset_preprocess import ContrastiveDataset
        dataset = ContrastiveDataset(cfg)
    else:
        logger.error('Unsupported dataset')
        return

    dataset.trainLoader()
    dataset.testLoader()

    logger.info(f'Train samples: {len(dataset.train_file_list)}, Test samples: {len(dataset.test_file_list)}')
    max_batch_iter = len(dataset.train_file_list) // cfg.batch_size

    # Restore checkpoint
    start_epoch, pretrain_file = log.checkpoint_restore(model, optimizer, cfg.logpath, pretrain_file=cfg.pretrain)
    logger.info(f'Starting from epoch {start_epoch}')

    # Best performance tracking
    best_object_auc = 0.0
    best_point_auc = 0.0
    best_average_auc = 0.0
    best_object_epoch = 0
    best_point_epoch = 0
    best_average_epoch = 0

    # Training loop
    for epoch in range(start_epoch, cfg.epochs):
        model.set_current_epoch(epoch)

        train_epoch_advanced(dataset.train_data_loader, model, model_fn, optimizer, epoch, max_batch_iter, cfg)

        # Evaluate every 5 epochs
        if epoch % 5 == 0:
            test_set = list(range(len(dataset.test_file_list)))
            deterministic_test_loader = torch.utils.data.DataLoader(
                test_set,
                batch_size=1,
                collate_fn=dataset.testMerge,
                num_workers=0,
                shuffle=False,
                drop_last=False,
                pin_memory=False
            )
            eval_results = test_epoch_advanced(deterministic_test_loader, model, eval_fn, epoch, cfg)

            if eval_results is None:
                continue

            cur_obj = eval_results['object_auc_roc']
            cur_pt = eval_results['point_auc_roc']
            cur_avg = eval_results['avg_auc_roc']

            # Save best Object AUC model
            if cur_obj > best_object_auc:
                best_object_auc = cur_obj
                best_object_epoch = epoch
                torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                            'epoch': epoch, 'object_auc_roc': cur_obj, 'point_auc_roc': cur_pt,
                            'average_auc_roc': cur_avg},
                           os.path.join(cfg.logpath, 'best_object_auc_model.pth'))

            # Save best Point AUC model
            if cur_pt > best_point_auc:
                best_point_auc = cur_pt
                best_point_epoch = epoch
                torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                            'epoch': epoch, 'object_auc_roc': cur_obj, 'point_auc_roc': cur_pt,
                            'average_auc_roc': cur_avg},
                           os.path.join(cfg.logpath, 'best_point_auc_model.pth'))

            # Save best Average AUC model
            if cur_avg > best_average_auc:
                best_average_auc = cur_avg
                best_average_epoch = epoch
                torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                            'epoch': epoch, 'object_auc_roc': cur_obj, 'point_auc_roc': cur_pt,
                            'average_auc_roc': cur_avg},
                           os.path.join(cfg.logpath, 'best_average_auc_model.pth'))

            logger.info(f'Epoch {epoch}: Obj={cur_obj:.4f}(best {best_object_auc:.4f}@{best_object_epoch}), '
                        f'Pt={cur_pt:.4f}(best {best_point_auc:.4f}@{best_point_epoch}), '
                        f'Avg={cur_avg:.4f}(best {best_average_auc:.4f}@{best_average_epoch})')

    # Final test
    logger.info('=> Final test...')
    test_set = list(range(len(dataset.test_file_list)))
    final_test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=1, collate_fn=dataset.testMerge,
        num_workers=0, shuffle=False, drop_last=False, pin_memory=False
    )
    final_results = test_epoch_advanced(final_test_loader, model, eval_fn, cfg.epochs, cfg)

    if final_results:
        logger.info(f'Final: Obj={final_results["object_auc_roc"]:.4f}, '
                    f'Pt={final_results["point_auc_roc"]:.4f}, Avg={final_results["avg_auc_roc"]:.4f}')

    # Training summary
    logger.info(f'Best Object AUC: {best_object_auc:.4f} (epoch {best_object_epoch})')
    logger.info(f'Best Point AUC: {best_point_auc:.4f} (epoch {best_point_epoch})')
    logger.info(f'Best Average AUC: {best_average_auc:.4f} (epoch {best_average_epoch})')

    writer.close()


if __name__ == '__main__':
    cfg = get_parser()
    cfg.use_shared_backbone = getattr(cfg, 'use_shared_backbone', False)

    if not hasattr(cfg, 'backbone_arch') or cfg.backbone_arch == 'MinkUNet34C':
        cfg.backbone_arch = 'MinkUNet34TransformerLocalGlobal'

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu_id

    # Deterministic config
    random.seed(cfg.manual_seed)
    np.random.seed(cfg.manual_seed)
    torch.manual_seed(cfg.manual_seed)
    torch.cuda.manual_seed_all(cfg.manual_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(cfg.manual_seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

    PA3AD_training(cfg)
