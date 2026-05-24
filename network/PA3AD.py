import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME
from network.Mink import Mink_unet as unet3d
import numpy as np


class PA3ADNet(nn.Module):
    def __init__(self, in_channels, out_channels, arch='MinkUNet34C'):
        super(PA3ADNet, self).__init__()

        self.normal_backbone = unet3d(in_channels=in_channels, out_channels=out_channels, arch=arch)
        self.anomaly_backbone = unet3d(in_channels=in_channels, out_channels=out_channels, arch=arch)

        self.feature_fusion = nn.Sequential(
            nn.Linear(out_channels * 2, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )

        self.difference_aware = nn.Sequential(
            nn.Linear(out_channels * 2, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.Sigmoid()
        )

        self.offset_predictor = nn.Sequential(
            nn.Linear(out_channels, out_channels, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.PReLU(),
            nn.Linear(out_channels, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.PReLU(),
            nn.Linear(64, 32, bias=False),
            nn.BatchNorm1d(32),
            nn.PReLU(),
            nn.Linear(32, 3, bias=True)
        )

        self.register_buffer('normal_prototype', torch.zeros(out_channels))
        self.register_buffer('prototype_momentum', torch.tensor(0.999))
        self.register_buffer('prototype_initialized', torch.tensor(False))

        self.register_buffer('update_count', torch.tensor(0))
        self.register_buffer('current_epoch', torch.tensor(0))
        self.momentum_schedule = {
            'initial': 0.100,
            'early': 0.500,
            'middle': 0.800,
            'late': 0.950,
            'final': 0.990,
            'bootstrap_steps': 50,
            'early_steps': 150,
            'middle_steps': 400,
            'late_steps': 600,
            'final_steps': 300
        }

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")
            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def set_current_epoch(self, epoch):
        self.current_epoch.data.fill_(epoch)

    def update_normal_prototype(self, normal_features):
        if not self.training:
            return

        self.update_count += 1
        current_momentum = self.get_adaptive_momentum()

        current_mean = torch.mean(normal_features.detach(), dim=0)

        if not self.prototype_initialized:
            self.normal_prototype.data.copy_(current_mean)
            self.prototype_initialized.data.fill_(True)
        else:
            self.normal_prototype.data.mul_(current_momentum).add_(
                current_mean, alpha=1.0 - current_momentum
            )

    def get_adaptive_momentum(self):
        epoch = self.current_epoch.item()
        schedule = self.momentum_schedule

        if epoch < schedule['bootstrap_steps']:
            return schedule['initial']
        elif epoch < schedule['bootstrap_steps'] + schedule['early_steps']:
            progress = (epoch - schedule['bootstrap_steps']) / schedule['early_steps']
            return schedule['initial'] + progress * (schedule['early'] - schedule['initial'])
        elif epoch < schedule['bootstrap_steps'] + schedule['early_steps'] + schedule['middle_steps']:
            start_epoch = schedule['bootstrap_steps'] + schedule['early_steps']
            progress = (epoch - start_epoch) / schedule['middle_steps']
            return schedule['early'] + progress * (schedule['middle'] - schedule['early'])
        elif epoch < schedule['bootstrap_steps'] + schedule['early_steps'] + schedule['middle_steps'] + schedule['late_steps']:
            start_epoch = schedule['bootstrap_steps'] + schedule['early_steps'] + schedule['middle_steps']
            progress = (epoch - start_epoch) / schedule['late_steps']
            return schedule['middle'] + progress * (schedule['late'] - schedule['middle'])
        else:
            return schedule['final']

    def get_normal_baseline(self, target_shape):
        if self.prototype_initialized:
            normal_baseline = self.normal_prototype.unsqueeze(0).expand(target_shape)
        else:
            normal_baseline = torch.zeros(target_shape, device=self.normal_prototype.device)
        return normal_baseline

    def extract_features(self, feat_voxel, xyz_voxel, v2p_index, backbone):
        cuda_cur_device = torch.cuda.current_device()
        inputs = ME.SparseTensor(feat_voxel, xyz_voxel, device=f'cuda:{cuda_cur_device}')
        voxel_feat = backbone(inputs)
        point_feat = voxel_feat.F[v2p_index]
        return point_feat

    def forward(self, normal_data, anomaly_data, batch_count):
        normal_feat = self.extract_features(
            normal_data['feat_voxel'],
            normal_data['xyz_voxel'],
            normal_data['v2p_index'],
            self.normal_backbone
        )

        anomaly_feat = self.extract_features(
            anomaly_data['feat_voxel'],
            anomaly_data['xyz_voxel'],
            anomaly_data['v2p_index'],
            self.anomaly_backbone
        )

        if self.training:
            self.update_normal_prototype(normal_feat)

        assert normal_feat.shape[0] == anomaly_feat.shape[0], \
            f"Normal and anomaly features must have same number of points: {normal_feat.shape[0]} vs {anomaly_feat.shape[0]}"

        concat_feat = torch.cat([normal_feat, anomaly_feat], dim=1)
        fused_feat = self.feature_fusion(concat_feat)
        diff_weight = self.difference_aware(concat_feat)
        enhanced_feat = fused_feat * diff_weight
        pred_offset = self.offset_predictor(enhanced_feat)

        return {
            'pred_offset': pred_offset,
            'normal_feat': normal_feat,
            'anomaly_feat': anomaly_feat,
            'fused_feat': fused_feat,
            'diff_weight': diff_weight,
            'enhanced_feat': enhanced_feat
        }

    def test_inference(self, feat_voxel, xyz_voxel, v2p_index):
        anomaly_feat = self.extract_features(feat_voxel, xyz_voxel, v2p_index, self.anomaly_backbone)
        normal_baseline = self.get_normal_baseline(anomaly_feat.shape)

        concat_feat = torch.cat([normal_baseline, anomaly_feat], dim=1)
        fused_feat = self.feature_fusion(concat_feat)
        diff_weight = self.difference_aware(concat_feat)
        enhanced_feat = fused_feat * diff_weight
        pred_offset = self.offset_predictor(enhanced_feat)

        return pred_offset


def concat_model_fn(batch, model, cfg):
    normal_data = {
        'feat_voxel': batch['normal_feat_voxel'],
        'xyz_voxel': batch['normal_xyz_voxel'],
        'v2p_index': batch['normal_v2p_index']
    }

    anomaly_data = {
        'feat_voxel': batch['anomaly_feat_voxel'],
        'xyz_voxel': batch['anomaly_xyz_voxel'],
        'v2p_index': batch['anomaly_v2p_index']
    }

    outputs = model(normal_data, anomaly_data, batch['batch_count'])

    pred_offset = outputs['pred_offset']
    gt_offsets = batch['batch_offset'].cuda()

    pt_diff = pred_offset - gt_offsets
    pt_dist = torch.sum(torch.abs(pt_diff), dim=-1)
    valid = torch.ones(pt_dist.shape[0]).cuda()
    offset_norm_loss = torch.sum(pt_dist * valid) / (torch.sum(valid) + 1e-6)

    gt_offsets_norm = torch.norm(gt_offsets, p=2, dim=1)
    gt_offsets_ = gt_offsets / (gt_offsets_norm.unsqueeze(-1) + 1e-8)
    pt_offsets_norm = torch.norm(pred_offset, p=2, dim=1)
    pt_offsets = pred_offset / (pt_offsets_norm.unsqueeze(-1) + 1e-8)
    direction_diff = -(gt_offsets_ * pt_offsets).sum(-1)
    offset_dir_loss = torch.sum(direction_diff * valid) / (torch.sum(valid) + 1e-6)

    l2_loss = torch.mean(torch.sum((pred_offset - gt_offsets) ** 2, dim=-1))

    normal_feat = outputs['normal_feat']
    anomaly_feat = outputs['anomaly_feat']
    feat_diff = torch.mean(torch.sum((normal_feat - anomaly_feat) ** 2, dim=-1))
    feature_regularization = torch.exp(-feat_diff)

    diff_weight = outputs['diff_weight']
    weight_reg = torch.mean(torch.abs(diff_weight - 0.5))

    total_loss = offset_norm_loss + offset_dir_loss + 0.5 * l2_loss + \
                 0.01 * feature_regularization + 0.001 * weight_reg

    with torch.no_grad():
        pred = {}
        visual_dict = {
            'total_loss': total_loss.item(),
            'offset_norm_loss': offset_norm_loss.item(),
            'offset_dir_loss': offset_dir_loss.item(),
            'l2_loss': l2_loss.item(),
            'feature_reg': feature_regularization.item(),
            'weight_reg': weight_reg.item()
        }

        meter_dict = {
            'total_loss': (total_loss.item(), pred_offset.shape[0]),
            'offset_norm_loss': (offset_norm_loss.item(), pred_offset.shape[0]),
            'offset_dir_loss': (offset_dir_loss.item(), pred_offset.shape[0]),
            'l2_loss': (l2_loss.item(), pred_offset.shape[0])
        }

    return total_loss, pred, visual_dict, meter_dict


def concat_eval_fn(batch, model):
    xyz_voxel = batch['xyz_voxel']
    feat_voxel = batch['feat_voxel']
    v2p_index = batch['v2p_index']

    with torch.no_grad():
        pred_offset = model.test_inference(feat_voxel, xyz_voxel, v2p_index)

    sample_score = torch.mean(torch.sum(torch.abs(pred_offset.detach().cpu()), dim=-1))
    return sample_score, pred_offset


class SharedPA3ADNet(nn.Module):
    def __init__(self, in_channels, out_channels, arch='MinkUNet34TransformerLocalGlobal'):
        super(SharedPA3ADNet, self).__init__()

        self.shared_backbone = unet3d(in_channels=in_channels, out_channels=out_channels, arch=arch)

        self.normal_adapter = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

        self.anomaly_adapter = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

        self.feature_fusion = nn.Sequential(
            nn.Linear(out_channels * 2, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )

        self.difference_aware = nn.Sequential(
            nn.Linear(out_channels * 2, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.Sigmoid()
        )

        self.offset_predictor = nn.Sequential(
            nn.Linear(out_channels, out_channels, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.PReLU(),
            nn.Linear(out_channels, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.PReLU(),
            nn.Linear(64, 32, bias=False),
            nn.BatchNorm1d(32),
            nn.PReLU(),
            nn.Linear(32, 3, bias=True)
        )

        self.register_buffer('normal_prototype', torch.zeros(out_channels))
        self.register_buffer('prototype_momentum', torch.tensor(0.999))
        self.register_buffer('prototype_initialized', torch.tensor(False))

        self.register_buffer('update_count', torch.tensor(0))
        self.register_buffer('current_epoch', torch.tensor(0))
        self.momentum_schedule = {
            'initial': 0.100,
            'early': 0.500,
            'middle': 0.800,
            'late': 0.900,
            'final': 0.990,
            'bootstrap_steps': 100,
            'early_steps': 200,
            'middle_steps': 300,
            'late_steps': 400,
            'final_steps': 500
        }

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")
            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def set_current_epoch(self, epoch):
        self.current_epoch.data.fill_(epoch)

    def update_normal_prototype(self, normal_features):
        if not self.training:
            return

        self.update_count += 1
        current_momentum = self.get_adaptive_momentum()

        current_mean = torch.mean(normal_features.detach(), dim=0)

        if not self.prototype_initialized:
            self.normal_prototype.data.copy_(current_mean)
            self.prototype_initialized.data.fill_(True)
        else:
            self.normal_prototype.data.mul_(current_momentum).add_(
                current_mean, alpha=1.0 - current_momentum
            )

    def get_adaptive_momentum(self):
        epoch = self.current_epoch.item()
        schedule = self.momentum_schedule

        if epoch < schedule['bootstrap_steps']:
            return schedule['initial']
        elif epoch < schedule['bootstrap_steps'] + schedule['early_steps']:
            progress = (epoch - schedule['bootstrap_steps']) / schedule['early_steps']
            return schedule['initial'] + progress * (schedule['early'] - schedule['initial'])
        elif epoch < schedule['bootstrap_steps'] + schedule['early_steps'] + schedule['middle_steps']:
            start_epoch = schedule['bootstrap_steps'] + schedule['early_steps']
            progress = (epoch - start_epoch) / schedule['middle_steps']
            return schedule['early'] + progress * (schedule['middle'] - schedule['early'])
        elif epoch < schedule['bootstrap_steps'] + schedule['early_steps'] + schedule['middle_steps'] + schedule['late_steps']:
            start_epoch = schedule['bootstrap_steps'] + schedule['early_steps'] + schedule['middle_steps']
            progress = (epoch - start_epoch) / schedule['late_steps']
            return schedule['middle'] + progress * (schedule['late'] - schedule['middle'])
        else:
            return schedule['final']

    def get_normal_baseline(self, target_shape):
        if self.prototype_initialized:
            normal_baseline = self.normal_prototype.unsqueeze(0).expand(target_shape)
        else:
            normal_baseline = torch.zeros(target_shape, device=self.normal_prototype.device)
        return normal_baseline

    def extract_features(self, feat_voxel, xyz_voxel, v2p_index):
        cuda_cur_device = torch.cuda.current_device()
        inputs = ME.SparseTensor(feat_voxel, xyz_voxel, device=f'cuda:{cuda_cur_device}')
        voxel_feat = self.shared_backbone(inputs)
        point_feat = voxel_feat.F[v2p_index]
        return point_feat

    def forward(self, normal_data, anomaly_data, batch_count):
        normal_base_feat = self.extract_features(
            normal_data['feat_voxel'],
            normal_data['xyz_voxel'],
            normal_data['v2p_index']
        )

        anomaly_base_feat = self.extract_features(
            anomaly_data['feat_voxel'],
            anomaly_data['xyz_voxel'],
            anomaly_data['v2p_index']
        )

        normal_feat = self.normal_adapter(normal_base_feat)
        anomaly_feat = self.anomaly_adapter(anomaly_base_feat)

        if self.training:
            self.update_normal_prototype(normal_feat)

        concat_feat = torch.cat([normal_feat, anomaly_feat], dim=1)
        fused_feat = self.feature_fusion(concat_feat)
        diff_weight = self.difference_aware(concat_feat)
        enhanced_feat = fused_feat * diff_weight
        pred_offset = self.offset_predictor(enhanced_feat)

        return {
            'pred_offset': pred_offset,
            'normal_feat': normal_feat,
            'anomaly_feat': anomaly_feat,
            'fused_feat': fused_feat,
            'diff_weight': diff_weight,
            'enhanced_feat': enhanced_feat
        }

    def test_inference(self, feat_voxel, xyz_voxel, v2p_index):
        anomaly_base_feat = self.extract_features(feat_voxel, xyz_voxel, v2p_index)
        anomaly_feat = self.anomaly_adapter(anomaly_base_feat)

        normal_baseline = self.get_normal_baseline(anomaly_feat.shape)

        concat_feat = torch.cat([normal_baseline, anomaly_feat], dim=1)
        fused_feat = self.feature_fusion(concat_feat)
        diff_weight = self.difference_aware(concat_feat)
        enhanced_feat = fused_feat * diff_weight
        pred_offset = self.offset_predictor(enhanced_feat)

        return pred_offset
