# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.metrics import OKS_SIGMA
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import RotatedTaskAlignedAssigner, TaskAlignedAssigner, dist2bbox, dist2rbox, make_anchors
from ultralytics.utils.torch_utils import autocast
from ultralytics.utils import SimpleClass, LOGGER #$#添加人员状态预测评估指标
import numpy as np #$#添加人员状态预测评估指标
try:
    from pytorch_metric_learning import miners, distances, losses, reducers
except ModuleNotFoundError:
    # 方案1 CosFace 不再使用 triplet(MetricLearningLoss); pytorch_metric_learning 改为可选(仅 DetectEmbed 变体需要)
    miners = distances = losses = reducers = None

from .metrics import bbox_iou, probiou
from .tal import bbox2dist


class MetricLearningLoss(nn.Module):
    def __init__(self):
        super(MetricLearningLoss, self).__init__()
        self.mining_func = miners.BatchEasyHardMiner(pos_strategy='hard', neg_strategy='semihard')
        self.loss_func = losses.TripletMarginLoss(margin=0.075)
        self.confidence_threshold = 1  # 1=全部前景参与embedding对比学习(原0.5为省显存只取top50%置信度样本,会系统性排除小目标;SARD稀疏场景triplet的N×N距离矩阵仅约1MB,不会OOM)

    def forward(self, embeddings, states, confidences=None, normalize=False):
        # Select only the embeddings and states for confidences on top X%
        if confidences is not None and self.confidence_threshold<1:
            top_k = int(self.confidence_threshold * len(confidences))
            _, indices = torch.topk(confidences, top_k, largest=True)
            embeddings = embeddings[indices]
            states = states[indices]
            

        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        # Sample triplets and calculate loss
        indices_tuples = self.mining_func(embeddings, states)
        loss = self.loss_func(embeddings, states, indices_tuples)
        #loss = self.loss_func(embeddings, states)
        return loss

class MetricLearningLoss1(nn.Module):
    """Metric learning loss for person re-identification using triplet loss with hard mining."""
    
    def __init__(self, margin=0.075):
        super(MetricLearningLoss1, self).__init__()
        self.margin = margin
        self.confidence_threshold = 1

    def forward(self, embeddings, states, confidences=None, normalize=False):
        """
        Args:
            embeddings: (N, embed_dim) tensor of embeddings
            states: (N,) tensor of person IDs
            confidences: (N,) tensor of detection confidences
            normalize: whether to normalize embeddings
        """
        # Select only the embeddings and states for confidences on top X%
        if confidences is not None and self.confidence_threshold < 1:
            top_k = int(self.confidence_threshold * len(confidences))
            _, indices = torch.topk(confidences, top_k, largest=True)
            embeddings = embeddings[indices]
            states = states[indices]

        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Use simplified triplet loss with hard mining
        return self._triplet_loss_hard_mining(embeddings, states)
    
    def _triplet_loss_hard_mining(self, embeddings, labels):
        """Triplet loss with hard positive and semi-hard negative mining."""
        if len(torch.unique(labels)) < 2:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
        # Compute pairwise distance matrix
        pairwise_dist = torch.cdist(embeddings, embeddings, p=2)
        
        # Get hard positives and semi-hard negatives
        losses = []
        for i, anchor_label in enumerate(labels):
            # Positive mask (same identity, excluding anchor itself)
            pos_mask = (labels == anchor_label) & (torch.arange(len(labels), device=labels.device) != i)
            # Negative mask (different identity)
            neg_mask = labels != anchor_label
            
            if not pos_mask.any() or not neg_mask.any():
                continue
                
            # Hard positive (farthest positive)
            pos_dists = pairwise_dist[i][pos_mask]
            hard_pos_dist = pos_dists.max()
            
            # Semi-hard negative (closest negative that's farther than hardest positive)
            neg_dists = pairwise_dist[i][neg_mask]
            semi_hard_negs = neg_dists[neg_dists > hard_pos_dist]
            
            if len(semi_hard_negs) > 0:
                hard_neg_dist = semi_hard_negs.min()
            else:
                # If no semi-hard negatives, use hardest negative
                hard_neg_dist = neg_dists.min()
            
            # Triplet loss
            loss = F.relu(hard_pos_dist - hard_neg_dist + self.margin)
            losses.append(loss)
        
        if len(losses) == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
        return torch.stack(losses).mean()

class VarifocalLoss(nn.Module):
    """
    Varifocal loss by Zhang et al.

    https://arxiv.org/abs/2008.13367.
    """

    def __init__(self):
        """Initialize the VarifocalLoss class."""
        super().__init__()

    @staticmethod
    def forward(pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        """Computes varfocal loss."""
        weight = alpha * pred_score.sigmoid().pow(gamma) * (1 - label) + gt_score * label
        with autocast(enabled=False):
            loss = (
                (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction="none") * weight)
                .mean(1)
                .sum()
            )
        return loss


class FocalLoss(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)."""

    def __init__(self):
        """Initializer for FocalLoss class with no parameters."""
        super().__init__()

    @staticmethod
    def forward(pred, label, gamma=1.5, alpha=0.25):
        """Calculates and updates confusion matrix for object detection/classification tasks."""
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** gamma
        loss *= modulating_factor
        if alpha > 0:
            alpha_factor = label * alpha + (1 - label) * (1 - alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()


class DFLoss(nn.Module):
    """Criterion class for computing DFL losses during training."""

    def __init__(self, reg_max=16) -> None:
        """Initialize the DFL module."""
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist, target):
        """
        Return sum of left and right DFL losses.

        Distribution Focal Loss (DFL) proposed in Generalized Focal Loss
        https://ieeexplore.ieee.org/document/9792391
        """
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)


class BboxLoss(nn.Module):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max=16):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


class RotatedBboxLoss(BboxLoss):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__(reg_max)

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = probiou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, xywh2xyxy(target_bboxes[..., :4]), self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


class KeypointLoss(nn.Module):
    """Criterion class for computing training losses."""

    def __init__(self, sigmas) -> None:
        """Initialize the KeypointLoss class."""
        super().__init__()
        self.sigmas = sigmas

    def forward(self, pred_kpts, gt_kpts, kpt_mask, area):
        """Calculates keypoint loss factor and Euclidean distance loss for predicted and actual keypoints."""
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
        # e = d / (2 * (area * self.sigmas) ** 2 + 1e-9)  # from formula
        e = d / ((2 * self.sigmas).pow(2) * (area + 1e-9) * 2)  # from cocoeval
        return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()


class v8DetectionLoss:
    """Criterion class for computing training losses."""

    def __init__(self, model, tal_topk=10):  # model must be de-paralleled
        """Initializes v8DetectionLoss with the model, defining model-related properties and BCE loss function."""
        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0) 
        self.bbox_loss = BboxLoss(m.reg_max).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            for j in range(batch_size):
                matches = i == j
                if n := matches.sum():
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        # dfl_conf = pred_distri.view(batch_size, -1, 4, self.reg_max).detach().softmax(-1)
        # dfl_conf = (dfl_conf.amax(-1).mean(-1) + dfl_conf.amax(-1).amin(-1)) / 2

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            # pred_scores.detach().sigmoid() * 0.8 + dfl_conf.unsqueeze(-1) * 0.2,
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

# DetectEmbed 类：在 cv3 分类分支的第一组卷积后旁路出一个小嵌入头，训练时通过 Triplet Loss 
# x[i] → [DWConv+Conv]₁ ──→ [DWConv+Conv]₂ → Conv2d → cls_out  (主路不变)
#                 │
#                 └──→ Conv2d(c3, 64, 1) → 64-d embed → Triplet Loss (仅训练时)
# 让中间特征更具判别性。推理时嵌入分支不参与计算，零额外开销。
class v8DetectEmbedLoss(v8DetectionLoss):
    """在标准检测损失基础上，添加分类分支中间特征的 Triplet Loss。
    
    利用 GT 类别标签作为 label，对 cv3 第一组卷积后的中间特征施加度量学习约束，
    使同类目标的中间特征更近，不同类更远，从而提升分类判别力。
    """

    def __init__(self, model, tal_topk=10):
        super().__init__(model, tal_topk)
        m = model.model[-1]  # DetectEmbed() module
        self.detect_module = m
        self.embed_dim = m.embed_dim
        self.embed_loss = MetricLearningLoss().to(self.device)

    def __call__(self, preds, batch):
        """计算 box + cls + dfl + embed 四项损失"""
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl, embed
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat(
            [xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2
        ).split((self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets（标准检测标签，不需要额外的 states 字段）
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss（和标准检测完全一样）
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        if fg_mask.sum():
            # Bbox loss（和标准检测完全一样）
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )

            # ===== 新增：嵌入 Triplet Loss =====
            embeds = self.detect_module._train_embeds
            if embeds is not None:
                # 拼接所有尺度的嵌入：(B, embed_dim, total_H*W)
                all_embeds = torch.cat(
                    [e.view(batch_size, self.embed_dim, -1) for e in embeds], 2
                )
                all_embeds = all_embeds.permute(0, 2, 1).contiguous()  # (B, total_H*W, embed_dim)
                
                # 取前景位置的嵌入
                fg_embeds = all_embeds[fg_mask]  # (num_fg, embed_dim)
                
                # 用 GT 类别标签作为 Triplet Loss 的 label
                # target_scores[fg_mask] → (num_fg, nc)，argmax 得到类别索引
                fg_labels = target_scores[fg_mask].argmax(dim=-1)  # (num_fg,)
                
                # 至少需要 2 个样本且至少 2 个不同类别才能挖掘三元组
                unique_labels = fg_labels.unique()
                if fg_embeds.shape[0] > 1 and unique_labels.shape[0] > 1:
                    loss[3] = self.embed_loss(fg_embeds, fg_labels)

        loss[0] *= self.hyp.box   # box gain
        loss[1] *= self.hyp.cls   # cls gain
        loss[2] *= self.hyp.dfl   # dfl gain
        loss[3] *= getattr(self.hyp, 'clr', 0.5)  # embed gain（可在 hyp 中调节）

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl, embed)


class v13JDELoss:
    """YOLOv13 JDE Loss for joint detection and embedding."""

    def __init__(self, model, tal_topk=10):  #tal_topk=10 model must be de-paralleled
        """Initializes v13JDELoss with the model, defining model-related properties and loss functions."""
        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # JDE() module
        m.state_grad_scale = float(getattr(h, "state_grad_scale", 1.0))  # 姿势分支→neck/backbone 共享特征的梯度缩放alpha, 从 default.yaml 注入; 0=硬切断, 1=全共享
        m.state_grad_mode = str(getattr(h, "state_grad_mode", "scale"))  # 姿态梯度回传模式: scale|gradvac|align, 从 default.yaml 注入
        m.state_grad_eps = float(getattr(h, "state_grad_eps", 0.0))  # gradvac模式的余弦下界eps, 从 default.yaml 注入; 仅gradvac模式生效
        m.pose_mode = str(getattr(h, "pose_mode", "none"))  # PFEM姿态特征增强模式: none=baseline | gate=①整层人员感知门控 | gate_roi=①门控+③ROI增强, 从 default.yaml 注入
        if getattr(m, "has_p2", False):  # PFEM sparse (JDE-PFEM yaml): 注入 ROI上限/人区域阈值
            m.max_rois = int(getattr(h, "pfem_max_rois", 20))
            m.pfem_tau = float(getattr(h, "pfem_tau", 0.3))
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        
        self.state_classes = getattr(m, 'state_classes', None) #$#embeddings预测状态
        
        self.no = m.nc + m.reg_max * 4 + m.embed_dim
        self.embed_dim = m.embed_dim    # embedding dimension
        if self.state_classes is not None: #$#embeddings预测状态
            self.no += self.state_classes #$#embeddings预测状态
        
            
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0, use_states=True) #最后添加, use_states=True#
        self.bbox_loss = BboxLoss(m.reg_max).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)
        
        # 方案1 CosFace: 去掉 triplet(embed_loss), 改用姿势原型分类; s/m 从 hyp 读(default.yaml 配置)
        self.cosface_s = getattr(h, 'cosface_s', 64.0)   # CosFace 缩放因子
        # margin: cosface_m_adaptive=True 时由 set_class_counts() 按全局类别样本数自动计算(LDAM); 否则用 cosface_m_per_class/cosface_m
        self.cosface_m_adaptive = getattr(h, 'cosface_m_adaptive', False)
        self.cosface_m_base  = float(getattr(h, 'cosface_m', 0.35))   # 多数类 margin (base)
        self.cosface_m_max   = float(getattr(h, 'cosface_m_max', 0.6))   # 最少数类 margin 上限
        self.cosface_m_power = float(getattr(h, 'cosface_m_power', 1.0))  # 自适应曲线幂次(=1线性)
        if self.state_classes is not None:
            if self.cosface_m_adaptive:
                # 自适应: 先用 base 占位, set_class_counts() 注入全局统计后覆盖
                self.cosface_m = torch.full((self.state_classes,), self.cosface_m_base, device=device, dtype=torch.float32)
                LOGGER.info(f"v13JDELoss: 自适应 CosFace margin 已启用 (base={self.cosface_m_base}, max={self.cosface_m_max}, power={self.cosface_m_power})")
            else:
                # 手动模式: 优先 cosface_m_per_class, 否则标量 cosface_m
                mpc = getattr(h, 'cosface_m_per_class', None)
                if mpc is not None:
                    assert len(mpc) == self.state_classes, f"cosface_m_per_class长度({len(mpc)})!=state_classes({self.state_classes})"
                    self.cosface_m = torch.tensor(mpc, device=device, dtype=torch.float32)
                else:
                    self.cosface_m = torch.full((self.state_classes,), self.cosface_m_base, device=device)
        else:
            self.cosface_m = None
        
        # Class-Balanced weighting for state prediction
        # state_counts_dynamic: 计数来源开关(False=静态扫描注入/现状, True=动态EMA跟踪); 与 use_cb / cosface_m_adaptive 解耦
        self.state_counts_dynamic = getattr(h, 'state_counts_dynamic', False)
        if self.state_classes is not None:
            self.cb_beta = getattr(h, 'state_cb_beta', 0.9999)  # CB 公式形状参数(动态模式兼作EMA衰减率): w_c=(1-β)/(1-β^n_c)
            self.use_cb = getattr(h, 'use_state_cb', True)
            self.class_counts_static = None  # 静态分支: 等待 set_class_counts() 注入全局统计
            self.cb_weights = torch.ones(self.state_classes, device=device, dtype=torch.float32)  # 占位, 注入/EMA更新后覆盖
            if self.state_counts_dynamic:
                # 动态 EMA 分支: 每 batch 跟踪实际采样分布(旧版风格, 适合在线过采样), 计数器初值 0
                self.class_sample_counts = torch.zeros(self.state_classes, device=device, dtype=torch.float32)

    def _recompute_cb_and_margin(self, counts):
        """从一份 counts 同时重算 cb_weights(若 use_cb) 和 cosface_m(若 cosface_m_adaptive). 静态/动态共用, 保证两模式可比."""
        counts = counts.to(device=self.cb_weights.device, dtype=torch.float32).clamp(min=1e-6)
        # Class-Balanced 权重: w_c = (1-β) / (1-β^n_c), 再按均值归一化
        if self.use_cb:
            beta = self.cb_beta
            self.cb_weights = (1.0 - beta) / (1.0 - torch.pow(beta, counts) + 1e-8)
            self.cb_weights = self.cb_weights / (self.cb_weights.mean() + 1e-8)
        # 自适应 CosFace margin: 样本越少 → margin 越大 (LDAM)
        if self.cosface_m_adaptive and self.cosface_m is not None:
            n_max, n_min = counts.max(), counts.min()
            t = (n_max - counts) / (n_max - n_min + 1e-8)  # [0,1]: 多数类=0, 最少数=1
            self.cosface_m = (self.cosface_m_base + (self.cosface_m_max - self.cosface_m_base) * t.pow(self.cosface_m_power))

    def set_class_counts(self, counts):
        """由 Trainer 在训练前注入全局类别样本数(静态统计, 一次性). 据此计算 CB 权重和自适应 margin.
        state_counts_dynamic=True(动态EMA)时直接 return, 不覆盖 loss 内部的动态计数."""
        if self.state_classes is None:
            return
        if self.state_counts_dynamic:
            LOGGER.info("v13JDELoss: state_counts_dynamic=True(动态EMA计数), 忽略静态注入")
            return
        counts = torch.as_tensor(counts, device=self.cosface_m.device, dtype=torch.float32).clamp(min=1e-6)
        self.class_counts_static = counts
        self._recompute_cb_and_margin(counts)
        LOGGER.info(f"v13JDELoss: 注入全局类别样本数 counts={[int(x) for x in counts.tolist()]} "
                    f"=> cb_weights={[round(float(x), 3) for x in self.cb_weights.tolist()]} "
                    f"=> cosface_m={[round(float(x), 3) for x in self.cosface_m.tolist()]}")

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        # Include states for person search
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out
        

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss_dim = 5 if self.state_classes is not None else 4 #$#embeddings预测状态
        loss = torch.zeros(loss_dim, device=self.device)  # box, cls, dfl, embed, state

        feats = preds[1] if isinstance(preds, tuple) else preds
        
        if self.state_classes is not None: #$#embeddings预测状态
            pred_distri, pred_scores, pred_embeds, pred_states = torch.cat( #$#embeddings预测状态
                [xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2    
            ).split((self.reg_max * 4, self.nc, self.embed_dim, self.state_classes), 1)
            pred_states = pred_states.permute(0, 2, 1).contiguous() #$#embeddings预测状态
        else: #$#embeddings预测状态
            pred_distri, pred_scores, pred_embeds = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
                (self.reg_max * 4, self.nc, self.embed_dim), 1
            )
        pred_embeds = pred_embeds.permute(0, 2, 1).contiguous()
        

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)
        #print(type(batch))  #￥#
        #print(batch.keys()) #￥#
        #for k, v in batch.items(): #￥#
        #    print(f"{k}: {type(v)} → {v.shape if hasattr(v, 'shape') else v}") #￥#

        # Targets
        if 'states' in batch: #
            targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"], batch["states"].view(-1, 1)), 1)
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes, gt_states = targets.split((1, 4, 1), 2)  # cls, xyxy, state
            
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        # dfl_conf = pred_distri.view(batch_size, -1, 4, self.reg_max).detach().softmax(-1)
        # dfl_conf = (dfl_conf.amax(-1).mean(-1) + dfl_conf.amax(-1).amin(-1)) / 2

        if 'states' in batch: #
            _, target_bboxes, target_scores, fg_mask, _, target_states = self.assigner(
                pred_scores.detach().sigmoid(),
                (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
                anchor_points * stride_tensor,
                gt_labels,
                gt_bboxes,
                mask_gt,
                gt_states
            )
        #print("$%#$#$#$#$#$#$#$#$#$#$#$#$#$",target_states)#,target_states.size()
        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        if fg_mask.sum():
            # Bbox loss
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )

            # Embedding loss
            # Select the predicted embeddings for the foreground masks and the corresponding target states
            if 'states' in batch: #￥#
                pred_embeds = pred_embeds[fg_mask]  # (batch_fg_objects, embed_dim)  保留供 last_fg_embeds 外部使用
                target_states = target_states[fg_mask]  # (batch_fg_objects, 1)
                # 方案1: 去掉 triplet(loss[3] 保持 0), 姿势改用 CosFace 原型分类(见下方 loss[4])
                # confidences = pred_scores[fg_mask].sigmoid().view(-1)
                # loss[3] = self.embed_loss(pred_embeds, target_states.squeeze(-1), confidences)
                if self.state_classes is not None: #$#embeddings预测状态
                    pred_states_fg = pred_states[fg_mask]  # 余弦相似度 (N, state_classes), 范围[-1,1]
                    target_states_fg = (target_states.squeeze(-1).long()).clamp_(min=0, max=self.state_classes-1)
                    # 动态 EMA 计数: 每 batch 统计当前前景类别分布并更新, 再重算 cb_weights/cosface_m
                    if self.state_counts_dynamic:
                        unique_cls, cls_cnt = torch.unique(target_states_fg, return_counts=True)
                        current_counts = torch.zeros(self.state_classes, device=self.device, dtype=torch.float32)
                        current_counts[unique_cls] = cls_cnt.float()
                        self.class_sample_counts = (self.cb_beta * self.class_sample_counts
                                                    + (1.0 - self.cb_beta) * current_counts)
                        self._recompute_cb_and_margin(self.class_sample_counts)
                    ############################方案1 CosFace: logits = s·cos − m·onehot(y)#####################################
                    # pred_states_fg 是 head 输出的余弦相似度; 乘缩放 s 并对正确类减 margin, 拉开 6 类姿势
                    # LDAM: 每样本取其GT类的 margin(少数类更大, 如running)
                    m_y = self.cosface_m.to(dtype)[target_states_fg]  # (N,)
                    cosface_logits = self.cosface_s * pred_states_fg - m_y.unsqueeze(1) * F.one_hot(target_states_fg, self.state_classes).to(dtype)
                    ############################Focal Loss + Class-Balanced weighting (基于 CosFace logits)#####################################
                    CE_loss = F.cross_entropy(cosface_logits, target_states_fg, reduction='none')  # (N,)
                    pred_probs = F.softmax(cosface_logits, dim=1)  # (N, state_classes)
                    p_t = pred_probs.gather(1, target_states_fg.unsqueeze(1)).squeeze(1)  # (N,) - probability of true class
                    # Apply focal weighting: (1 - p_t)^gamma
                    gamma = getattr(self.hyp, 'state_focal_gamma', 2.0)  # default gamma=2.0
                    focal_weight = (1.0 - p_t) ** gamma
                    
                    # Class-Balanced weighting: 静态全局统计, 直接查表(不再 EMA)
                    if self.use_cb:
                        sample_cb_weights = self.cb_weights[target_states_fg]  # (N,) 静态权重, set_class_counts() 预算好
                        # Combine Focal Loss and Class-Balanced weighting
                        loss[4] = (sample_cb_weights * focal_weight * CE_loss).mean()
                    else:
                        # Only Focal Loss without Class-Balanced weighting
                        loss[4] = (focal_weight * CE_loss).mean()
        
        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain
        loss[3] *= getattr(self.hyp, 'clr', 0.5)  #yolov13-xiugai不同于YOLO11-JDE的loss[3] *= self.hyp.clr# # contrastive embedding gain
        self.last_fg_embeds = pred_embeds.detach() #￥# 2025.8.15添加这一行 供人员搜索外部使用
        self.last_fg_states = target_states.detach().squeeze(-1).long() #￥# 2025.8.15添加这一行 供人员搜索外部使用
        if self.state_classes is not None: #$#embeddings预测状态
            loss[4] *= getattr(self.hyp, 'state', 1.0)  # state prediction gain #$#embeddings预测状态
        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl, embed)或loss(box, cls, dfl, embed, state)

class StateMetrics(SimpleClass):
    """
    人员状态预测评估指标类
    """
    
    def __init__(self, num_states=166):
        """初始化状态预测指标"""
        self.num_states = num_states
        self.correct = 0
        self.total = 0
        self.state_accuracy = 0.0
        self.confusion_matrix = np.zeros((num_states, num_states))
        self.per_state_accuracy = np.zeros(num_states)
        # 添加累积统计
        self.state_correct_counts = np.zeros(num_states)  # 每个状态的正确预测数
        self.state_total_counts = np.zeros(num_states)    # 每个状态的总样本数
        self.state_image_counts = np.zeros(num_states)    # 每个状态出现在多少张图像中
        self.state_images_set = [set() for _ in range(num_states)]  # 每类状态出现的图像索引集合
        # 用于输出格式化的指标
        self.p = []  # 每个状态的精度列表
        self.r = []  # 每个状态的召回率列表
        self.f1 = []  # 每个状态的F1分数列表
        self.acc = []  # 每个状态的准确率列表
        self.ap_class_index = []  # 有样本的状态类别索引

    def reset(self):
        """重置所有累加器，供每次验证开始时调用，避免跨 epoch 累积导致 accuracy/混淆矩阵失真。"""
        self.correct = 0
        self.total = 0
        self.state_accuracy = 0.0
        self.confusion_matrix = np.zeros((self.num_states, self.num_states))
        self.per_state_accuracy = np.zeros(self.num_states)
        self.state_correct_counts = np.zeros(self.num_states)  # 每个状态的正确预测数
        self.state_total_counts = np.zeros(self.num_states)    # 每个状态的总样本数
        self.state_image_counts = np.zeros(self.num_states)    # 每个状态出现在多少张图像中
        self.state_images_set = [set() for _ in range(self.num_states)]  # 每类状态出现的图像索引集合
        # 用于输出格式化的指标
        self.p = []
        self.r = []
        self.f1 = []
        self.acc = []
        self.ap_class_index = []

    def process(self, pred_states, target_states, image_indices=None):
        """
        处理状态预测结果
        Args:
            pred_states: 预测的状态概率 (N, num_states)
            target_states: 真实状态标签 (N,)
            image_indices: 图像索引列表 (N,)，用于统计每个状态出现在多少张图像中
        """
        pred_labels = pred_states.argmax(dim=1)
        correct_mask = (pred_labels == target_states)
        
        self.correct += correct_mask.sum().item()
        self.total += len(target_states)
        self.state_accuracy = self.correct / self.total if self.total > 0 else 0.0
        
        # 更新混淆矩阵
        for pred, true in zip(pred_labels.cpu().numpy(), target_states.cpu().numpy()):
            self.confusion_matrix[true, pred] += 1
        
        
        # 统计每个状态出现在哪些图像中
        if image_indices is not None:
            for idx, true_state in zip(image_indices, target_states.cpu().numpy()):
                self.state_images_set[true_state].add(idx)
        
        # 累积更新每个状态的统计信息
        for i in range(self.num_states):
            state_mask = (target_states == i)
            if state_mask.sum() > 0:
                state_correct = correct_mask[state_mask].sum().item()
                state_total = state_mask.sum().item()
                # 累积计数
                self.state_correct_counts[i] += state_correct
                self.state_total_counts[i] += state_total
                # 重新计算累积准确率
                self.per_state_accuracy[i] = self.state_correct_counts[i] / self.state_total_counts[i]
        
        # 更新图像计数
        for i in range(self.num_states):
            self.state_image_counts[i] = len(self.state_images_set[i])
    
    def get_tp_fp_fn(self):
        """
        从混淆矩阵计算每个状态的 TP、FP、FN
        Returns:
            tp: (num_states,) 每个状态的 True Positive 数量
            fp: (num_states,) 每个状态的 False Positive 数量
            fn: (num_states,) 每个状态的 False Negative 数量
        """
        tp = np.diag(self.confusion_matrix)  # 对角线元素是 TP
        fp = self.confusion_matrix.sum(axis=0) - tp  # 列和减去对角线是 FP
        fn = self.confusion_matrix.sum(axis=1) - tp  # 行和减去对角线是 FN
        return tp, fp, fn
    
    def get_precision_recall_f1(self):
        """
        计算每个状态的精度、召回率和 F1 分数
        Returns:
            precision: (num_states,) 每个状态的精度
            recall: (num_states,) 每个状态的召回率
            f1: (num_states,) 每个状态的 F1 分数
        """
        tp, fp, fn = self.get_tp_fp_fn()
        
        # 计算精度：TP / (TP + FP)
        precision = np.zeros(self.num_states)
        for i in range(self.num_states):
            if tp[i] + fp[i] > 0:
                precision[i] = tp[i] / (tp[i] + fp[i])
        
        # 计算召回率：TP / (TP + FN)
        recall = np.zeros(self.num_states)
        for i in range(self.num_states):
            if tp[i] + fn[i] > 0:
                recall[i] = tp[i] / (tp[i] + fn[i])
        
        # 计算 F1 分数：2 * Precision * Recall / (Precision + Recall)
        f1 = np.zeros(self.num_states)
        for i in range(self.num_states):
            if precision[i] + recall[i] > 0:
                f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
        
        return precision, recall, f1
    
    def update_formatted_metrics(self):
        """更新格式化指标，用于类似检测指标的输出"""
        tp, fp, fn = self.get_tp_fp_fn()
        precision, recall, f1 = self.get_precision_recall_f1()
        
        # 找到有样本的状态类别
        support = self.state_total_counts > 0
        self.ap_class_index = np.where(support)[0].tolist()
        
        # 为每个有样本的状态计算指标
        self.p = []
        self.r = []
        self.f1 = []
        self.acc = []
        
        for i in self.ap_class_index:
            self.p.append(float(precision[i]))
            self.r.append(float(recall[i]))
            self.f1.append(float(f1[i]))
            self.acc.append(float(self.per_state_accuracy[i]))
        
        # 转换为numpy数组以便计算均值
        self.p = np.array(self.p) if self.p else np.array([])
        self.r = np.array(self.r) if self.r else np.array([])
        self.f1 = np.array(self.f1) if self.f1 else np.array([])
        self.acc = np.array(self.acc) if self.acc else np.array([])
    
    def mean_results(self):
        """返回平均指标，格式类似检测指标: [mp, mr, mf1, macc, macc]"""
        if len(self.p) == 0:
            return [0.0, 0.0, 0.0, 0.0, 0.0]
        mp = float(self.p.mean()) if len(self.p) > 0 else 0.0
        mr = float(self.r.mean()) if len(self.r) > 0 else 0.0
        mf1 = float(self.f1.mean()) if len(self.f1) > 0 else 0.0
        macc = float(self.acc.mean()) if len(self.acc) > 0 else 0.0
        # 返回5个值，格式类似检测: [P, R, F1, Acc, Acc] (最后两个相同，保持格式一致)
        return [mp, mr, mf1, 0, 0] #macc, macc
    
    def class_result(self, i):
        """返回第i个状态类别的指标，格式类似检测指标: [p, r, f1, acc, acc]"""
        if i >= len(self.ap_class_index):
            return [0.0, 0.0, 0.0, 0.0, 0.0]
        idx = self.ap_class_index[i]
        p = float(self.p[i]) if i < len(self.p) else 0.0
        r = float(self.r[i]) if i < len(self.r) else 0.0
        f1 = float(self.f1[i]) if i < len(self.f1) else 0.0
        acc = float(self.acc[i]) if i < len(self.acc) else 0.0
        # 返回5个值，格式类似检测: [P, R, F1, Acc, Acc]
        return [p, r, f1, 0, 0] #acc, acc
    def state_result(self, state_idx):
        """根据状态索引直接获取该状态的指标，格式类似检测指标: [p, r, f1, acc, acc]"""
        if state_idx < 0 or state_idx >= self.num_states:
            return [0.0, 0.0, 0.0, 0.0, 0.0]
        
        # 如果该状态在ap_class_index中，使用已计算的指标
        if state_idx in self.ap_class_index:
            ap_idx = self.ap_class_index.index(state_idx)
            return self.class_result(ap_idx)
        
        # 如果该状态没有样本，返回0
        if self.state_total_counts[state_idx] == 0:
            return [0.0, 0.0, 0.0, 0.0, 0.0]
        
        # 否则，从混淆矩阵计算指标
        tp, fp, fn = self.get_tp_fp_fn()
        precision, recall, f1 = self.get_precision_recall_f1()
        
        p = float(precision[state_idx]) if state_idx < len(precision) else 0.0
        r = float(recall[state_idx]) if state_idx < len(recall) else 0.0
        f1_score = float(f1[state_idx]) if state_idx < len(f1) else 0.0
        acc = float(self.per_state_accuracy[state_idx]) if state_idx < len(self.per_state_accuracy) else 0.0
        
        return [p, r, f1_score, acc, acc]
    
    @property
    def results_dict(self):
        """返回结果字典"""
        # 使用累积的state_total_counts而不是confusion_matrix
        support = self.state_total_counts > 0  # 每个状态是否出现过
        macro = float(self.per_state_accuracy[support].mean()) if support.any() else 0.0
        
        # 计算 TP、FP、FN 和指标
        tp, fp, fn = self.get_tp_fp_fn()
        precision, recall, f1 = self.get_precision_recall_f1()
        
        # 计算宏平均（只考虑有样本的状态）
        macro_precision = float(precision[support].mean()) if support.any() else 0.0
        macro_recall = float(recall[support].mean()) if support.any() else 0.0
        macro_f1 = float(f1[support].mean()) if support.any() else 0.0
        
        # 计算微平均（所有样本）
        total_tp = tp.sum()
        total_fp = fp.sum()
        total_fn = fn.sum()
        micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0
        
        return {
            "metrics/state_accuracy": self.state_accuracy,
            "metrics/state_macro_accuracy": macro,
            "metrics/state_macro_precision": macro_precision,
            "metrics/state_macro_recall": macro_recall,
            "metrics/state_macro_f1": macro_f1,
            "metrics/state_micro_precision": micro_precision,
            "metrics/state_micro_recall": micro_recall,
            "metrics/state_micro_f1": micro_f1,
            "metrics/state_total_tp": int(total_tp),
            "metrics/state_total_fp": int(total_fp),
            "metrics/state_total_fn": int(total_fn),
        }
        
    
    @property
    def keys(self):
        """返回指标键列表"""
        return [
            "metrics/state_accuracy",
            "metrics/state_macro_accuracy",
            "metrics/state_macro_precision",
            "metrics/state_macro_recall",
            "metrics/state_macro_f1",
            "metrics/state_micro_precision",
            "metrics/state_micro_recall",
            "metrics/state_micro_f1",
            "metrics/state_total_tp",
            "metrics/state_total_fp",
            "metrics/state_total_fn",
        ]

class v8SegmentationLoss(v8DetectionLoss):
    """Criterion class for computing training losses."""

    def __init__(self, model):  # model must be de-paralleled
        """Initializes the v8SegmentationLoss class, taking a de-paralleled model as argument."""
        super().__init__(model)
        self.overlap = model.args.overlap_mask

    def __call__(self, preds, batch):
        """Calculate and return the loss for the YOLO model."""
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl
        feats, pred_masks, proto = preds if len(preds) == 3 else preds[1]
        batch_size, _, mask_h, mask_w = proto.shape  # batch size, number of masks, mask height, mask width
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_masks = pred_masks.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ segment dataset incorrectly formatted or not a segment dataset.\n"
                "This error can occur when incorrectly training a 'segment' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolov8n-seg.pt data=coco8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'segment' dataset using 'data=coco8-seg.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/segment/ for help."
            ) from e

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[2] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        if fg_mask.sum():
            # Bbox loss
            loss[0], loss[3] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
            )
            # Masks loss
            masks = batch["masks"].to(self.device).float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
                masks = F.interpolate(masks[None], (mask_h, mask_w), mode="nearest")[0]

            loss[1] = self.calculate_segmentation_loss(
                fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto, pred_masks, imgsz, self.overlap
            )

        # WARNING: lines below prevent Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.box  # seg gain
        loss[2] *= self.hyp.cls  # cls gain
        loss[3] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    @staticmethod
    def single_mask_loss(
        gt_mask: torch.Tensor, pred: torch.Tensor, proto: torch.Tensor, xyxy: torch.Tensor, area: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the instance segmentation loss for a single image.

        Args:
            gt_mask (torch.Tensor): Ground truth mask of shape (n, H, W), where n is the number of objects.
            pred (torch.Tensor): Predicted mask coefficients of shape (n, 32).
            proto (torch.Tensor): Prototype masks of shape (32, H, W).
            xyxy (torch.Tensor): Ground truth bounding boxes in xyxy format, normalized to [0, 1], of shape (n, 4).
            area (torch.Tensor): Area of each ground truth bounding box of shape (n,).

        Returns:
            (torch.Tensor): The calculated mask loss for a single image.

        Notes:
            The function uses the equation pred_mask = torch.einsum('in,nhw->ihw', pred, proto) to produce the
            predicted masks from the prototype masks and predicted mask coefficients.
        """
        pred_mask = torch.einsum("in,nhw->ihw", pred, proto)  # (n, 32) @ (32, 80, 80) -> (n, 80, 80)
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
        return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).sum()

    def calculate_segmentation_loss(
        self,
        fg_mask: torch.Tensor,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        target_bboxes: torch.Tensor,
        batch_idx: torch.Tensor,
        proto: torch.Tensor,
        pred_masks: torch.Tensor,
        imgsz: torch.Tensor,
        overlap: bool,
    ) -> torch.Tensor:
        """
        Calculate the loss for instance segmentation.

        Args:
            fg_mask (torch.Tensor): A binary tensor of shape (BS, N_anchors) indicating which anchors are positive.
            masks (torch.Tensor): Ground truth masks of shape (BS, H, W) if `overlap` is False, otherwise (BS, ?, H, W).
            target_gt_idx (torch.Tensor): Indexes of ground truth objects for each anchor of shape (BS, N_anchors).
            target_bboxes (torch.Tensor): Ground truth bounding boxes for each anchor of shape (BS, N_anchors, 4).
            batch_idx (torch.Tensor): Batch indices of shape (N_labels_in_batch, 1).
            proto (torch.Tensor): Prototype masks of shape (BS, 32, H, W).
            pred_masks (torch.Tensor): Predicted masks for each anchor of shape (BS, N_anchors, 32).
            imgsz (torch.Tensor): Size of the input image as a tensor of shape (2), i.e., (H, W).
            overlap (bool): Whether the masks in `masks` tensor overlap.

        Returns:
            (torch.Tensor): The calculated loss for instance segmentation.

        Notes:
            The batch loss can be computed for improved speed at higher memory usage.
            For example, pred_mask can be computed as follows:
                pred_mask = torch.einsum('in,nhw->ihw', pred, proto)  # (i, 32) @ (32, 160, 160) -> (i, 160, 160)
        """
        _, _, mask_h, mask_w = proto.shape
        loss = 0

        # Normalize to 0-1
        target_bboxes_normalized = target_bboxes / imgsz[[1, 0, 1, 0]]

        # Areas of target bboxes
        marea = xyxy2xywh(target_bboxes_normalized)[..., 2:].prod(2)

        # Normalize to mask size
        mxyxy = target_bboxes_normalized * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=proto.device)

        for i, single_i in enumerate(zip(fg_mask, target_gt_idx, pred_masks, proto, mxyxy, marea, masks)):
            fg_mask_i, target_gt_idx_i, pred_masks_i, proto_i, mxyxy_i, marea_i, masks_i = single_i
            if fg_mask_i.any():
                mask_idx = target_gt_idx_i[fg_mask_i]
                if overlap:
                    gt_mask = masks_i == (mask_idx + 1).view(-1, 1, 1)
                    gt_mask = gt_mask.float()
                else:
                    gt_mask = masks[batch_idx.view(-1) == i][mask_idx]

                loss += self.single_mask_loss(
                    gt_mask, pred_masks_i[fg_mask_i], proto_i, mxyxy_i[fg_mask_i], marea_i[fg_mask_i]
                )

            # WARNING: lines below prevents Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
            else:
                loss += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        return loss / fg_mask.sum()


class v8PoseLoss(v8DetectionLoss):
    """Criterion class for computing training losses."""

    def __init__(self, model):  # model must be de-paralleled
        """Initializes v8PoseLoss with model, sets keypoint variables and declares a keypoint loss instance."""
        super().__init__(model)
        self.kpt_shape = model.model[-1].kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        sigmas = torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.ones(nkpt, device=self.device) / nkpt
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)

    def __call__(self, preds, batch):
        """Calculate the total loss and detach it."""
        loss = torch.zeros(5, device=self.device)  # box, cls, dfl, kpt_location, kpt_visibility
        feats, pred_kpts = preds if isinstance(preds[0], list) else preds[1]
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        batch_size = pred_scores.shape[0]
        batch_idx = batch["batch_idx"].view(-1, 1)
        targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (b, h*w, 17, 3)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[3] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[4] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
            keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]

            loss[1], loss[2] = self.calculate_keypoints_loss(
                fg_mask, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.pose  # pose gain
        loss[2] *= self.hyp.kobj  # kobj gain
        loss[3] *= self.hyp.cls  # cls gain
        loss[4] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    @staticmethod
    def kpts_decode(anchor_points, pred_kpts):
        """Decodes predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y

    def calculate_keypoints_loss(
        self, masks, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
    ):
        """
        Calculate the keypoints loss for the model.

        This function calculates the keypoints loss and keypoints object loss for a given batch. The keypoints loss is
        based on the difference between the predicted keypoints and ground truth keypoints. The keypoints object loss is
        a binary classification loss that classifies whether a keypoint is present or not.

        Args:
            masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).
            target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
            keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
            batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
            stride_tensor (torch.Tensor): Stride tensor for anchors, shape (N_anchors, 1).
            target_bboxes (torch.Tensor): Ground truth boxes in (x1, y1, x2, y2) format, shape (BS, N_anchors, 4).
            pred_kpts (torch.Tensor): Predicted keypoints, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).

        Returns:
            kpts_loss (torch.Tensor): The keypoints loss.
            kpts_obj_loss (torch.Tensor): The keypoints object loss.
        """
        batch_idx = batch_idx.flatten()
        batch_size = len(masks)

        # Find the maximum number of keypoints in a single image
        max_kpts = torch.unique(batch_idx, return_counts=True)[1].max()

        # Create a tensor to hold batched keypoints
        batched_keypoints = torch.zeros(
            (batch_size, max_kpts, keypoints.shape[1], keypoints.shape[2]), device=keypoints.device
        )

        # TODO: any idea how to vectorize this?
        # Fill batched_keypoints with keypoints based on batch_idx
        for i in range(batch_size):
            keypoints_i = keypoints[batch_idx == i]
            batched_keypoints[i, : keypoints_i.shape[0]] = keypoints_i

        # Expand dimensions of target_gt_idx to match the shape of batched_keypoints
        target_gt_idx_expanded = target_gt_idx.unsqueeze(-1).unsqueeze(-1)

        # Use target_gt_idx_expanded to select keypoints from batched_keypoints
        selected_keypoints = batched_keypoints.gather(
            1, target_gt_idx_expanded.expand(-1, -1, keypoints.shape[1], keypoints.shape[2])
        )

        # Divide coordinates by stride
        selected_keypoints /= stride_tensor.view(1, -1, 1, 1)

        kpts_loss = 0
        kpts_obj_loss = 0

        if masks.any():
            gt_kpt = selected_keypoints[masks]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # pose loss

            if pred_kpt.shape[-1] == 3:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss

        return kpts_loss, kpts_obj_loss


class v8ClassificationLoss:
    """Criterion class for computing training losses."""

    def __call__(self, preds, batch):
        """Compute the classification loss between predictions and true labels."""
        preds = preds[1] if isinstance(preds, (list, tuple)) else preds
        loss = F.cross_entropy(preds, batch["cls"], reduction="mean")
        loss_items = loss.detach()
        return loss, loss_items


class v8OBBLoss(v8DetectionLoss):
    """Calculates losses for object detection, classification, and box distribution in rotated YOLO models."""

    def __init__(self, model):
        """Initializes v8OBBLoss with model, assigner, and rotated bbox loss; note model must be de-paralleled."""
        super().__init__(model)
        self.assigner = RotatedTaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = RotatedBboxLoss(self.reg_max).to(self.device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 6, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 6, device=self.device)
            for j in range(batch_size):
                matches = i == j
                if n := matches.sum():
                    bboxes = targets[matches, 2:]
                    bboxes[..., :4].mul_(scale_tensor)
                    out[j, :n] = torch.cat([targets[matches, 1:2], bboxes], dim=-1)
        return out

    def __call__(self, preds, batch):
        """Calculate and return the loss for the YOLO model."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats, pred_angle = preds if isinstance(preds[0], list) else preds[1]
        batch_size = pred_angle.shape[0]  # batch size, number of masks, mask height, mask width
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # b, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_angle = pred_angle.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"].view(-1, 5)), 1)
            rw, rh = targets[:, 4] * imgsz[0].item(), targets[:, 5] * imgsz[1].item()
            targets = targets[(rw >= 2) & (rh >= 2)]  # filter rboxes of tiny size to stabilize training
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 5), 2)  # cls, xywhr
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ OBB dataset incorrectly formatted or not a OBB dataset.\n"
                "This error can occur when incorrectly training a 'OBB' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolov8n-obb.pt data=dota8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'OBB' dataset using 'data=dota8.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/obb/ for help."
            ) from e

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri, pred_angle)  # xyxy, (b, h*w, 4)

        bboxes_for_assigner = pred_bboxes.clone().detach()
        # Only the first four elements need to be scaled
        bboxes_for_assigner[..., :4] *= stride_tensor
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            bboxes_for_assigner.type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes[..., :4] /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
        else:
            loss[0] += (pred_angle * 0).sum()

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    def bbox_decode(self, anchor_points, pred_dist, pred_angle):
        """
        Decode predicted object bounding box coordinates from anchor points and distribution.

        Args:
            anchor_points (torch.Tensor): Anchor points, (h*w, 2).
            pred_dist (torch.Tensor): Predicted rotated distance, (bs, h*w, 4).
            pred_angle (torch.Tensor): Predicted angle, (bs, h*w, 1).

        Returns:
            (torch.Tensor): Predicted rotated bounding boxes with angles, (bs, h*w, 5).
        """
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return torch.cat((dist2rbox(pred_dist, pred_angle, anchor_points), pred_angle), dim=-1)


class E2EDetectLoss:
    """Criterion class for computing training losses."""

    def __init__(self, model):
        """Initialize E2EDetectLoss with one-to-many and one-to-one detection losses using the provided model."""
        self.one2many = v8DetectionLoss(model, tal_topk=10)
        self.one2one = v8DetectionLoss(model, tal_topk=1)

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        preds = preds[1] if isinstance(preds, tuple) else preds
        one2many = preds["one2many"]
        loss_one2many = self.one2many(one2many, batch)
        one2one = preds["one2one"]
        loss_one2one = self.one2one(one2one, batch)
        return loss_one2many[0] + loss_one2one[0], loss_one2many[1] + loss_one2one[1]
