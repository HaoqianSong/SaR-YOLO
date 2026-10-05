# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Model head modules."""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_

from ultralytics.utils.tal import TORCH_1_10, dist2bbox, dist2rbox, make_anchors
from torchvision.ops import roi_align as _roi_align

from .block import DFL, BNContrastiveHead, ContrastiveHead, Proto
from .conv import Conv, DWConv
from .transformer import MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer
from .utils import bias_init_with_prob, linear_init

__all__ = "Detect", "Segment", "Pose", "Classify", "OBB", "RTDETRDecoder", "v10Detect", "JDE", "DetectEmbed"


class GradScale(torch.autograd.Function):
    """梯度缩放: 前向 identity, 反向把回传梯度乘以 alpha.

    用于 JDE head 的 cv4 输入, 缩放姿势/状态分支流向共享 neck/backbone 特征的梯度:
    alpha=0 等价 detach (检测严格无损, 姿势上限受限); alpha=1 等价全共享 (现状);
    0<alpha<1 为折中. cv4 与 pose_proto 处于 GradScale 下游, 仍由 state loss 正常更新,
    只有传回 x[i]/neck/backbone 的那条梯度边被缩放.
    """

    @staticmethod
    def forward(ctx, x, alpha):
        """Identity forward; stash alpha for backward."""
        ctx.alpha = float(alpha)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        """Scale gradient by alpha; alpha 本身不需要梯度."""
        return grad * ctx.alpha, None


class PoseGradSurgery(torch.autograd.Function):
    """特征级姿态梯度方向手术: 在 neck 输出 x[i] 分叉点, 一次 backward 同时拿到
    检测梯度 grad_det (box+cls 经 cv2/cv3) 和姿态梯度 grad_pose (state 经 cv4),
    按 mode 对 grad_pose 做方向处理, 保护检测主任务 (检测梯度原样保留).

    forward: x → (x_det, x_pose) 分叉, 前向两路都等于 x (数值不变);
    backward: (grad_det, grad_pose) → grad_det + 处理后的 grad_pose, 合并回传给 neck.

    mode=1 (gradvac, 软投影): 保证 cos(grad_det, grad_pose) ≥ eps.
        冲突时 (cos<eps) 把 grad_pose 的反向平行分量削减到使新 cos=eps, 垂直分量始终保留;
        eps=0 退化为硬投影 (grad_pose 投影到检测法平面, 即非对称 PCGrad);
        eps>0 更严格 (连轻微同向也拉到 eps); eps<0 更宽松 (允许一定反向).
    mode=2 (align, 只同向): 只保留 grad_pose 沿 grad_det 同向的分量, 切掉垂直+反向.
        最激进, 姿态梯度被强制对齐检测方向.

    全局视角: 把 (B,C,H,W) 张量当作一个高维向量算内积/余弦. 一次 backward,
    AMP/DDP/梯度累积均无额外复杂度 (对比标准 PCGrad 的两次 backward).
    """

    @staticmethod
    def forward(ctx, x, mode, eps):
        """分叉前向; stash mode/eps 供 backward."""
        ctx.mode = int(mode)
        ctx.eps = float(eps)
        return x, x.view_as(x)  # 两路输出, 前向数值不变

    @staticmethod
    def backward(ctx, grad_det, grad_pose):
        """grad_det=检测梯度 (原样保留), grad_pose=姿态梯度 (按 mode 处理)."""
        dot = (grad_det * grad_pose).sum()  # <g_det, g_pose>
        nd = (grad_det * grad_det).sum().clamp_min(1e-12)  # |g_det|^2
        if ctx.mode == 1:  # gradvac: 软投影, 保证 cos >= eps
            np_ = (grad_pose * grad_pose).sum().clamp_min(1e-12)  # |g_pose|^2
            cos = dot / (nd.sqrt() * np_.sqrt())
            if cos < ctx.eps:  # 冲突超阈值才动
                g_perp = grad_pose - (dot / nd) * grad_det  # 垂直分量 (始终保留)
                norm_perp = g_perp.norm()
                # 调整平行分量, 使新 cos(g_det, g_pose_new) = eps, 保持 g_perp 不变
                denom = max(1.0 - ctx.eps ** 2, 1e-12) ** 0.5  # eps 是 float, 用 python 标量运算
                a = ctx.eps * norm_perp / denom
                grad_pose = a * (grad_det / nd.sqrt()) + g_perp
        else:  # align: 只保留与检测同向的分量
            coef = (dot / nd).clamp_min(0.0)  # 反向 (dot<0) 截到 0
            grad_pose = coef * grad_det
        return grad_det + grad_pose, None, None  # 检测原样 + 处理后姿态; mode/eps 无梯度


class Detect(nn.Module):
    """YOLO Detect head for detection models."""

    dynamic = False  # force grid reconstruction
    export = False  # export mode
    format = None  # export format
    end2end = False  # end2end
    max_det = 300  # max_det
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init
    legacy = False  # backward compatibility for v3/v5/v8/v9 models

    def __init__(self, nc=80, ch=()):
        """Initializes the YOLO detection layer with specified number of classes and channels."""
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = (
            nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
            if self.legacy
            else nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                    nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                    nn.Conv2d(c3, self.nc, 1),
                )
                for x in ch
            )
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        if self.end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        if self.end2end:
            return self.forward_end2end(x)

        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:  # Training path
            return x
        y = self._inference(x)
        return y if self.export else (y, x)

    def forward_end2end(self, x):
        """
        Performs forward pass of the v10Detect module.

        Args:
            x (tensor): Input tensor.

        Returns:
            (dict, tensor): If not in training mode, returns a dictionary containing the outputs of both one2many and one2one detections.
                           If in training mode, returns a dictionary containing the outputs of one2many and one2one detections separately.
        """
        x_detach = [xi.detach() for xi in x]
        one2one = [
            torch.cat((self.one2one_cv2[i](x_detach[i]), self.one2one_cv3[i](x_detach[i])), 1) for i in range(self.nl)
        ]
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:  # Training path
            return {"one2many": x, "one2one": one2one}

        y = self._inference(one2one)
        y = self.postprocess(y.permute(0, 2, 1), self.max_det, self.nc)
        return y if self.export else (y, {"one2many": x, "one2one": one2one})

    def _inference(self, x):
        """Decode predicted bounding boxes and class probabilities based on multiple-level feature maps."""
        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.format != "imx" and (self.dynamic or self.shape != shape):
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in {"saved_model", "pb", "tflite", "edgetpu", "tfjs"}:  # avoid TF FlexSplitV ops
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        if self.export and self.format in {"tflite", "edgetpu"}:
            # Precompute normalization factor to increase numerical stability
            # See https://github.com/ultralytics/ultralytics/issues/7371
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        elif self.export and self.format == "imx":
            dbox = self.decode_bboxes(
                self.dfl(box) * self.strides, self.anchors.unsqueeze(0) * self.strides, xywh=False
            )
            return dbox.transpose(1, 2), cls.sigmoid().permute(0, 2, 1)
        else:
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        return torch.cat((dbox, cls.sigmoid()), 1)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)
        if self.end2end:
            for a, b, s in zip(m.one2one_cv2, m.one2one_cv3, m.stride):  # from
                a[-1].bias.data[:] = 1.0  # box
                b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)

    def decode_bboxes(self, bboxes, anchors, xywh=True):
        """Decode bounding boxes."""
        return dist2bbox(bboxes, anchors, xywh=xywh and (not self.end2end), dim=1)

    @staticmethod
    def postprocess(preds: torch.Tensor, max_det: int, nc: int = 80):
        """
        Post-processes YOLO model predictions.

        Args:
            preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc) with last dimension
                format [x, y, w, h, class_probs].
            max_det (int): Maximum detections per image.
            nc (int, optional): Number of classes. Default: 80.

        Returns:
            (torch.Tensor): Processed predictions with shape (batch_size, min(max_det, num_anchors), 6) and last
                dimension format [x, y, w, h, max_class_prob, class_index].
        """
        batch_size, anchors, _ = preds.shape  # i.e. shape(16,8400,84)
        boxes, scores = preds.split([4, nc], dim=-1)
        index = scores.amax(dim=-1).topk(min(max_det, anchors))[1].unsqueeze(-1)
        boxes = boxes.gather(dim=1, index=index.repeat(1, 1, 4))
        scores = scores.gather(dim=1, index=index.repeat(1, 1, nc))
        scores, index = scores.flatten(1).topk(min(max_det, anchors))
        i = torch.arange(batch_size)[..., None]  # batch indices
        return torch.cat([boxes[i, index // nc], scores[..., None], (index % nc)[..., None].float()], dim=-1)

# DetectEmbed 类：在 cv3 分类分支的第一组卷积后旁路出一个小嵌入头，训练时通过 Triplet Loss 
# 让中间特征更具判别性。推理时嵌入分支不参与计算，零额外开销。
class DetectEmbed(Detect):
    """Detect head with discriminative embedding learning in classification branch.
    x[i] → [DWConv+Conv]₁ ──→ [DWConv+Conv]₂ → Conv2d → cls_out  (主路不变)
                 │
                 └──→ Conv2d(c3, 64, 1) → 64-d embed → Triplet Loss (仅训练时)
    在 cv3 分类分支的第一组卷积后旁路出一个小嵌入头，训练时通过 Triplet Loss 
    让中间特征更具判别性。推理时嵌入分支不参与计算，零额外开销。
    """

    def __init__(self, nc=80, embed_dim=64, ch=()):
        """初始化，embed_dim 是旁路嵌入的维度（默认64）"""
        super().__init__(nc, ch)
        self.embed_dim = embed_dim
        
        # 嵌入投影层：从 cv3 第一组输出的 c3 通道投影到 embed_dim
        # c3 的计算方式和父类 Detect 一致
        c3 = max(ch[0], min(self.nc, 100))
        self.cv3_embed = nn.ModuleList(
            nn.Conv2d(c3, embed_dim, 1) for _ in ch  # 每个尺度一个 1×1 投影
        )
        self._train_embeds = None  # 训练时存储嵌入，供 Loss 函数使用

    def forward(self, x):
        """前向传播：主路不变，训练时额外提取嵌入特征"""
        if self.end2end:
            return self.forward_end2end(x)

        embeds = [] if self.training else None
        for i in range(self.nl):
            # ===== cv3 分支拆成两段 =====
            # 第一段：第一组 (DWConv+Conv 或 Conv，取决于 legacy)
            intermediate = self.cv3[i][0](x[i])  # (B, c3, H, W)
            
            # 训练时：旁路提取 64-d 嵌入
            if self.training:
                embed = self.cv3_embed[i](intermediate)  # (B, embed_dim, H, W)
                embeds.append(embed)
            
            # 第二段：第二组 + 输出层（和原来的 cv3 完全一样）
            cls_out = self.cv3[i][2](self.cv3[i][1](intermediate))  # (B, nc, H, W)
            
            # cv2 框回归分支不变
            x[i] = torch.cat((self.cv2[i](x[i]), cls_out), 1)

        if self.training:
            self._train_embeds = embeds  # 存储给 Loss 函数
            return x

        # 推理路径：和标准 Detect 完全相同
        y = self._inference(x)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize biases and clear non-leaf tensors from init forward pass."""
        super().bias_init()
        # 清除初始化前向传播时产生的非叶子张量，
        # 避免 ModelEMA 的 deepcopy 失败
        self._train_embeds = None




class ROIFusion(nn.Module):
    """PFEM ROI 融合 (Person-aware Feature Enhancement Module 的局部高分辨率增强核心).

    用 P3 深层语义生成门控 M_guide, 筛选 backbone-P2 的高频细节, 残差融合:
        M_guide  = σ(W2 · δ(W1 · RoI_P3.detach()))                        # P3语义筛P2细节
        F_fusion = RoI_P3.detach() + (p2_proj(RoI_P2.detach()) ⊙ M_guide)  # 残差融合
    输入: roi_p3, roi_p2 均为 ROIAlign 后 (N, C, 7, 7); p2_proj 把 P2 通道对齐到检测层通道.
    梯度: RoI 输入 detach 挡 backbone; W1/W2/p2_proj 由 state loss 正常训练.
    """

    def __init__(self, channels, p2_channels, reduction=4):
        """channels: 检测层(P3/P4/P5)通道; p2_channels: backbone层2(P2)通道; reduction: MLP瓶颈比."""
        super().__init__()
        r = max(channels // reduction, 8)
        self.W1 = nn.Conv2d(channels, r, 1)  # 通道压缩
        self.W2 = nn.Conv2d(r, channels, 1)  # 通道扩展
        self.p2_proj = nn.Conv2d(p2_channels, channels, 1)  # P2通道对齐到检测层
        self.act = nn.ReLU(inplace=True)

    def forward(self, roi_p3, roi_p2):
        """返回 F_fusion (N, C, 7, 7), 通道同 RoI_P3, 供 cv4 共享卷积."""
        roi_p3_d = roi_p3.detach()
        m = torch.sigmoid(self.W2(self.act(self.W1(roi_p3_d))))  # M_guide (N,C,7,7)
        return roi_p3_d + self.p2_proj(roi_p2.detach()) * m


# 在文件末尾添加以下代码
class JDE(Detect):
    """YOLOv13 JDE head for joint detection and embedding models."""
    # # 类级默认值: 兼容旧 checkpoint(用更早的 JDE 代码训练, pickle 还原后实例 __dict__ 缺这些后加的配置属性时回退到 baseline 行为)
    # has_p2 = False  # PFEM/P2 高分辨率源开关; False=baseline 无 P2 增强
    # state_grad_mode = "scale"  # 姿态梯度回传模式; scale=原标量缩放(配合 state_grad_scale=1.0 即训练时原始行为)
    # state_grad_scale = 1.0  # scale 模式梯度缩放; 1.0=梯度全共享(原行为)
    # state_grad_eps = 0.0  # gradvac 模式余弦下界(仅 state_grad_mode=gradvac 生效)
    # pose_mode = "none"  # PFEM 姿态特征增强模式; none=baseline 无增强(不访问 pfem_alpha/pfem)
    # state_classes = None  # state 分类数; None=无 state(走无 state 分支); 新 checkpoint 实例值会覆盖

    def __init__(self, nc=80, embed_dim=128, state_classes=None, ch=()):
        """Initialize the JDE model attributes such as the number of classes, embedding dimension, and the convolution.
        PFEM: ch 末尾可选 backbone-P2 高分辨率源(约定 from=[P3,P4,P5,P2] 时 len(ch)==4),
              前 self.nl 个是检测层, 最后一个是 P2 源(仅 sparse 模式用).
        """
        # PFEM: 分离检测层 ch 与可选的 backbone-P2 高分辨率源
        self.has_p2 = isinstance(ch, (list, tuple)) and len(ch) == 4
        p2_channels = ch[-1] if self.has_p2 else 0
        det_ch = list(ch[:-1]) if self.has_p2 else list(ch)
        super().__init__(nc, det_ch)  # cv2/cv3 只建在检测层, self.nl=len(det_ch)=3
        self.embed_dim = embed_dim  # embedding dimension
        self.state_classes = state_classes #$#embeddings预测状态
        self.state_grad_scale = 1.0  # 姿势分支→neck/backbone 梯度缩放alpha; 运行时由 v13JDELoss 从 default.yaml 注入; 1.0=全共享, 0=硬切断
        self.state_grad_mode = "scale"  # 姿态梯度回传模式: scale=现有标量缩放(state_grad_scale) | gradvac=软投影保证cos≥eps(eps=0即硬投影) | align=只保留与检测同向分量; 运行时由 v13JDELoss 从 default.yaml 注入
        self.state_grad_eps = 0.0  # gradvac 模式的余弦下界 eps; eps=0=硬投影(非对称PCGrad); >0更严格; <0更宽松; 仅 gradvac 模式生效
        # PFEM: 人员感知特征增强 (Person-aware Feature Enhancement Module)
        self.pose_mode = "none"  # none=baseline无增强 | gate=①整层人员感知门控F'=F+α(A⊙F) | gate_roi=①门控+③ROI增强(增量加到embed_out, cv4后端) | gate_roi_front=①门控+③ROI增强(增量加到F_pose_i, cv4前端, 经完整cv4三段); 运行时由 v13JDELoss 从 default.yaml 注入

        self.no = nc + self.reg_max * 4 + embed_dim     # number of outputs per anchor
        # TODO: maybe min as a bottleneck?
        c4 = max(det_ch[0] // 4, self.embed_dim)
        ##整体cv4表达式self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.embed_dim, 1)) for x in det_ch)   
        self.cv4_1 = nn.ModuleList(Conv(x, c4, 3) for x in det_ch)                    # 第1个3×3: C_i→c4
        self.cv4_2 = nn.ModuleList(Conv(c4, c4, 3) for _ in det_ch)                   # 第2个3×3: c4→c4
        self.cv4_3 = nn.ModuleList(nn.Conv2d(c4, self.embed_dim, 1) for _ in det_ch)  # 1×1: c4→embed_dim
        self.pfem_alpha = nn.Parameter(torch.ones(self.nl))  # PFEM 整层门控残差强度 α(每层独立可学习, 初值1.0); F'=F+α(A⊙F), A=σ(cls)人员响应图
        if state_classes is not None: #$#embeddings预测状态
            self.no += state_classes #$#embeddings预测状态
            # 方案1 CosFace: 可学习姿势原型矩阵 (state_classes × embed_dim), 替代原 MLP state_predictor
            # 跨 batch 持久; 配合 L2归一化 + 余弦 + margin 实现 6 类姿势原型分类(不依赖 batch 内同类, 避开 triplet 退化)
            self.pose_proto = nn.Parameter(torch.empty(state_classes, embed_dim))
            nn.init.orthogonal_(self.pose_proto)  # 正交初始化, 利于初始类别分开
        # PFEM ROI 融合 (③局部高分辨率增强, sparse模式用): 每层独立 ROIFusion(P3语义筛 backbone-P2 高频)
        #... | gate_roi_2cv4=共享cv4的ROI增强 | gate_roi_2cv4_indep=ROI独立cv4(全图cv4与ROI cv4解耦)
        if self.has_p2:
            self.pfem = nn.ModuleList(ROIFusion(det_ch[i], p2_channels) for i in range(self.nl))   # 前端cv4_1用(F_pose_i, det_ch通道)
            self.pfem_c4 = nn.ModuleList(ROIFusion(c4, p2_channels) for _ in range(self.nl))         # cv4_2/cv4_3前用(f1/f2, c4通道)
            # ROI分支独立cv4 (gate_roi_2cv4_indep用): 与cv4_1/2/3同结构、权重独立, 专处理F_fusion→embed
            self.cv4_roi_1 = nn.ModuleList(Conv(x, c4, 3) for x in det_ch)
            self.cv4_roi_2 = nn.ModuleList(Conv(c4, c4, 3) for _ in det_ch)
            self.cv4_roi_3 = nn.ModuleList(nn.Conv2d(c4, self.embed_dim, 1) for _ in det_ch)            
            self.max_rois = 20  # ROI数量上限(按cls置信度top-N), 密集人群算力可控; 运行时由 v13JDELoss 注入
            self.pfem_tau = 0.3  # 人区域cls阈值(推理筛选预测框); 运行时由 v13JDELoss 注入

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        if self.end2end: #yolov13-xiugai多于YOLO11-JDE#
            return self.forward_end2end(x) #yolov13-xiugai多于YOLO11-JDE#

        # PFEM: 分离 backbone-P2 高分辨率源 (has_p2 时 x 末尾是 P2, 不参与检测层循环, 仅供 sparse ROI增强)
        p2_feat = x[-1] if self.has_p2 else None
        x = list(x[:-1]) if self.has_p2 else x  # 前 self.nl 个为检测层

        for i in range(self.nl):
            if self.state_classes is not None: #$#embeddings预测状态
                # 梯度处理: 统一得到 x_det_i(检测路) 与 x_pose_i(姿态路)
                if self.state_grad_mode == "scale":  # 现有标量缩放, 零改动 (数值与改造前完全一致)
                    x_det_i = x[i]
                    x_pose_i = GradScale.apply(x[i], self.state_grad_scale) #$#embeddings预测状态
                else:  # gradvac / align: neck 输出 x[i] 分叉, 一次backward同时取检测/姿态梯度做方向手术
                    mode_int = 1 if self.state_grad_mode == "gradvac" else 2
                    x_det_i, x_pose_i = PoseGradSurgery.apply(x[i], mode_int, self.state_grad_eps)
                cls_i = self.cv3[i](x_det_i)  # 复用检测cls分支(单类即人员响应源), 避免重复计算 #$#embeddings预测状态
                if self.pose_mode == "gate":
                    # PFEM ①整层人员感知门控: A=人员响应图(σ(cls), 多类amax聚合), 残差增强 F'=F+α(A⊙F); detach挡姿态梯度回cv3
                    A_i = self._person_response(cls_i).detach()  # (B,1,H,W) 单通道, 广播到C通道
                    F_pose_i = x_pose_i + self.pfem_alpha[i] * (A_i * x_pose_i)
                    ##整体cv4表达式embed_out = self.cv4[i](F_pose_i) #$#embeddings预测状态
                    embed_out = self.cv4_3[i](self.cv4_2[i](self.cv4_1[i](F_pose_i)))   #……#拆分cv4三段表达式
                elif self.pose_mode == "gate_roi_2cv4":
                    # PFEM ①整层门控 + 全图cv4 (保持anchor对齐, loss不改), 再叠加 ③ROI局部高分辨率增强
                    A_i = self._person_response(cls_i).detach()
                    F_pose_i = x_pose_i + self.pfem_alpha[i] * (A_i * x_pose_i)
                    ##整体cv4表达式embed_out = self.cv4[i](F_pose_i) #$#embeddings预测状态
                    embed_out = self.cv4_3[i](self.cv4_2[i](self.cv4_1[i](F_pose_i)))  #……#拆分cv4三段表达式
                    # ③PFEM ROI局部高分辨率增强: 人区域anchor → ROIAlign(P3门控特征+backbone-P2) → M_guide融合 → cv4 → scatter增量
                    if self.has_p2 and p2_feat is not None:
                        embed_out = self._pfem_roi_enhance(i, F_pose_i, cls_i, p2_feat, self.pfem[i], embed_out) #$#embeddings预测状态
                elif self.pose_mode == "gate_roi_2cv4-13":
                    # PFEM ①整层门控 + 全图cv4 (保持anchor对齐, loss不改), 再叠加 ③ROI局部高分辨率增强
                    A_i = self._person_response(cls_i).detach()
                    F_pose_i = x_pose_i + self.pfem_alpha[i] * (A_i * x_pose_i)
                    ##整体cv4表达式embed_out = self.cv4[i](F_pose_i) #$#embeddings预测状态
                    embed_out = self.cv4_3[i](self.cv4_1[i](F_pose_i))  #……#拆分cv4三段表达式
                    # ③PFEM ROI局部高分辨率增强: 人区域anchor → ROIAlign(P3门控特征+backbone-P2) → M_guide融合 → cv4 → scatter增量
                    if self.has_p2 and p2_feat is not None:
                        cv4_roi = (self.cv4_1, self.cv4_3)
                        embed_out = self._pfem_roi_enhance(i, F_pose_i, cls_i, p2_feat, self.pfem[i], embed_out, cv4_roi) #$#embeddings预测状态
                elif self.pose_mode == "gate_roi_2cv4_indep":
                    # PFEM ①门控 + 全图cv4_main + ③ROI增强(独立cv4_roi): ROI分支用独立cv4_roi产embed增量, 两路cv4解耦
                    A_i = self._person_response(cls_i).detach()
                    F_pose_i = x_pose_i + self.pfem_alpha[i] * (A_i * x_pose_i)
                    embed_out = self.cv4_3[i](self.cv4_2[i](self.cv4_1[i](F_pose_i)))  # 全图 cv4_main
                    if self.has_p2 and p2_feat is not None:
                        cv4_roi = (self.cv4_roi_1, self.cv4_roi_2, self.cv4_roi_3)
                        embed_out = self._pfem_roi_enhance(i, F_pose_i, cls_i, p2_feat, self.pfem[i], embed_out, cv4_roi)  # ROI 独立 cv4_roi                
                elif self.pose_mode == "gate_roi_2cv4-13_indep":
                    # PFEM ①门控 + 全图cv4_main + ③ROI增强(独立cv4_roi): ROI分支用独立cv4_roi产embed增量, 两路cv4解耦
                    A_i = self._person_response(cls_i).detach()
                    F_pose_i = x_pose_i + self.pfem_alpha[i] * (A_i * x_pose_i)
                    embed_out = self.cv4_3[i](self.cv4_1[i](F_pose_i))  # 全图 cv4_main
                    if self.has_p2 and p2_feat is not None:
                        cv4_roi = (self.cv4_roi_1, self.cv4_roi_3)
                        embed_out = self._pfem_roi_enhance(i, F_pose_i, cls_i, p2_feat, self.pfem[i], embed_out, cv4_roi)  # ROI 独立 cv4_roi
                elif self.pose_mode == "gate_roi_cv4_1":
                    # PFEM ①门控 + ③ROI局部高分辨率增强(cv4前端版): 增量作用在F_pose_i(cv4_1前, det_ch空间),
                    # 加到F_pose_i后走完整cv4_1→cv4_2→cv4_3 → embed_out; ROIAlign源=F_pose_i, ROIFusion用det_ch[i]
                    A_i = self._person_response(cls_i).detach()
                    F_pose_i = x_pose_i + self.pfem_alpha[i] * (A_i * x_pose_i)
                    if self.has_p2 and p2_feat is not None:
                        F_pose_i = self._pfem_roi_enhance(i, F_pose_i, cls_i, p2_feat, self.pfem[i])  # cv4_1前: det_ch, 用pfem # 增量加到F_pose_i(det_ch), 后续经完整cv4
                    embed_out = self.cv4_3[i](self.cv4_2[i](self.cv4_1[i](F_pose_i)))  # 完整cv4三段 #$#embeddings预测状态
                elif self.pose_mode == "gate_roi_cv4_2":
                    # PFEM ①门控 + ③ROI局部高分辨率增强(cv4_2前端版): 增量作用在f1(cv4_2前, c4特征空间),
                    # 加到f1后走完整cv4_2→cv4_3 → embed_out; ROIAlign源=f1, ROIFusion用det_ch[i]
                    A_i = self._person_response(cls_i).detach()
                    F_pose_i = x_pose_i + self.pfem_alpha[i] * (A_i * x_pose_i)
                    f1 = self.cv4_1[i](F_pose_i)
                    if self.has_p2 and p2_feat is not None:
                        f1 = self._pfem_roi_enhance(i, f1, cls_i, p2_feat, self.pfem_c4[i])              # cv4_2前: c4, 用pfem_c4
                    embed_out = self.cv4_3[i](self.cv4_2[i](f1))  # 完整cv4三段 #$#embeddings预测状态
                elif self.pose_mode == "gate_roi_cv4_3":
                    # PFEM ①整层门控(cv4_1前) + cv4_1/cv4_2, 再在 cv4_3前 叠加 ③ROI局部高分辨率增强(作用在c4特征f2)
                    A_i = self._person_response(cls_i).detach()
                    F_pose_i = x_pose_i + self.pfem_alpha[i] * (A_i * x_pose_i)
                    f2 = self.cv4_2[i](self.cv4_1[i](F_pose_i))  # cv4前两段 → c4特征 (B,c4,H,W)
                    # ③PFEM ROI局部高分辨率增强: 人区域anchor → ROIAlign(f2+backbone-P2) → M_guide融合 → scatter增量回f2
                    if self.has_p2 and p2_feat is not None:
                        f2 = self._pfem_roi_enhance(i, f2, cls_i, p2_feat, self.pfem_c4[i])              # cv4_3前: c4, 用pfem_c4
                    embed_out = self.cv4_3[i](f2)  # 1×1: c4→embed_dim #$#embeddings预测状态
                else:  # "none": 无门控 baseline(原v13行为), 用于消融对比①的价值
                    ##整体cv4表达式embed_out = self.cv4[i](x_pose_i) #$#embeddings预测状态
                    embed_out = self.cv4_3[i](self.cv4_2[i](self.cv4_1[i](x_pose_i)))   #……#拆分cv4三段表达式
                b, c, h, w = embed_out.shape #$#embeddings预测状态
                embed_flat = embed_out.view(b, c, -1).permute(0, 2, 1)  # (B, H*W, embed_dim) #$#embeddings预测状态
                # 方案1 CosFace: embedding 与原型都 L2 归一化后算余弦相似度 (剥离外观强度, 只留姿势方向)
                emb_norm = F.normalize(embed_flat, dim=-1)           # (B, H*W, embed_dim)
                proto_norm = F.normalize(self.pose_proto, dim=-1)    # (state_classes, embed_dim)
                state_out = emb_norm @ proto_norm.t()                # (B, H*W, state_classes) 余弦相似度[-1,1]
                state_out = state_out.permute(0, 2, 1).view(b, self.state_classes, h, w)  # 恢复空间维度 #$#embeddings预测状态
                x[i] = torch.cat((self.cv2[i](x_det_i), cls_i, embed_out, state_out), 1) #$#embeddings预测状态 复用cls_i
            else:
                ##整体cv4表达式x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i]), self.cv4[i](x[i])), 1) #$#embeddings预测状态
                x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i]), self.cv4_3[i](self.cv4_2[i](self.cv4_1[i](x[i])))), 1)  #……#拆分cv4三段表达式
        
        if self.training:  # Training path
            return x
        
        y = self._inference(x)
        return y if self.export else (y, x)

    @staticmethod
    def _person_response(cls_i):
        """人员响应图 (B,1,H,W): σ(cls); nc>1(多姿态类)时 amax 聚合(任一姿态类高=有人), nc=1 时直接用."""
        a = cls_i.sigmoid()
        return a.amax(dim=1, keepdim=True) if a.shape[1] > 1 else a


    def _pfem_roi_enhance(self, i, feat, cls_i, p2_feat, roifusion, embed_dense=None, cv4_tuple=None):
        """PFEM ③ROI局部高分辨率增强(通用版): 在 cv4_1/cv4_2/cv4_3 任意位置前做 ROI 增强, 增量加到 feat 并 scatter 回原位置.
        embed_dense: 不为None时走embed_out端模式(ROIAlign源=feat/F_pose_i, F_fusion过cv4, scatter到embed_dense);
                     None时走前端/中端模式(ROIAlign源=feat, F_fusion直接mean, scatter回feat). 此时feat同时是作用对象+ROIAlign源+scatter目标.
          - embed_dense=None：feat = 作用对象（ROIAlign源 + scatter目标）
          - embed_dense≠None：feat = F_pose_i（仅 ROIAlign 源，scatter 目标是 embed_dense）

        统一 _pfem_roi_enhance_cv4_1(前端 F_pose_i/det_ch) 与 _pfem_roi_enhance_cv4_3(中端 f2/f1, c4):
        结构完全相同(F_fusion 直接 mean, 不过整个cv4), 仅通道与变量名不同 → 通过 feat(作用对象) +
        roifusion(与 feat 通道匹配的 ROIFusion 模块) 参数化. 增量经后续 cv4 段投影成 embed.

        Args:
            i: 检测层索引 (0=P3,1=P4,2=P5)
            feat: 作用对象 (B,C,H,W), C=det_ch(F_pose_i@cv4_1前) 或 c4(f1@cv4_2前 / f2@cv4_3前)
            cls_i: 检测cls (B,1,H,W), 用于选人区域
            p2_feat: backbone-P2 高分辨率特征 (B,C_p2,H_p2,W_p2)
            roifusion: 与 feat 通道匹配的 ROIFusion 模块 (self.pfem[i]@det_ch 或 self.pfem_c4[i]@c4)
        Returns:
            feat + ROI增量 (B,C,H,W)
        """
        if embed_dense is not None:
            B, embed_dim, H, W = embed_dense.shape
            device = embed_dense.device
            stride_i = float(self.stride[i])
            if stride_i <= 0:  # model 构建期算 stride 时 self.stride 尚为 0, 跳过 ROI 增强(避免 1/0=inf 导致 roi_align 爆显存)
                return embed_dense
        else:
            B, C, H, W = feat.shape
            device = feat.device
            stride_i = float(self.stride[i])
            if stride_i <= 0:  # model 构建期算 stride 时 self.stride 尚为 0, 跳过 ROI 增强(避免 1/0=inf 导致 roi_align 爆显存)
                return feat
        # 1. 选人区域: 每图 cls top-N (max_rois) 的 anchor
        cls_prob = self._person_response(cls_i).flatten(2).squeeze(1)  # (B, H*W) 单通道人员响应
        N = min(int(self.max_rois), H * W)
        _, topk_idx = cls_prob.topk(N, dim=1)  # (B, N)
        ay = (topk_idx // W).reshape(-1).clamp(0, H - 1)  # (B*N,) anchor 行
        ax = (topk_idx % W).reshape(-1).clamp(0, W - 1)   # (B*N,) anchor 列
        bi = torch.arange(B, device=device).view(B, 1).expand(B, N).reshape(-1)  # (B*N,) batch idx
        # # 加置信度排查
        # cls_prob = self._person_response(cls_i).flatten(2).squeeze(1)  # (B, H*W) 单通道人员响应
        # N = min(int(self.max_rois), H * W)
        # val, topk_idx = cls_prob.topk(N, dim=1)                             # (B, N) top-N 候选(含置信度)
        # keep = val >= float(getattr(self, "pfem_tau", 0.0))                 # (B, N) 阈值掩码
        # topk_idx = topk_idx[keep]                                           # (K,) 1D anchor 索引(布尔掩码自动拍平)
        # bi = torch.arange(B, device=device).view(B, 1).expand(B, N)[keep]   # (K,) batch idx ← 必须同步[keep]过滤!
        # if topk_idx.numel() == 0:                                           # 空图/P5无信号 → 不增强
        #     return embed_dense if embed_dense is not None else feat
        # ay = (topk_idx // W).clamp(0, H - 1)  # (K,) anchor 行  ← reshape(-1)此处是空操作,保留无害
        # ax = (topk_idx % W).clamp(0, W - 1)   # (K,) anchor 列
        # 2. anchor 中心 → 固定 4×stride 感受野框 (原图坐标, 覆盖 anchor 邻域供 ROIAlign 取 P2 高频)
        cx = (ax.float() + 0.5) * stride_i
        cy = (ay.float() + 0.5) * stride_i
        half = 2.0 * stride_i    #￥#￥#￥#￥#固定4×stride的感受野
        rois = torch.stack([bi.float(), cx - half, cy - half, cx + half, cy + half], dim=1)  # (B*N, 5)
        # 3. ROIAlign: feat(任意通道C) + backbone-P2; sampling_ratio=2 固定(自适应模式在边界框会算出巨大采样数致OOM)
        roi_feat = _roi_align(feat, rois, spatial_scale=1.0 / stride_i, output_size=7, sampling_ratio=2, aligned=True)
        roi_p2 = _roi_align(p2_feat, rois, spatial_scale=0.25, output_size=7, sampling_ratio=2, aligned=True)  # P2 stride/4
        # 4. M_guide融合(通道匹配的roifusion, C语义筛P2细节) → 每 ROI 一个 C 维向量 (不过cv4, 增量经后续 cv4 段投影成 embed)
        F_fusion = roifusion(roi_feat, roi_p2)  # (N, C, 7, 7)
        if embed_dense is not None:
            if cv4_tuple is not None:
                if len(cv4_tuple) == 3:
                    cv41, cv42, cv43 = cv4_tuple  
                    embed_roi_vec = cv43[i](cv42[i](cv41[i](F_fusion))).mean(dim=(2, 3))  #……#拆分cv4三段表达式
                elif len(cv4_tuple) == 2:
                    cv41, cv43 = cv4_tuple  
                    embed_roi_vec = cv43[i](cv41[i](F_fusion)).mean(dim=(2, 3))  #……#拆分cv4三段表达式
            else:
                cv41, cv42, cv43 = self.cv4_1, self.cv4_2, self.cv4_3
                embed_roi_vec = cv43[i](cv42[i](cv41[i](F_fusion))).mean(dim=(2, 3))  #……#拆分cv4三段表达式
            
            # 5. scatter 增量回 anchor 位置 (index_add 累加, 同 anchor 多 ROI 叠加)
            increment = torch.zeros_like(embed_dense)
            inc_flat = increment.view(B, embed_dim, H * W)
        else:
            roi_vec = F_fusion.mean(dim=(2, 3))  # (B*N, C)
            # 5. scatter 增量回 anchor 位置 (index_add 累加, 同 anchor 多 ROI 叠加)
            increment = torch.zeros_like(feat)
            inc_flat = increment.view(B, C, H * W)
        for b in range(B):
            mask = bi == b
            if mask.any():
                hw = ay[mask] * W + ax[mask]  # (n_b,) within-image 索引
                if embed_dense is not None:
                    inc_flat[b].index_add_(1, hw, embed_roi_vec[mask].t())  # (embed_dim, HW)
                else:
                    inc_flat[b].index_add_(1, hw, roi_vec[mask].t())  # (C, HW)
        if embed_dense is not None:
            return embed_dense + increment
        else:
            return feat + increment

        
    def _inference(self, x):
        """Decode predicted bounding boxes and class probabilities based on multiple-level feature maps."""
        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in {"saved_model", "pb", "tflite", "edgetpu", "tfjs"}:  # avoid TF FlexSplitV ops
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4: self.reg_max * 4 + self.nc]
            if self.state_classes is not None: #$#embeddings预测状态
                emb = x_cat[:, self.reg_max * 4 + self.nc: self.reg_max * 4 + self.nc + self.embed_dim] #$#embeddings预测状态
                state = x_cat[:, self.reg_max * 4 + self.nc + self.embed_dim:] #$#embeddings预测状态
            else:
                emb = x_cat[:, self.reg_max * 4 + self.nc:]
        else:
            if self.state_classes is not None: #$#embeddings预测状态
                box, cls, emb, state = x_cat.split((self.reg_max * 4, self.nc, self.embed_dim, self.state_classes), 1) #$#embeddings预测状态
            else:
                box, cls, emb = x_cat.split((self.reg_max * 4, self.nc, self.embed_dim), 1) #$#embeddings预测状态

        if self.export and self.format in {"tflite", "edgetpu"}:
            # Precompute normalization factor to increase numerical stability
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        else:
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides
        if self.state_classes is not None: #$#embeddings预测状态
            return torch.cat((dbox, cls.sigmoid(), emb, state), 1)  # 方案1 CosFace: state 已是余弦相似度[-1,1], 无需激活, argmax 取姿势类; cls 仍 sigmoid(单类"人")
        else:
            return torch.cat((dbox, cls.sigmoid(), emb), 1) #$#embeddings预测状态 添加state.sigmoid()

                
class Segment(Detect):
    """YOLO Segment head for segmentation models."""

    def __init__(self, nc=80, nm=32, npr=256, ch=()):
        """Initialize the YOLO model attributes such as the number of masks, prototypes, and the convolution layers."""
        super().__init__(nc, ch)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos

        c4 = max(ch[0] // 4, self.nm)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1)) for x in ch)

    def forward(self, x):
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        p = self.proto(x[0])  # mask protos
        bs = p.shape[0]  # batch size

        mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)  # mask coefficients
        x = Detect.forward(self, x)
        if self.training:
            return x, mc, p
        return (torch.cat([x, mc], 1), p) if self.export else (torch.cat([x[0], mc], 1), (x[1], mc, p))


class OBB(Detect):
    """YOLO OBB detection head for detection with rotation models."""

    def __init__(self, nc=80, ne=1, ch=()):
        """Initialize OBB with number of classes `nc` and layer channels `ch`."""
        super().__init__(nc, ch)
        self.ne = ne  # number of extra parameters

        c4 = max(ch[0] // 4, self.ne)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.ne, 1)) for x in ch)

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        bs = x[0].shape[0]  # batch size
        angle = torch.cat([self.cv4[i](x[i]).view(bs, self.ne, -1) for i in range(self.nl)], 2)  # OBB theta logits
        # NOTE: set `angle` as an attribute so that `decode_bboxes` could use it.
        angle = (angle.sigmoid() - 0.25) * math.pi  # [-pi/4, 3pi/4]
        # angle = angle.sigmoid() * math.pi / 2  # [0, pi/2]
        if not self.training:
            self.angle = angle
        x = Detect.forward(self, x)
        if self.training:
            return x, angle
        return torch.cat([x, angle], 1) if self.export else (torch.cat([x[0], angle], 1), (x[1], angle))

    def decode_bboxes(self, bboxes, anchors):
        """Decode rotated bounding boxes."""
        return dist2rbox(bboxes, self.angle, anchors, dim=1)


class Pose(Detect):
    """YOLO Pose head for keypoints models."""

    def __init__(self, nc=80, kpt_shape=(17, 3), ch=()):
        """Initialize YOLO network with default parameters and Convolutional Layers."""
        super().__init__(nc, ch)
        self.kpt_shape = kpt_shape  # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)
        self.nk = kpt_shape[0] * kpt_shape[1]  # number of keypoints total

        c4 = max(ch[0] // 4, self.nk)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nk, 1)) for x in ch)

    def forward(self, x):
        """Perform forward pass through YOLO model and return predictions."""
        bs = x[0].shape[0]  # batch size
        kpt = torch.cat([self.cv4[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], -1)  # (bs, 17*3, h*w)
        x = Detect.forward(self, x)
        if self.training:
            return x, kpt
        pred_kpt = self.kpts_decode(bs, kpt)
        return torch.cat([x, pred_kpt], 1) if self.export else (torch.cat([x[0], pred_kpt], 1), (x[1], kpt))

    def kpts_decode(self, bs, kpts):
        """Decodes keypoints."""
        ndim = self.kpt_shape[1]
        if self.export:
            if self.format in {
                "tflite",
                "edgetpu",
            }:  # required for TFLite export to avoid 'PLACEHOLDER_FOR_GREATER_OP_CODES' bug
                # Precompute normalization factor to increase numerical stability
                y = kpts.view(bs, *self.kpt_shape, -1)
                grid_h, grid_w = self.shape[2], self.shape[3]
                grid_size = torch.tensor([grid_w, grid_h], device=y.device).reshape(1, 2, 1)
                norm = self.strides / (self.stride[0] * grid_size)
                a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * norm
            else:
                # NCNN fix
                y = kpts.view(bs, *self.kpt_shape, -1)
                a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * self.strides
            if ndim == 3:
                a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
            return a.view(bs, self.nk, -1)
        else:
            y = kpts.clone()
            if ndim == 3:
                y[:, 2::3] = y[:, 2::3].sigmoid()  # sigmoid (WARNING: inplace .sigmoid_() Apple MPS bug)
            y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (self.anchors[0] - 0.5)) * self.strides
            y[:, 1::ndim] = (y[:, 1::ndim] * 2.0 + (self.anchors[1] - 0.5)) * self.strides
            return y


class Classify(nn.Module):
    """YOLO classification head, i.e. x(b,c1,20,20) to x(b,c2)."""

    export = False  # export mode

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        """Initializes YOLO classification head to transform input tensor from (b,c1,20,20) to (b,c2) shape."""
        super().__init__()
        c_ = 1280  # efficientnet_b0 size
        self.conv = Conv(c1, c_, k, s, p, g)
        self.pool = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)
        self.drop = nn.Dropout(p=0.0, inplace=True)
        self.linear = nn.Linear(c_, c2)  # to x(b,c2)

    def forward(self, x):
        """Performs a forward pass of the YOLO model on input image data."""
        if isinstance(x, list):
            x = torch.cat(x, 1)
        x = self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
        if self.training:
            return x
        y = x.softmax(1)  # get final output
        return y if self.export else (y, x)


class WorldDetect(Detect):
    """Head for integrating YOLO detection models with semantic understanding from text embeddings."""

    def __init__(self, nc=80, embed=512, with_bn=False, ch=()):
        """Initialize YOLO detection layer with nc classes and layer channels ch."""
        super().__init__(nc, ch)
        c3 = max(ch[0], min(self.nc, 100))
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, embed, 1)) for x in ch)
        self.cv4 = nn.ModuleList(BNContrastiveHead(embed) if with_bn else ContrastiveHead() for _ in ch)

    def forward(self, x, text):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv4[i](self.cv3[i](x[i]), text)), 1)
        if self.training:
            return x

        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.nc + self.reg_max * 4, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in {"saved_model", "pb", "tflite", "edgetpu", "tfjs"}:  # avoid TF FlexSplitV ops
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        if self.export and self.format in {"tflite", "edgetpu"}:
            # Precompute normalization factor to increase numerical stability
            # See https://github.com/ultralytics/ultralytics/issues/7371
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        else:
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            # b[-1].bias.data[:] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)


class RTDETRDecoder(nn.Module):
    """
    Real-Time Deformable Transformer Decoder (RTDETRDecoder) module for object detection.

    This decoder module utilizes Transformer architecture along with deformable convolutions to predict bounding boxes
    and class labels for objects in an image. It integrates features from multiple layers and runs through a series of
    Transformer decoder layers to output the final predictions.
    """

    export = False  # export mode

    def __init__(
        self,
        nc=80,
        ch=(512, 1024, 2048),
        hd=256,  # hidden dim
        nq=300,  # num queries
        ndp=4,  # num decoder points
        nh=8,  # num head
        ndl=6,  # num decoder layers
        d_ffn=1024,  # dim of feedforward
        dropout=0.0,
        act=nn.ReLU(),
        eval_idx=-1,
        # Training args
        nd=100,  # num denoising
        label_noise_ratio=0.5,
        box_noise_scale=1.0,
        learnt_init_query=False,
    ):
        """
        Initializes the RTDETRDecoder module with the given parameters.

        Args:
            nc (int): Number of classes. Default is 80.
            ch (tuple): Channels in the backbone feature maps. Default is (512, 1024, 2048).
            hd (int): Dimension of hidden layers. Default is 256.
            nq (int): Number of query points. Default is 300.
            ndp (int): Number of decoder points. Default is 4.
            nh (int): Number of heads in multi-head attention. Default is 8.
            ndl (int): Number of decoder layers. Default is 6.
            d_ffn (int): Dimension of the feed-forward networks. Default is 1024.
            dropout (float): Dropout rate. Default is 0.
            act (nn.Module): Activation function. Default is nn.ReLU.
            eval_idx (int): Evaluation index. Default is -1.
            nd (int): Number of denoising. Default is 100.
            label_noise_ratio (float): Label noise ratio. Default is 0.5.
            box_noise_scale (float): Box noise scale. Default is 1.0.
            learnt_init_query (bool): Whether to learn initial query embeddings. Default is False.
        """
        super().__init__()
        self.hidden_dim = hd
        self.nhead = nh
        self.nl = len(ch)  # num level
        self.nc = nc
        self.num_queries = nq
        self.num_decoder_layers = ndl

        # Backbone feature projection
        self.input_proj = nn.ModuleList(nn.Sequential(nn.Conv2d(x, hd, 1, bias=False), nn.BatchNorm2d(hd)) for x in ch)
        # NOTE: simplified version but it's not consistent with .pt weights.
        # self.input_proj = nn.ModuleList(Conv(x, hd, act=False) for x in ch)

        # Transformer module
        decoder_layer = DeformableTransformerDecoderLayer(hd, nh, d_ffn, dropout, act, self.nl, ndp)
        self.decoder = DeformableTransformerDecoder(hd, decoder_layer, ndl, eval_idx)

        # Denoising part
        self.denoising_class_embed = nn.Embedding(nc, hd)
        self.num_denoising = nd
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # Decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(nq, hd)
        self.query_pos_head = MLP(4, 2 * hd, hd, num_layers=2)

        # Encoder head
        self.enc_output = nn.Sequential(nn.Linear(hd, hd), nn.LayerNorm(hd))
        self.enc_score_head = nn.Linear(hd, nc)
        self.enc_bbox_head = MLP(hd, hd, 4, num_layers=3)

        # Decoder head
        self.dec_score_head = nn.ModuleList([nn.Linear(hd, nc) for _ in range(ndl)])
        self.dec_bbox_head = nn.ModuleList([MLP(hd, hd, 4, num_layers=3) for _ in range(ndl)])

        self._reset_parameters()

    def forward(self, x, batch=None):
        """Runs the forward pass of the module, returning bounding box and classification scores for the input."""
        from ultralytics.models.utils.ops import get_cdn_group

        # Input projection and embedding
        feats, shapes = self._get_encoder_input(x)

        # Prepare denoising training
        dn_embed, dn_bbox, attn_mask, dn_meta = get_cdn_group(
            batch,
            self.nc,
            self.num_queries,
            self.denoising_class_embed.weight,
            self.num_denoising,
            self.label_noise_ratio,
            self.box_noise_scale,
            self.training,
        )

        embed, refer_bbox, enc_bboxes, enc_scores = self._get_decoder_input(feats, shapes, dn_embed, dn_bbox)

        # Decoder
        dec_bboxes, dec_scores = self.decoder(
            embed,
            refer_bbox,
            feats,
            shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask,
        )
        x = dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta
        if self.training:
            return x
        # (bs, 300, 4+nc)
        y = torch.cat((dec_bboxes.squeeze(0), dec_scores.squeeze(0).sigmoid()), -1)
        return y if self.export else (y, x)

    def _generate_anchors(self, shapes, grid_size=0.05, dtype=torch.float32, device="cpu", eps=1e-2):
        """Generates anchor bounding boxes for given shapes with specific grid size and validates them."""
        anchors = []
        for i, (h, w) in enumerate(shapes):
            sy = torch.arange(end=h, dtype=dtype, device=device)
            sx = torch.arange(end=w, dtype=dtype, device=device)
            grid_y, grid_x = torch.meshgrid(sy, sx, indexing="ij") if TORCH_1_10 else torch.meshgrid(sy, sx)
            grid_xy = torch.stack([grid_x, grid_y], -1)  # (h, w, 2)

            valid_WH = torch.tensor([w, h], dtype=dtype, device=device)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH  # (1, h, w, 2)
            wh = torch.ones_like(grid_xy, dtype=dtype, device=device) * grid_size * (2.0**i)
            anchors.append(torch.cat([grid_xy, wh], -1).view(-1, h * w, 4))  # (1, h*w, 4)

        anchors = torch.cat(anchors, 1)  # (1, h*w*nl, 4)
        valid_mask = ((anchors > eps) & (anchors < 1 - eps)).all(-1, keepdim=True)  # 1, h*w*nl, 1
        anchors = torch.log(anchors / (1 - anchors))
        anchors = anchors.masked_fill(~valid_mask, float("inf"))
        return anchors, valid_mask

    def _get_encoder_input(self, x):
        """Processes and returns encoder inputs by getting projection features from input and concatenating them."""
        # Get projection features
        x = [self.input_proj[i](feat) for i, feat in enumerate(x)]
        # Get encoder inputs
        feats = []
        shapes = []
        for feat in x:
            h, w = feat.shape[2:]
            # [b, c, h, w] -> [b, h*w, c]
            feats.append(feat.flatten(2).permute(0, 2, 1))
            # [nl, 2]
            shapes.append([h, w])

        # [b, h*w, c]
        feats = torch.cat(feats, 1)
        return feats, shapes

    def _get_decoder_input(self, feats, shapes, dn_embed=None, dn_bbox=None):
        """Generates and prepares the input required for the decoder from the provided features and shapes."""
        bs = feats.shape[0]
        # Prepare input for decoder
        anchors, valid_mask = self._generate_anchors(shapes, dtype=feats.dtype, device=feats.device)
        features = self.enc_output(valid_mask * feats)  # bs, h*w, 256

        enc_outputs_scores = self.enc_score_head(features)  # (bs, h*w, nc)

        # Query selection
        # (bs, num_queries)
        topk_ind = torch.topk(enc_outputs_scores.max(-1).values, self.num_queries, dim=1).indices.view(-1)
        # (bs, num_queries)
        batch_ind = torch.arange(end=bs, dtype=topk_ind.dtype).unsqueeze(-1).repeat(1, self.num_queries).view(-1)

        # (bs, num_queries, 256)
        top_k_features = features[batch_ind, topk_ind].view(bs, self.num_queries, -1)
        # (bs, num_queries, 4)
        top_k_anchors = anchors[:, topk_ind].view(bs, self.num_queries, -1)

        # Dynamic anchors + static content
        refer_bbox = self.enc_bbox_head(top_k_features) + top_k_anchors

        enc_bboxes = refer_bbox.sigmoid()
        if dn_bbox is not None:
            refer_bbox = torch.cat([dn_bbox, refer_bbox], 1)
        enc_scores = enc_outputs_scores[batch_ind, topk_ind].view(bs, self.num_queries, -1)

        embeddings = self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1) if self.learnt_init_query else top_k_features
        if self.training:
            refer_bbox = refer_bbox.detach()
            if not self.learnt_init_query:
                embeddings = embeddings.detach()
        if dn_embed is not None:
            embeddings = torch.cat([dn_embed, embeddings], 1)

        return embeddings, refer_bbox, enc_bboxes, enc_scores

    # TODO
    def _reset_parameters(self):
        """Initializes or resets the parameters of the model's various components with predefined weights and biases."""
        # Class and bbox head init
        bias_cls = bias_init_with_prob(0.01) / 80 * self.nc
        # NOTE: the weight initialization in `linear_init` would cause NaN when training with custom datasets.
        # linear_init(self.enc_score_head)
        constant_(self.enc_score_head.bias, bias_cls)
        constant_(self.enc_bbox_head.layers[-1].weight, 0.0)
        constant_(self.enc_bbox_head.layers[-1].bias, 0.0)
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            # linear_init(cls_)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight, 0.0)
            constant_(reg_.layers[-1].bias, 0.0)

        linear_init(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for layer in self.input_proj:
            xavier_uniform_(layer[0].weight)


class v10Detect(Detect):
    """
    v10 Detection head from https://arxiv.org/pdf/2405.14458.

    Args:
        nc (int): Number of classes.
        ch (tuple): Tuple of channel sizes.

    Attributes:
        max_det (int): Maximum number of detections.

    Methods:
        __init__(self, nc=80, ch=()): Initializes the v10Detect object.
        forward(self, x): Performs forward pass of the v10Detect module.
        bias_init(self): Initializes biases of the Detect module.

    """

    end2end = True

    def __init__(self, nc=80, ch=()):
        """Initializes the v10Detect object with the specified number of classes and input channels."""
        super().__init__(nc, ch)
        c3 = max(ch[0], min(self.nc, 100))  # channels
        # Light cls head
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(Conv(x, x, 3, g=x), Conv(x, c3, 1)),
                nn.Sequential(Conv(c3, c3, 3, g=c3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, self.nc, 1),
            )
            for x in ch
        )
        self.one2one_cv3 = copy.deepcopy(self.cv3)
