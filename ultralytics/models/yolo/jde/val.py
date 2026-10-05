# Ultralytics YOLO 🚀, AGPL-3.0 license

from pathlib import Path, PosixPath
import os

import numpy as np
import torch

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.metrics import DetMetrics, box_iou, EmbeddingMetrics
from ultralytics.utils.torch_utils import smart_inference_mode
from ultralytics.utils.loss import StateMetrics #￥#添加人员状态预测评估指标


class JDEDetMetrics(DetMetrics):
    """DetMetrics 子类：让 .fitness 返回「检测 mAP50-95 + 姿势 macro accuracy」复合值，
    使独立 model.val() 返回的 metrics.fitness 与训练 best.pt 选择口径一致。
    训练 best.pt 选择仍走 get_stats() 返回 dict 的 "fitness" 键；两者算法保持相同。
    无姿势标签时(_pose_acc is None)退化为纯 mAP50-95，兼容纯人员检测数据集。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pose_acc = None              # 姿势 macro accuracy；None=无姿势标签，由 get_stats() 刷新
        self._fitness_w_det = 0.7
        self._fitness_w_pose = 0.3

    @property
    def fitness(self):
        # 无姿势标签(_pose_acc is None)时退化为纯 mAP50-95，与 get_stats() 的 dict 路径一致
        if self._pose_acc is None:
            return float(self.box.map)
        return self._fitness_w_det * float(self.box.map) + self._fitness_w_pose * float(self._pose_acc)


class JDEValidator(DetectionValidator):
    """
    A class extending the DetectionValidator class for validation based on a joint detection and embedding model.

    Example:
        ```python
        from ultralytics.models.yolo.jde import JDEValidator

        args = dict(model="yolov8n-jde.pt", data="coco8-seg.yaml")
        validator = JDEValidator(args=args)
        validator()
        ```
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """Initialize SegmentationValidator and set task to 'segment', metrics to SegmentMetrics."""
        # ========== 在调用super()之前提取自定义参数 ==========
        # 从args中提取自定义参数（如果args是字典）
        if args is not None and isinstance(args, dict):
            # 提取自定义参数
            self._model_name = args.pop('model_name', None)
            self._save_excel = args.pop('save_excel', False)
            self._excel_save_dir = args.pop('excel_save_dir', None)
            self._excel_name = args.pop('excel_name', 'result_all.xlsx')
            self._save_state_to_txt = args.pop('save_state_to_txt', False)
        else:
            # 如果args不是字典或为None，设置默认值
            self._model_name = None
            self._save_excel = False
            self._excel_save_dir = None
            self._excel_name = 'result_all.xlsx'
            self._save_state_to_txt = False
        self.state_class_images = {}  # 记录每个状态类别出现在哪些图像中
        # 现在调用super().__init__()，此时args中已经不包含自定义参数了
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.plot_masks = None
        self.process = None
        self.args.task = "jde"  # 确保task设置为jde，以支持6列标签格式
        self.metrics = JDEDetMetrics(save_dir=self.save_dir, on_plot=self.on_plot)
        self.embedding_metrics = EmbeddingMetrics()
        #model.person_states = data.get("person_states", {})
        
        # 添加状态预测指标 #￥#添加人员状态预测评估指标
        #self.state_metrics = StateMetrics(num_states=166) #&#初始化不能访问model #根据实际状态数量设置 #￥#添加人员状态预测评估指标
        self.state_metrics = None #&#初始化不能访问model替换
        self._num_states_hint = getattr(self.args, "state_classes", None)  # #&#初始化不能访问model替换 可选：从 args 提示
        # 存储状态预测数据用于批处理评估
        self.all_pred_states = [] #￥#添加人员状态预测评估指标
        self.all_target_states = [] #￥#添加人员状态预测评估指标
        self.state_iou = 0.5  # 默认0.5，0.75等
        
        # 添加状态检测指标（复用检测指标计算，使用相同的IoU阈值和置信度阈值）
        self.state_det_metrics = DetMetrics(save_dir=self.save_dir, on_plot=self.on_plot)
        # 存储状态检测的统计数据（类似self.stats）
        self.state_det_stats = {
            "conf": [],
            "pred_cls": [],
            "tp": [],
            "target_cls": [],
            "target_img": [],
        }
        # ========== 新增：按目标尺寸统计检测/状态检测 ==========
        self._init_size_buckets()
    # ========== 新增：按目标尺寸统计检测/状态检测 ==========
    def _init_size_buckets(self):
        
        # 初始化尺寸分桶定义，并重置统计计数。
        # 尺寸单位：原图像素，按 GT 框的宽和高同时落入区间来划分。
        
        # (lo, hi]: 左开右闭区间；第一个区间 lo=0 时等价于 <=hi
        self.size_bins = [
            (0, 16),
            (16, 24),
            (24, 32),
            (32, 40),
            (40, 48),
            (48, 56),
            (56, 64),
            (64, 72),
            (72, 80),
            (80, 88),
            (88, 96),
            (96, 112),
            (112, 128),
            (128, 192),
            (192, float("inf")),
        ]
        self.size_bucket_names = [
            "<=16",
            "16-24",
            "24-32",
            "32-40",
            "32-48",
            "48-56",
            "56-64",
            "64-72",
            "72-80",
            "80-88",
            "88-96",
            "96-112",
            "112-128",
            "128-192",
            ">192",
        ]
        
        # 初始化尺寸分桶定义：改为按目标像素面积分桶。
        # 面积单位：原图上的像素数（w*h）。
        # 分桶：
        # 0: area < 20
        # 1: 20 <= area <= 50
        # #2: area > 50
        """
        # 这里的 (lo, hi) 只是保存阈值，真正的比较逻辑在 update_metrics 里单独写
        self.size_bins = [
            (0.0, 200.0),             # area < 20
            (200.0, 300.0),
            (300.0, 400.0),
            (400.0, 500.0),
            (500.0, 1000.0),
            (1000.0, 1500.0),
            (1500.0, 2000.0),
            (2000.0, 2500.0),
            (2500.0, 3000.0),
            (3000.0, 3500.0),
            (3500.0, 4000.0),
            (4000.0, 4500.0),
            (4500.0, 5000.0),
            (5000.0, float("inf")),    # area > 50
        ]
        self.size_bucket_names = [
            "<=200",
            "200-300",
            "300-400",
            "400-500",
            "500-1000",
            "1000-1500",
            "1500-2000",
            "2000-2500",
            "2500-3000",
            "3000-3500",
            "3500-4000",
            "4000-4500",
            "4500-5000",
            ">5000",
        ]"""
        self._reset_size_bucket_counters()
    # ========== 新增：按目标尺寸统计检测/状态检测 ==========
    def _reset_size_bucket_counters(self):
        """每次验证前重置按尺寸的统计结果。"""
        # 检测（box）按尺寸统计
        self.size_det_counts = [
            {"gt": 0, "tp": 0, "pred": 0} for _ in self.size_bins
        ]
        # 状态检测按尺寸统计
        self.size_state_det_counts = [
            {"gt": 0, "tp": 0, "pred": 0} for _ in self.size_bins
        ]

    def _reset_state_evaluation_stats(self):
        """每次验证前重置状态检测(state_det_stats)和状态识别(state_metrics)的统计累加器，
        避免它们跨 epoch 累积导致 instances/accuracy/mAP 失真。

        注：state_det_metrics / self.metrics / embedding_metrics 无需在此重置——
        DetMetrics.process() 内部是覆盖式更新(self.box.update(results))，每次基于当前 stats 重算；
        EmbeddingMetrics.get_metrics() 末尾已清空 embeds/states。
        """
        # 1) 状态检测统计（dict of lists）——重建为空列表，与 __init__ 中结构一致
        self.state_det_stats = {
            "conf": [],
            "pred_cls": [],
            "tp": [],
            "target_cls": [],
            "target_img": [],
        }
        # 2) 状态识别累加器（StateMetrics）——_ensure_state_metrics_initialized 之后必不为 None，判空仅为保险
        if self.state_metrics is not None:
            self.state_metrics.reset()
        # 3) “每个状态类别出现在哪些图像”的追踪字典
        self.state_class_images = {}

    @smart_inference_mode()
    def __call__(self, trainer=None, model=None):
        """Performs validation on the model and sets the epoch and best attributes."""
        if trainer is None and model is not None : #￥#评估阶段用
            self.model_path = model if (isinstance(model, str) or isinstance(model, PosixPath)) else model.pt_path
            if hasattr(model, 'model') and hasattr(model.model[-1], 'state_classes'):
                self.model = model #$#添加人员状态预测评估指标
        elif trainer is not None: #$#训练验证阶段用
            self.epoch = trainer.epoch + 1
            self.best = trainer.best
            self.trainer = trainer #￥#
            if model is None and hasattr(trainer, 'model'): #￥#
                self.model = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
                #self.model = trainer.model.module #￥#
            elif model is not None:
                self.model_path = model if (isinstance(model, str) or isinstance(model, PosixPath)) else model.pt_path
                self.model = model #$#添加人员状态预测评估指标        
        
        # 确保在这里完成 state_metrics 的初始化 #$$$$$$$$$$$$$#  #&#初始化不能访问model替换
        self._ensure_state_metrics_initialized()#$$$$$$$$$$$$$# #&#初始化不能访问model替换

        # ========== 新增：每次验证前重置尺寸统计 ==========
        self._reset_size_bucket_counters()

        # ========== 修复：每次验证前重置状态检测/状态识别统计累加器，避免跨 epoch 累积 ==========
        self._reset_state_evaluation_stats()

        stats = super().__call__(trainer, model)
        return stats
    def _ensure_state_metrics_initialized(self): #$$$$$$$$$$$$$# #&#初始化不能访问model替换
        if self.state_metrics is not None: #&#初始化不能访问model替换
            return  # 已经初始化过 #&#初始化不能访问model替换

        # 1) 优先从 args 里拿（可选，若你在外部配置过）
        num_states = self._num_states_hint #&#初始化不能访问model替换
        
        # 3) 从模型检测头读取state_classes
        if num_states is None and self.model is not None:
            head = getattr(self.model, "model", None)  # 可能是 nn.Sequential/ModuleList/自定义容器
            last = None
            if head is not None:
                # 常见两种结构：list-like 或再嵌套一层 .model
                if hasattr(head, "__getitem__"):
                    last = head[-1]
                else:
                    inner = getattr(head, "model", None)
                    if inner is not None and hasattr(inner, "__getitem__"):
                        last = inner[-1]
            if last is not None:
                num_states = getattr(last, "state_classes", None)

        # 4) 如果还没有拿到，使用默认值而不是报错
        if num_states is None:
            print("警告：无法获取state_classes，使用默认值166")
            num_states = 166  # 使用默认值，避免报错

        self.state_metrics = StateMetrics(num_states=int(num_states))
    def _prepare_batch(self, si, batch):
        """Prepares a batch of images and annotations for validation."""
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1)
        states = batch["states"][idx].squeeze(-1)
        bbox = batch["bboxes"][idx]
        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        if len(cls):
            bbox = ops.xywh2xyxy(bbox) * torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]]  # target boxes
            ops.scale_boxes(imgsz, bbox, ori_shape)  # native-space labels
        return {"cls": cls, "bbox": bbox, "ori_shape": ori_shape, "imgsz": imgsz, "states": states}

    def _prepare_pred(self, pred, pbatch):
        """修改后的_prepare_pred函数，与predict.py保持一致"""
        predn = pred.clone()
        # 使用与predict.py完全相同的参数调用scale_boxes
        ops.scale_boxes(pbatch["imgsz"], predn[:, :4], pbatch["ori_shape"], padding=True)
        return predn
    def build_dataset(self, img_path, mode="val", batch=None):
        """Build YOLO Dataset with predict-consistent transforms."""
        from ultralytics.data import YOLODataset
        from ultralytics.data.augment import LetterBox, Format, Compose
        
        # 创建数据集
        dataset = YOLODataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=False,
            hyp=self.args,
            rect=True,  # ← 强制禁用rect模式，与predict一致   旧版：rect=False
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride=int(self.stride),
            pad=0.5,  # ← 改为0.0，与predict一致（predict不使用padding）  旧版：pad=0.5
            prefix=f"{mode}: ",
            task=self.args.task,
            classes=self.args.classes,
            data=self.data,
            fraction=1.0,
        )
        
        # 覆盖transforms，使用与predict一致的LetterBox参数
        transforms = Compose([LetterBox(
            new_shape=(self.args.imgsz, self.args.imgsz),
            auto=True,      # 与predict一致
            scaleup=True,   # 与predict一致
            stride=int(self.stride),
        )])
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=False,
                return_keypoint=False,
                return_obb=False,
                batch_idx=True,
                mask_ratio=self.args.mask_ratio,
                mask_overlap=self.args.overlap_mask,
                bgr=0.0,
            )
        )
        dataset.transforms = transforms
        
        return dataset
    def update_metrics(self, preds, batch):
        """Metrics."""
        batch_matched_states = []
        
        # 清空当前批次的状态数据
        self.current_batch_pred_states = [] #￥#人员状态预测评估指标
        self.current_batch_target_states = [] #$#人员状态预测评估指标
        
        for si, pred in enumerate(preds):
            self.seen += 1
            npr = len(pred)
            stat = dict(
                conf=torch.zeros(0, device=self.device),
                pred_cls=torch.zeros(0, device=self.device),
                tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
            )
            matched_states = torch.zeros(npr, dtype=torch.int, device=self.device)    # Initialize matched states tensor
            pbatch = self._prepare_batch(si, batch)
            cls, bbox, states = pbatch.pop("cls"), pbatch.pop("bbox"), pbatch.pop("states")
            nl = len(cls)
            stat["target_cls"] = cls
            stat["target_img"] = cls.unique()

            # ========== 新增：为每个 GT 目标打上尺寸桶标签，并统计 GT 数量 ==========
            gt_size_bucket = None
            if nl:
                # bbox 为 xyxy（像素坐标）
                wh = bbox[:, 2:4] - bbox[:, 0:2]  # (nl, 2)
                w, h = wh[:, 0], wh[:, 1]
                gt_size_bucket = torch.full((nl,), -1, dtype=torch.long, device=self.device)
                # 使用 max(w, h) 来决定每个目标属于哪个尺寸桶（避免一个目标被分到多个桶）
                max_dim = torch.max(w, h)  # (nl,)
                for bi, (lo, hi) in enumerate(self.size_bins):
                    # 第一个桶 (0, 12] 等价于 w<=12, h<=12
                    if lo == 0:
                        mask = max_dim <= hi
                    else:
                        mask = (max_dim > lo) & (max_dim <= hi)
                    # 只给还没有分配桶的目标分配（避免覆盖）
                    unassigned = gt_size_bucket == -1
                    gt_size_bucket[mask & unassigned] = bi

                # 按尺寸桶累计 GT 数量（检测 + 状态检测共用同一 GT）
                for bi in range(len(self.size_bins)):
                    cnt = int((gt_size_bucket == bi).sum().item())
                    if cnt > 0:
                        self.size_det_counts[bi]["gt"] += cnt
                        self.size_state_det_counts[bi]["gt"] += cnt  # ========== 新增：按目标尺寸统计检测/状态检测 ==========
            """
            # ========== 为每个 GT 目标按“像素面积”打尺寸桶标签，并统计 GT 数量 ==========
            gt_size_bucket = None
            if nl:
                # bbox 为 xyxy（像素坐标）
                wh = bbox[:, 2:4] - bbox[:, 0:2]  # (nl, 2)
                w, h = wh[:, 0], wh[:, 1]
                area = w * h  # 每个GT框的像素面积

                gt_size_bucket = torch.full((nl,), -1, dtype=torch.long, device=self.device)

                for bi, (lo, hi) in enumerate(self.size_bins):
                    # 三个桶：
                    #  0: area < 20      -> (0.0, 20.0)
                    #  1: 20 <= area <= 50 -> (20.0, 50.0)
                    #  2: area > 50      -> (50.0, inf)
                    if bi == 0:
                        # 面积 < hi  ( <20 )
                        mask = area <= hi
                    elif hi == float("inf"):
                        # 面积 > lo  ( >50 )
                        mask = area > lo
                    else:
                        # lo <= area <= hi  (20~50)
                        mask = (area > lo) & (area <= hi)

                    # 只给还没分桶的 GT 赋值，避免重复
                    unassigned = gt_size_bucket == -1
                    gt_size_bucket[mask & unassigned] = bi

                # 按尺寸桶累计 GT 数量（检测 + 状态检测共用同一 GT）
                for bi in range(len(self.size_bins)):
                    cnt = int((gt_size_bucket == bi).sum().item())
                    if cnt > 0:
                        self.size_det_counts[bi]["gt"] += cnt
                        self.size_state_det_counts[bi]["gt"] += cnt # ========== 为每个 GT 目标按“像素面积”打尺寸桶标签，并统计 GT 数量 ==========
            """

            if npr == 0:
                if nl:
                    for k in self.stats.keys():
                        self.stats[k].append(stat[k])
                    if self.args.plots:
                        self.confusion_matrix.process_batch(detections=None, gt_bboxes=bbox, gt_cls=cls)
                batch_matched_states.append(matched_states)#%￥￥%￥# 添加空的matched_states，确保列表长度匹配
                continue

            # Predictions
            if self.args.single_cls:
                pred[:, 5] = 0
            predn = self._prepare_pred(pred, pbatch)
            stat["conf"] = predn[:, 4]
            stat["pred_cls"] = predn[:, 5]
            
 #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            # Evaluate #$#$#$#$ 
            if nl:
                stat["tp"], matched_states = self._process_batch(predn, bbox, cls, states)  
                if self.args.plots:
                    self.confusion_matrix.process_batch(predn, bbox, cls)
            for k in self.stats.keys():
                self.stats[k].append(stat[k])
            batch_matched_states.append(matched_states)

            # ========== 新增：按尺寸统计“检测” TP（IoU=self.state_iou，类别正确） ==========
            # 在 match_predictions 中，我们会在 IoU==self.state_iou 时缓存当前 (gt_idx, pred_idx) 匹配到 self.last_matches
            if nl and gt_size_bucket is not None and hasattr(self, "last_matches") and self.last_matches is not None:
                # self.last_matches: numpy 数组，形状 (K, 2)，每行 [gt_idx, pred_idx]
                matched_gt_idx = set(int(g) for g in np.unique(self.last_matches[:, 0]))
                for gt_idx in matched_gt_idx:
                    if 0 <= gt_idx < nl:
                        bi = int(gt_size_bucket[gt_idx].item())
                        if bi >= 0:
                            self.size_det_counts[bi]["tp"] += 1  # ========== 新增：按尺寸统计“检测” TP（IoU=self.state_iou，类别正确） ==========

            # ========== 新增：按尺寸统计“检测”预测数量，用于计算precision ==========
            if npr > 0 and nl > 0 and gt_size_bucket is not None:
                # 计算 GT 与预测的 IoU 矩阵
                iou_matrix = box_iou(bbox, predn[:, :4])  # (nl, npr)

                for pred_i in range(npr):
                    # 找到与该预测 IoU 最大的 GT
                    ious = iou_matrix[:, pred_i]
                    max_iou, gt_idx_max = torch.max(ious, dim=0)
                    if max_iou <= 0:
                        # 与任何 GT 都没有重叠，不归入任何尺寸桶（这样 P 会略微偏高一点）
                        continue
                    gt_idx_use = int(gt_idx_max)
                    bi = int(gt_size_bucket[gt_idx_use].item())
                    if bi < 0:
                        continue
                    self.size_det_counts[bi]["pred"] += 1  # ========== 新增：按尺寸统计“检测”预测数量，用于计算precision ==========


            # ========== 状态检测指标计算（复用检测指标代码，完全替换类别和置信度）==========
            # ========== 状态检测指标计算（复用检测指标代码，完全替换类别和置信度）==========
            if hasattr(self, "state_det_stats") and hasattr(self.model.model[-1], "state_classes") and self.model.model[-1].state_classes is not None:
                embed_dim = self.model.model[-1].embed_dim
                state_classes = self.model.model[-1].state_classes
                
                # 提取状态预测概率
                if pred.shape[1] > 6 + embed_dim and len(predn) > 0:
                    # 直接从pred中提取状态预测（pred和predn是一一对应的，不需要IoU匹配）
                    # 方案1 CosFace: 模型输出 state 已是余弦相似度[-1,1](越大越像该姿势), argmax 取姿势类; cls 仍是 sigmoid(单类人)
                    state_probs = pred[:, 6 + embed_dim:6 + embed_dim + state_classes]  # (N, state_classes)
                    
                    # 直接使用argmax获取预测的状态类别
                    state_conf, state_cls = state_probs.max(1)  # (N,), (N,)
                    
                    # 使用检测框的置信度作为状态检测的置信度
                    det_conf = predn[:, 4]  # 检测框的原始置信度
                    
                    # 构造"状态检测"的预测矩阵：使用predn的bbox，替换conf和cls为状态相关的
                    pred_state_det = predn.clone()
                    pred_state_det[:, 4] = det_conf   # * (state_conf ** 12) 之前的结果都 使用检测框置信度det_conf
                    pred_state_det[:, 5] = state_cls.float()  # 替换类别为状态类别
                    
                    # 用"状态id"作为GT类别（states是0-based，0-5对应6个状态）
                    if states is not None and states.numel() > 0:
                        gt_state_cls = states.to(dtype=torch.long, device=pred.device).view(-1)
                        gt_state_cls = gt_state_cls.clamp_(min=0, max=state_classes-1)
                    else:
                        gt_state_cls = torch.zeros(0, dtype=torch.long, device=pred.device)
                    
                    # 追踪每个状态类别出现在哪些图像中
                    for state in gt_state_cls:
                        state_val = int(state.item())
                        if state_val not in self.state_class_images:
                            self.state_class_images[state_val] = set()
                        self.state_class_images[state_val].add(self.seen)
                    
                    # 构造占位的gt_states
                    gt_dummy_states = torch.zeros_like(gt_state_cls, dtype=torch.long, device=pred.device)
                    
                    # 计算状态检测的TP矩阵
                    if pred_state_det.numel() > 0 and len(gt_state_cls) > 0:
                        original_nc = self.nc
                        self.nc = state_classes
                        
                        state_stat = dict(
                            conf=torch.zeros(0, device=self.device),
                            pred_cls=torch.zeros(0, device=self.device),
                            tp=torch.zeros(len(pred_state_det), self.niou, dtype=torch.bool, device=self.device),
                        )
                        state_stat["target_cls"] = gt_state_cls
                        state_stat["target_img"] = gt_state_cls.unique()
                        
                        state_stat["tp"], _ = self._process_batch(pred_state_det, bbox, gt_state_cls, gt_dummy_states)
                        state_stat["conf"] = pred_state_det[:, 4]
                        state_stat["pred_cls"] = pred_state_det[:, 5].long()
                        
                        self.nc = original_nc
                        
                        for k in self.state_det_stats.keys():
                            self.state_det_stats[k].append(state_stat[k])

                        # ========== 新增：按尺寸统计“状态检测” TP（IoU=self.state_iou，状态类别正确） ==========
                        # 此时 match_predictions 再次被调用（用于状态检测），self.last_matches 对应的是状态检测的匹配
                        if nl and gt_size_bucket is not None and hasattr(self, "last_matches") and self.last_matches is not None:
                            matched_gt_idx = set(int(g) for g in np.unique(self.last_matches[:, 0]))
                            for gt_idx in matched_gt_idx:
                                if 0 <= gt_idx < nl:
                                    bi = int(gt_size_bucket[gt_idx].item())
                                    if bi >= 0:
                                        self.size_state_det_counts[bi]["tp"] += 1 # ========== 新增：按尺寸统计“状态检测” TP（IoU=self.state_iou，状态类别正确） ==========
                        # ========== 新增：按尺寸统计“状态检测”预测数量，用于计算precision ==========
                        if pred_state_det.numel() > 0 and nl > 0 and gt_size_bucket is not None:
                            # 计算 GT 与状态检测预测框的 IoU 矩阵
                            iou_matrix_state = box_iou(bbox, pred_state_det[:, :4])  # (nl, N_pred)

                            for pred_i in range(len(pred_state_det)):
                                ious = iou_matrix_state[:, pred_i]
                                max_iou, gt_idx_max = torch.max(ious, dim=0)
                                if max_iou <= 0:
                                    # 与任何 GT 都没有重叠，不归入任何尺寸桶
                                    continue
                                gt_idx_use = int(gt_idx_max)
                                bi = int(gt_size_bucket[gt_idx_use].item())
                                if bi < 0:
                                    continue
                                self.size_state_det_counts[bi]["pred"] += 1 # ========== 新增：按尺寸统计“状态检测”预测数量，用于计算precision ==========


                    elif len(gt_state_cls) > 0:
                        # 只有GT没有预测的情况
                        state_stat = dict(
                            conf=torch.zeros(0, device=self.device),
                            pred_cls=torch.zeros(0, device=self.device),
                            tp=torch.zeros(0, self.niou, dtype=torch.bool, device=self.device),
                        )
                        state_stat["target_cls"] = gt_state_cls
                        state_stat["target_img"] = gt_state_cls.unique()
                        for k in self.state_det_stats.keys():
                            self.state_det_stats[k].append(state_stat[k])

            # 收集当前图像的状态预测和目标（基于GT统计）
            if npr > 0 and nl > 0: #￥#人员状态预测评估指标
                # 传入TP信息，确保只统计TP
                tp_mask = stat["tp"][:, 0] if stat["tp"].numel() > 0 else torch.zeros(npr, dtype=torch.bool, device=self.device)  # 使用IoU=0.5的TP
                self._collect_state_data_for_image(si, pred, predn, bbox, states, matched_states, tp_mask) #￥#人员状态预测评估指标

            # Save
            if self.args.save_json:
                self.pred_to_json(predn, batch["im_file"][si])
            if self.args.save_txt:
                # 获取图像文件的完整路径
                im_file_path = Path(batch["im_file"][si])
                
                # 尝试从self.data获取数据集根目录
                dataset_root = None
                if hasattr(self, 'data') and self.data is not None:
                    dataset_root = self.data.get("path")
                    if dataset_root:
                        dataset_root = Path(dataset_root)
                
                # 计算相对路径并构建保存路径
                if dataset_root and dataset_root.exists():
                    try:
                        # 获取相对于数据集根目录的相对路径
                        relative_path = im_file_path.relative_to(dataset_root)
                        # 移除文件名，保留目录结构
                        relative_dir = relative_path.parent
                        
                        # 如果相对路径的第一部分是images，则跳过它
                        relative_parts = list(relative_dir.parts)
                        if relative_parts and relative_parts[0] == 'images':
                            relative_parts = relative_parts[1:]  # 跳过images目录
                            if relative_parts:
                                relative_dir = Path(*relative_parts)
                            else:
                                relative_dir = Path()  # 空路径
                        
                        # 构建保存路径：save_dir/labels/相对目录/文件名.txt
                        if relative_dir.parts:
                            save_path = self.save_dir / "labels" / relative_dir / f'{im_file_path.stem}.txt'
                        else:
                            save_path = self.save_dir / "labels" / f'{im_file_path.stem}.txt'
                    except (ValueError, AttributeError):
                        # 如果无法计算相对路径，使用路径哈希区分
                        path_str = str(im_file_path.parent).replace(os.sep, '_').replace('/', '_')
                        # 只保留最后几级路径避免文件名过长
                        path_parts = path_str.split('_')
                        path_suffix = '_'.join(path_parts[-3:]) if len(path_parts) > 3 else path_str[-50:]
                        save_path = self.save_dir / "labels" / f'{im_file_path.stem}_{path_suffix}.txt'
                else:
                    # 如果找不到数据集根目录，从图像路径中提取子目录结构
                    # 查找images、train、val、test等关键目录
                    path_parts = list(im_file_path.parts)
                    subdir_parts = []
                    start_idx = None
                    
                    # 找到images、train、val、test等目录的索引
                    for i, part in enumerate(path_parts):
                        if part in ['images', 'train', 'val', 'test']:
                            start_idx = i + 1  # 从该目录之后开始（跳过images等目录）
                            break
                    
                    if start_idx is not None and start_idx < len(path_parts) - 1:
                        # 提取子目录部分（不包括文件名）
                        subdir_parts = path_parts[start_idx:-1]
                        if subdir_parts:
                            # 构建保存路径：save_dir/labels/子目录/文件名.txt
                            save_path = self.save_dir / "labels" / Path(*subdir_parts) / f'{im_file_path.stem}.txt'
                        else:
                            save_path = self.save_dir / "labels" / f'{im_file_path.stem}.txt'
                    else:
                        # 如果找不到关键目录，使用父目录名+文件名（但跳过images）
                        parent_name = im_file_path.parent.name
                        if parent_name and parent_name not in ['images', 'train', 'val', 'test']:
                            save_path = self.save_dir / "labels" / parent_name / f'{im_file_path.stem}.txt'
                        else:
                            save_path = self.save_dir / "labels" / f'{im_file_path.stem}.txt'
                
                self.save_one_txt(
                    predn,
                    self.args.save_conf,
                    pbatch["ori_shape"],
                    save_path,
                    pred=pred,  # 传入原始pred以提取状态信息   
                )
        
        # 在调用embedding_metrics之前检查batch_matched_states是否为空 #%￥￥%￥#
        if batch_matched_states and any(len(states) > 0 for states in batch_matched_states): #%￥￥%￥#
            # Process batch for reid metrics
            self.embedding_metrics.process_batch(preds, batch_matched_states) #往后处理%￥￥%￥#
        
        self._process_batch_state_metrics() #￥#人员状态预测评估指标# 处理当前批次的状态预测评估

    def _collect_state_data_for_image(self, image_idx, pred, predn, gt_bboxes, gt_states, matched_states, tp_mask):
        """
        收集单张图像的状态预测数据（只统计匹配成功的预测-GT对，与zhibiao.py一致）
        """
        if not hasattr(self.model.model[-1], 'state_classes') or self.model.model[-1].state_classes is None:
            return
            
        embed_dim = self.model.model[-1].embed_dim
        state_classes = self.model.model[-1].state_classes
        
        # 从预测结果中提取状态预测
        if pred.shape[1] > 6 + embed_dim:
            state_preds = pred[:, 6+embed_dim:6+embed_dim+state_classes]  # (N_detections, state_classes)
            
            # 安全地检查张量是否为空（处理0维张量的情况）
            if gt_states.numel() == 0 or len(predn) == 0:
                return
            
            # 确保gt_states和gt_bboxes是正确的维度
            if gt_states.dim() == 0:
                gt_states = gt_states.unsqueeze(0)
            if gt_bboxes.dim() == 1:
                gt_bboxes = gt_bboxes.unsqueeze(0)
            
            # 计算IoU矩阵，用于找到每个GT匹配的最佳预测框
            from ultralytics.utils.metrics import box_iou
            iou_matrix = box_iou(gt_bboxes, predn[:, :4])  # (M, N)
            
            # 设置IoU阈值（与zhibiao.py一致）
            iou_threshold = 0.5
            
            # 直接统计匹配成功的GT实例（与zhibiao.py一致）
            pred_states_list = []
            target_states_list = []
            
            for gt_idx, gt_state in enumerate(gt_states):
                gt_state_value = gt_state.item() if gt_state.dim() == 0 else int(gt_state)
                
                # 状态索引（0-based）
                gt_state_0based = int(gt_state_value)
                
                # 检查范围：应该在0到5之间（0-based，对应6个状态）
                if gt_state_0based < 0 or gt_state_0based >= 6:
                    continue
                
                # 找到该GT的最佳匹配预测框（基于IoU）
                if gt_idx < len(iou_matrix):
                    iou_scores = iou_matrix[gt_idx]
                    max_iou = torch.max(iou_scores)
                    
                    # 只有IoU >= 阈值才算匹配成功（与zhibiao.py一致）
                    if max_iou >= iou_threshold:
                        best_pred_idx = torch.argmax(iou_scores).item()
                        
                        # 使用匹配预测的状态
                        pred_states_list.append(state_preds[best_pred_idx:best_pred_idx+1])
                        
                        # 使用0-based索引用于StateMetrics
                        target_states_list.append(torch.tensor([gt_state_0based], device=pred.device, dtype=torch.long))
                    # else: FN情况，不计入样本（与zhibiao.py一致）

            # 合并所有匹配成功的GT实例的状态预测
            if pred_states_list and target_states_list:
                batch_pred_states = torch.cat(pred_states_list, dim=0)
                batch_target_states = torch.cat(target_states_list, dim=0)
                
                self.current_batch_pred_states.append(batch_pred_states)
                self.current_batch_target_states.append(batch_target_states)

    def _process_batch_state_metrics(self): #￥#人员状态预测评估指标
        """处理当前批次的状态预测指标"""
        if not self.current_batch_pred_states or not self.current_batch_target_states: #￥#人员状态预测评估指标
            return #￥#人员状态预测评估指标
            
        if self.state_metrics is None: #￥#人员状态预测评估指标
            return #￥#人员状态预测评估指标
            
        # 合并当前批次的所有状态预测数据
        try: #￥#人员状态预测评估指标
            batch_pred_states = torch.cat(self.current_batch_pred_states, dim=0) #￥#人员状态预测评估指标
            batch_target_states = torch.cat(self.current_batch_target_states, dim=0) #￥#人员状态预测评估指标
            
            # 构建图像索引（每个样本对应的图像索引）
            image_indices = []
            for img_idx, states in enumerate(self.current_batch_target_states):
                image_indices.extend([img_idx + self.seen - len(self.current_batch_target_states)] * len(states))
            image_indices = np.array(image_indices)
            
            # 确保维度匹配
            if len(batch_pred_states) == len(batch_target_states): #￥#人员状态预测评估指标
                self.state_metrics.process(batch_pred_states, batch_target_states, image_indices) #￥#人员状态预测评估指标
                #print(f"✅ 成功处理 {len(batch_pred_states)} 个状态预测样本")
                #print(f"📈 当前状态准确率: {self.state_metrics.state_accuracy:.4f}")
            else:
                print(f"警告：状态预测维度不匹配 - pred: {len(batch_pred_states)}, target: {len(batch_target_states)}") #￥#人员状态预测评估指标
                 
        except Exception as e: #￥#人员状态预测评估指标
            print(f"状态指标处理错误: {e}") #￥#人员状态预测评估指标

    def get_stats(self):
        """Returns metrics statistics and results dictionary."""
        stats = {k: torch.cat(v, 0).cpu().numpy() for k, v in self.stats.items()}  # to numpy
        self.nt_per_class = np.bincount(stats["target_cls"].astype(int), minlength=self.nc)
        self.nt_per_image = np.bincount(stats["target_img"].astype(int), minlength=self.nc)
        stats.pop("target_img", None)
        

        # 定义完整的ReID metrics结构（无论是否有检测都返回）
        embedding_metrics = {
            "val/pos_cos": 0.0,
            "val/neg_cos": 1.0,
            "val/pos_euc": 0.0,
            "val/neg_euc": 1.0,
            "val/cos_sep_ratio": 1.0,
            "val/euc_sep_ratio": 1.0,
            "val/cos_silhouette": 0.0,
            "val/euc_silhouette": 0.0,
            "val/davies_bouldin": 0.0,
            "val/calinski_harabasz": 0.0,
            "val/r1_acc": 0.0,
            "val/r5_acc": 0.0,
            "val/mean_ap": 0.0,
            "val/hota": 0.0,
            "val/mota": 0.0,
            "val/idf1": 0.0,
        }
        
        if len(stats) and stats["tp"].any():
            self.metrics.process(**stats)
            # 只有在有正确检测时才更新embedding_metrics
            computed_embedding_metrics = self.embedding_metrics.get_metrics()
            embedding_metrics.update(computed_embedding_metrics)  # 用实际计算值覆盖默认值
        else:
            # 当没有正确检测时，使用默认的0值占位
            print("⚠️ 当前批次没有正确检测，跳过ReID metrics计算")
            
        detector_results = self.metrics.results_dict
        detector_results.update(embedding_metrics)
        
        # ========== 修复：确保状态预测指标始终返回11个键 ==========
        # 定义完整的状态预测指标结构（与StateMetrics.results_dict返回的键完全一致）
        default_state_metrics = {
            "metrics/state_accuracy": 0.0,
            "metrics/state_macro_accuracy": 0.0,
            "metrics/state_macro_precision": 0.0,
            "metrics/state_macro_recall": 0.0,
            "metrics/state_macro_f1": 0.0,
            "metrics/state_micro_precision": 0.0,
            "metrics/state_micro_recall": 0.0,
            "metrics/state_micro_f1": 0.0,
            "metrics/state_total_tp": 0,
            "metrics/state_total_fp": 0,
            "metrics/state_total_fn": 0,
        }
        
        if self.state_metrics is not None:
            state_results = self.state_metrics.results_dict
            # 用实际计算值更新默认值（确保键完全匹配）
            default_state_metrics.update(state_results)
        
        # 无论是否有state_metrics，都使用相同的键结构
        detector_results.update(default_state_metrics)

        # ========== 修复：确保状态检测指标始终返回6个键 ==========
        # 定义完整的状态检测指标结构（与DetMetrics.results_dict返回的键完全一致）
        default_state_det_metrics = {
            "state_det/metrics/precision(B)": 0.0,
            "state_det/metrics/recall(B)": 0.0,
            "state_det/metrics/mAP50(B)": 0.0,
            "state_det/metrics/mAP75(B)": 0.0,
            "state_det/metrics/mAP50-95(B)": 0.0,
            "state_det/fitness": 0.0,
        }
        
        if hasattr(self, "state_det_stats") and len(self.state_det_stats) > 0:
            state_det_stats = {k: torch.cat(v, 0).cpu().numpy() if v else np.array([]) 
                          for k, v in self.state_det_stats.items()}
            
            if len(state_det_stats) > 0 and state_det_stats["tp"].size > 0 and state_det_stats["tp"].any():
                # 临时保存原始nc和nt_per_class
                original_nc = self.nc
                original_nt_per_class = self.nt_per_class
                
                # 设置状态类别数
                state_classes = getattr(self.model.model[-1], "state_classes", 6) if hasattr(self, "model") and self.model is not None else 6
                self.nc = state_classes
                
                # 计算状态检测的每类统计信息（在pop target_img之前）
                self.state_nt_per_class = np.bincount(
                    state_det_stats["target_cls"].astype(int), 
                    minlength=state_classes
                )
                # 计算每类出现的图像数
                if "target_img" in state_det_stats and state_det_stats["target_img"].size > 0:
                    self.state_nt_per_image = np.bincount(
                        state_det_stats["target_img"].astype(int), 
                        minlength=state_classes
                    )
                else:
                    self.state_nt_per_image = np.zeros(state_classes, dtype=np.int64)
                
                state_det_stats.pop("target_img", None)
                
                # 使用DetMetrics计算状态检测指标
                self.state_det_metrics.process(**state_det_stats)
                state_det_results = self.state_det_metrics.results_dict
                
                # 恢复原始nc
                self.nc = original_nc
                self.nt_per_class = original_nt_per_class
                
                # 重命名状态检测指标（添加前缀）
                state_det_results_renamed = {
                    f"state_det/{k}": v for k, v in state_det_results.items()
                }
                # 用实际计算值更新默认值（确保键完全匹配）
                default_state_det_metrics.update(state_det_results_renamed)
        
            # 初始化空的统计信息（如果需要）
            if not hasattr(self, 'state_nt_per_class'):
                state_classes = getattr(self.model.model[-1], "state_classes", 6) if hasattr(self, "model") and self.model is not None else 6
                self.state_nt_per_class = np.zeros(state_classes, dtype=np.int64)
                self.state_nt_per_image = np.zeros(state_classes, dtype=np.int64)
        
        # 无论是否有state_det_stats，都使用相同的键结构
        detector_results.update(default_state_det_metrics)

        # ===== 复合 fitness：检测 mAP50-95 + 姿势 macro accuracy（多任务模型选择）=====
        # 覆盖顶层 "fitness"（原本 = 1.0×mAP50-95，见 metrics.py:761-764），
        # 让 best.pt 保存（trainer.py:548）与早停（torch_utils.py:713）同时考虑检测和姿势。
        # 无姿势标签/无 TP 时退化为纯 mAP50-95，保持向后兼容。
        if self.state_metrics is not None and getattr(self.state_metrics, "total", 0) > 0:
            pose_acc = float(default_state_metrics.get("metrics/state_macro_accuracy", 0.0))
            w_det = float(getattr(self.args, "fitness_det_weight", 0.7))
            w_pose = float(getattr(self.args, "fitness_pose_weight", 0.3))
            detector_results["fitness"] = w_det * float(self.metrics.box.map) + w_pose * pose_acc
            # 同步到 self.metrics，使独立 model.val() 返回的 .fitness 属性也是复合值
            self.metrics._pose_acc = pose_acc
            self.metrics._fitness_w_det = w_det
            self.metrics._fitness_w_pose = w_pose

        return detector_results

    def preprocess(self, batch):
        """Preprocesses batch by converting masks to float and sending to device."""
        batch = super().preprocess(batch)
        batch["states"] = batch["states"].to(self.device).float()
        return batch

    def postprocess(self, preds):
        """Apply Non-maximum suppression to prediction outputs."""
        
        # 应用NMS（与predict.py保持一致的参数）
        preds = ops.non_max_suppression(
            preds[0],
            self.args.conf,
            self.args.iou,
            labels=self.lb,
            multi_label=True,  # ← 改为False，与predict一致    旧版：multi_label=True
            agnostic=self.args.agnostic_nms,  # ← 简化，与predict一致
            max_det=self.args.max_det,
            nc=self.nc,
            classes=self.args.classes,  # ← 添加classes参数，与predict一致
        )
        return preds

    def _process_batch(self, detections, gt_bboxes, gt_cls, gt_states):
        """
        Return correct prediction matrix.

        Args:
            detections (torch.Tensor): Tensor of shape (N, 6) representing detections where each detection is
                (x1, y1, x2, y2, conf, class).
            gt_bboxes (torch.Tensor): Tensor of shape (M, 4) representing ground-truth bounding box coordinates. Each
                bounding box is of the format: (x1, y1, x2, y2).
            gt_cls (torch.Tensor): Tensor of shape (M,) representing target class indices.
            gt_states (torch.Tensor): Tensor of shape (M,) representing target states.

        Returns:
            (torch.Tensor): Correct prediction matrix of shape (N, 10) for 10 IoU levels.

        Note:
            The function does not return any value directly usable for metrics calculation. Instead, it provides an
            intermediate representation used for evaluating predictions against ground truth.
        """
        iou = box_iou(gt_bboxes, detections[:, :4])
        return self.match_predictions(detections[:, 5], gt_cls, gt_states, iou)

    def match_predictions(self, pred_classes, true_classes, true_states, iou, use_scipy=False):
        """
        Matches predictions to ground truth objects (pred_classes, true_classes) using IoU.

        Args:
            pred_classes (torch.Tensor): Predicted class indices of shape(N,).
            true_classes (torch.Tensor): Target class indices of shape(M,).
            true_states (torch.Tensor): Target states of shape(M,).
            iou (torch.Tensor): An NxM tensor containing the pairwise IoU values for predictions and ground of truth
            use_scipy (bool): Whether to use scipy for matching (more precise).

        Returns:
            (torch.Tensor): Correct tensor of shape(N,10) for 10 IoU thresholds.
        """
        # =================== 新增：每次调用时重置尺寸统计用的匹配缓存  ==================
        # 将在 IoU == self.state_iou 时保存当前 (gt_idx, pred_idx) 匹配
        self.last_matches = None # ===================


        # Initialize the list for storing matched states using IoU threshold of 0.5
        matched_states = [False] * pred_classes.shape[0]  # Default to None if no match

        # Dx10 matrix, where D - detections, 10 - IoU thresholds
        correct = np.zeros((pred_classes.shape[0], self.iouv.shape[0])).astype(bool)
        # LxD matrix where L - labels (rows), D - detections (columns)
        correct_class = true_classes[:, None] == pred_classes
        iou = iou * correct_class  # zero out the wrong classes
        iou = iou.cpu().numpy()
        for i, threshold in enumerate(self.iouv.cpu().tolist()):
            if use_scipy:
                # WARNING: known issue that reduces mAP in https://github.com/ultralytics/ultralytics/pull/4708
                import scipy  # scope import to avoid importing for all commands

                cost_matrix = iou * (iou >= threshold)
                if cost_matrix.any():
                    labels_idx, detections_idx = scipy.optimize.linear_sum_assignment(cost_matrix, maximize=True)
                    valid = cost_matrix[labels_idx, detections_idx] > 0
                    if valid.any():
                        correct[detections_idx[valid], i] = True
                        # Assign states to matched predictions
                        if threshold == self.state_iou:  #￥#￥#￥#￥#￥#￥#

                            # ================== 新增： 缓存 (gt_idx, pred_idx) 匹配，用于尺寸统计  ==================
                            matches_np = np.stack([labels_idx[valid], detections_idx[valid]], axis=1)
                            self.last_matches = matches_np # ==================

                            for gt_idx, pred_idx in zip(labels_idx[valid], detections_idx[valid]):
                                matched_states[pred_idx] = true_states[gt_idx].item()
            else:
                matches = np.nonzero(iou >= threshold)  # IoU > threshold and classes match
                matches = np.array(matches).T
                if matches.shape[0]:
                    if matches.shape[0] > 1:
                        matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                        # matches = matches[matches[:, 2].argsort()[::-1]]
                        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                    correct[matches[:, 1].astype(int), i] = True
                    # Assign states to matched predictions
                    if threshold == self.state_iou:  #￥#￥#￥#￥#￥#￥#
                        
                        # ================== 新增： 缓存 (gt_idx, pred_idx) 匹配，用于尺寸统计  ==================
                        self.last_matches = matches.copy() # ==================

                        for gt_idx, pred_idx in matches:
                            if true_states.dim() > 0: #$#$#
                                matched_states[pred_idx] = true_states[gt_idx].item()
        return torch.tensor(correct, dtype=torch.bool, device=pred_classes.device), torch.tensor(matched_states, dtype=torch.int, device=pred_classes.device)

    def match_predictions_for_state_detection(self, pred_classes, true_classes, iou, use_scipy=False):
        """
        专门用于状态检测的匹配函数：先匹配框（基于IoU），再判断状态类别是否正确
        
        与match_predictions的区别：
        - match_predictions: 先检查类别匹配，再匹配框（用于常规检测）
        - match_predictions_for_state_detection: 先匹配框，再检查状态类别（用于状态检测）
        
        Args:
            pred_classes (torch.Tensor): 预测的状态类别 (N,)
            true_classes (torch.Tensor): 真实的状态类别 (M,)
            iou (torch.Tensor): IoU矩阵 (M, N)，M是GT数量，N是预测数量
            use_scipy (bool): 是否使用scipy进行匹配（更精确）
        
        Returns:
            (torch.Tensor): TP矩阵 (N, 10) 对于10个IoU阈值，dtype=bool
        """
        # Dx10 matrix, where D - detections, 10 - IoU thresholds
        correct = np.zeros((pred_classes.shape[0], self.iouv.shape[0])).astype(bool)
        iou_np = iou.cpu().numpy()  # 转换为numpy用于计算
        
        for i, threshold in enumerate(self.iouv.cpu().tolist()):
            if use_scipy:
                # WARNING: known issue that reduces mAP in https://github.com/ultralytics/ultralytics/pull/4708
                import scipy.optimize  # scope import to avoid importing for all commands
                
                # 先基于IoU匹配框（不考虑类别）
                cost_matrix = iou_np * (iou_np >= threshold)
                if cost_matrix.any():
                    labels_idx, detections_idx = scipy.optimize.linear_sum_assignment(
                        cost_matrix, maximize=True
                    )
                    valid = cost_matrix[labels_idx, detections_idx] > 0
                    if valid.any():
                        # 匹配到框后，再检查状态类别是否匹配
                        for gt_idx, pred_idx in zip(labels_idx[valid], detections_idx[valid]):
                            if true_classes[gt_idx].item() == pred_classes[pred_idx].item():
                                correct[pred_idx, i] = True
            else:
                # 先找到所有IoU >= threshold的匹配（不考虑类别）
                matches = np.nonzero(iou_np >= threshold)  # IoU >= threshold
                matches = np.array(matches).T  # (num_matches, 2)，每行是[gt_idx, pred_idx]
                if matches.shape[0]:
                    if matches.shape[0] > 1:
                        # 按IoU降序排序
                        matches = matches[iou_np[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                        # 每个预测框只保留IoU最高的GT（去重预测框）
                        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                        # 每个GT只保留IoU最高的预测框（去重GT）
                        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                    
                    # 检查匹配的框的状态类别是否也匹配
                    for gt_idx, pred_idx in matches:
                        if true_classes[gt_idx].item() == pred_classes[pred_idx].item():
                            correct[pred_idx, i] = True
        
        return torch.tensor(correct, dtype=torch.bool, device=pred_classes.device)

    def print_results(self):
        """Prints training/validation set metrics per class and state metrics."""
        # 调用父类方法显示检测指标
        super().print_results()
        
        # ========== 添加：打印状态检测指标（类似zhibiao.py的格式）==========
        if hasattr(self, "state_det_stats") and hasattr(self, "state_det_metrics"):
            # 检查是否有状态检测统计数据
            state_det_stats = {k: torch.cat(v, 0).cpu().numpy() if v else np.array([]) 
                            for k, v in self.state_det_stats.items()}
            
            if len(state_det_stats) > 0 and state_det_stats["tp"].size > 0 and state_det_stats["tp"].any():
                # 获取状态类别数和名称
                state_classes = getattr(self.model.model[-1], "state_classes", 6) if hasattr(self, "model") and self.model is not None else 6
                
                # 状态名称映射
                state_names = getattr(self.model, "person_states", None)
                if state_names is None:
                    state_names = {
                        0: "stands", 1: "seated", 2: "laying_down",
                        3: "walking", 4: "running", 5: "not_defined"
                    }
                
                # 临时保存原始设置
                original_nc = self.nc
                original_nt_per_class = self.nt_per_class
                original_nt_per_image = self.nt_per_image
                
                # 设置状态类别数
                self.nc = state_classes
                
                # 计算状态检测的每类统计信息
                self.state_nt_per_class = np.bincount(
                    state_det_stats["target_cls"].astype(int), 
                    minlength=state_classes
                )
                
                # 计算每个状态类别出现的图像数
                self.state_nt_per_image = np.zeros(state_classes, dtype=np.int64)
                if hasattr(self, 'state_class_images') and self.state_class_images:
                    for cls_id, img_set in self.state_class_images.items():
                        if 0 <= cls_id < state_classes:
                            self.state_nt_per_image[cls_id] = len(img_set)
                
                # 处理状态检测指标
                state_det_stats_copy = {k: v.copy() for k, v in state_det_stats.items()}
                state_det_stats_copy.pop("target_img", None)
                
                # 使用DetMetrics计算状态检测指标
                self.state_det_metrics.names = state_names  # 设置名称
                self.state_det_metrics.process(**state_det_stats_copy)
                
                # 打印表头
                LOGGER.info("")
                LOGGER.info("State Detection Results:")
                pf = "%22s" + "%11i" * 2 + "%11.3g" * len(self.state_det_metrics.keys)
                
                # 打印总体结果
                total_images = len(set().union(*self.state_class_images.values())) if hasattr(self, 'state_class_images') and self.state_class_images else self.seen
                LOGGER.info(pf % ("all", total_images, int(self.state_nt_per_class.sum()), *self.state_det_metrics.mean_results()))
                
                # 打印每个类别的结果
                if self.args.verbose and not self.training and state_classes > 1:
                    for i, c in enumerate(self.state_det_metrics.ap_class_index):
                        cls_name = state_names.get(c, f"state_{c}")
                        LOGGER.info(
                            pf % (cls_name, self.state_nt_per_image[c], self.state_nt_per_class[c], 
                                *self.state_det_metrics.class_result(i))
                        )
                
                # 恢复原始设置
                self.nc = original_nc
                self.nt_per_class = original_nt_per_class
                self.nt_per_image = original_nt_per_image
        
        # 显示状态预测指标（原有代码）
        if self.state_metrics is not None and self.state_metrics.total > 0:
            # ... 原有的状态预测指标打印代码 ...
            self.state_metrics.update_formatted_metrics()
            
            tp, fp, fn = self.state_metrics.get_tp_fp_fn()
            precision, recall, f1 = self.state_metrics.get_precision_recall_f1()
            
            print(
                f"✅State Prediction Results (IoU={self.state_iou}): 🔄Total samples: {self.state_metrics.total}, "
                f"📈Accuracy: {self.state_metrics.state_accuracy:.4f}, "
                f"📊Macro Accuracy: {self.state_metrics.per_state_accuracy.mean():.4f}"
            )
        else:
            LOGGER.info("No state prediction data available for evaluation")
        
        # ========== 按目标尺寸统计的检测/状态检测结果（基于 IoU=self.state_iou），增加P、FP、FN ==========
        if hasattr(self, "size_bins"):
            LOGGER.info("")
            LOGGER.info("Per-size Detection / State-Detection Metrics (IoU=%.2f):" % self.state_iou)
            # 表头：增加 FP 和 FN 列
            header = "%10s%8s%8s%8s%8s%8s%8s%8s%8s%8s%8s%8s%8s"
            LOGGER.info(header % (
                "size", "det_GT", "det_TP", "det_FP", "det_FN",
                "st_GT", "st_TP", "st_FP", "st_FN",
                "det_P", "det_R", "st_P", "st_R"
            ))
            
            for bi, (rng, name) in enumerate(zip(self.size_bins, self.size_bucket_names)):
                det_c = self.size_det_counts[bi]
                st_c = self.size_state_det_counts[bi]
                det_gt, det_tp = det_c["gt"], det_c["tp"]
                st_gt, st_tp = st_c["gt"], st_c["tp"]
                det_pred = det_c.get("pred", 0)
                st_pred = st_c.get("pred", 0)

                # 计算 FP 和 FN
                det_fp = det_pred - det_tp  # FP = 预测数 - TP
                det_fn = det_gt - det_tp   # FN = GT数 - TP
                st_fp = st_pred - st_tp
                st_fn = st_gt - st_tp

                # 计算 P 和 R
                det_rec = det_tp / det_gt if det_gt > 0 else 0.0
                st_rec = st_tp / st_gt if st_gt > 0 else 0.0
                det_prec = det_tp / det_pred if det_pred > 0 else 0.0
                st_prec = st_tp / st_pred if st_pred > 0 else 0.0

                LOGGER.info(
                    f"{name:>10s}{det_gt:8d}{det_tp:8d}{det_fp:8d}{det_fn:8d}  "
                    f"{st_gt:8d}{st_tp:8d}{st_fp:8d}{st_fn:8d}  "
                    f"{det_prec:8.3f}{det_rec:8.3f}{st_prec:8.3f}{st_rec:8.3f}"
                ) # ========== 按目标尺寸统计的检测/状态检测结果（基于 IoU=self.state_iou），增加P、FP、FN ==========


        # 保存Excel的逻辑
        save_excel = getattr(self, '_save_excel', False)
        if save_excel and not self.training:
            LOGGER.info(f"✅ 开始保存评估结果到Excel (save_excel={save_excel})")
            self._save_results_to_excel()

    def save_one_txt(self, predn, save_conf, shape, file, pred=None):
        """
        修改后的save_one_txt函数，使其处理流程与predict.py完全一致
        """
        from ultralytics.engine.results import Results
        import torch
        import numpy as np
        from pathlib import Path
        
        # 检查是否需要保存状态信息
        save_state_to_txt = getattr(self, '_save_state_to_txt', False)
        
        # 提取状态信息（与predict.py保持一致）
        state_cls_list = None
        if save_state_to_txt and pred is not None:
            if hasattr(self.model.model[-1], "state_classes") and self.model.model[-1].state_classes is not None:
                embed_dim = self.model.model[-1].embed_dim
                state_classes = self.model.model[-1].state_classes
                
                # 检查pred是否包含状态信息
                if pred.shape[1] > 6 + embed_dim and len(pred) > 0:
                    # 直接从pred中提取状态预测，与predict.py完全一致
                    # pred和predn是一一对应的（predn = pred.clone()后scale），不需要IoU匹配
                    states_data = pred[:, 6 + embed_dim:6 + embed_dim + state_classes]
                    
                    # 与predict.py一致：直接argmax，不需要softmax（argmax结果相同）
                    state_ids = states_data.argmax(dim=1)  # (N,)
                    state_cls_list = state_ids.cpu().numpy().tolist()
        
        # 创建与predict.py完全相同的Results对象
        Path(file).parent.mkdir(parents=True, exist_ok=True)
        with open(file, "a") as f:
            # 直接创建与predict.py相同的Results对象
            results = Results(
                np.zeros((shape[0], shape[1]), dtype=np.uint8),
                path=None,
                names=self.names,
                boxes=predn[:, :6],
            )
            
            # 使用与predict.py相同的方式处理每个检测框
            for i, box in enumerate(results.boxes):
                c = int(box.cls)
                conf = float(box.conf)
                
                # 使用相同的xywhn属性获取归一化坐标
                xywhn = box.xywhn[0].cpu().numpy()
                
                # 构建输出行
                line = [c, xywhn[0], xywhn[1], xywhn[2], xywhn[3]]
                if save_conf:
                    line.append(conf)
                
                # 添加状态类别（如果有）- 直接按顺序对应，不需要匹配
                if state_cls_list is not None and i < len(state_cls_list):
                    line.append(state_cls_list[i])
                
                # 写入文件
                f.write(("%g " * len(line)).rstrip() % tuple(line) + "\n")

    def _save_results_to_excel(self):
        """保存评估结果到Excel文件"""
        # 先尝试导入必要的库
        try:
            import pandas as pd
        except ImportError as e:
            LOGGER.warning(f"⚠️ 无法导入pandas: {e}")
            LOGGER.warning(f"⚠️ 请安装pandas: pip install pandas")
            return
        
        try:
            import openpyxl
        except ImportError as e:
            LOGGER.warning(f"⚠️ 无法导入openpyxl: {e}")
            LOGGER.warning(f"⚠️ 请安装openpyxl: pip install openpyxl")
            return
        
        try:
            from pathlib import Path
            
            # 获取模型名称（使用实例变量）
            model_name = getattr(self, '_model_name', 'unknown')
            
            # 获取Excel保存目录（使用实例变量）
            excel_save_dir = getattr(self, '_excel_save_dir', None)
            excel_name = getattr(self, '_excel_name', 'result_all.xlsx')
            if excel_save_dir:
                excel_path = Path(excel_save_dir) / excel_name
            else:
                # 如果没有指定，使用项目目录的父目录
                excel_path = Path(self.save_dir).parent / excel_name
            
            LOGGER.info(f"📝 Excel文件路径: {excel_path}")
            
            # 准备数据字典
            row_data = {}
            
            # 第1列：模型名称
            row_data['Model'] = model_name
            
            # 第2-8列：检测指标 (Images, Instances, P, R, mAP50, mAP75, mAP50-95)
            det_results = self.metrics.mean_results()  # [P, R, mAP50, mAP75, mAP50-95]
            row_data['Images'] = self.seen
            row_data['Instances'] = int(self.nt_per_class.sum())
            row_data['Box_P'] = round(float(det_results[0]), 3) if len(det_results) > 0 else 0.0
            row_data['Box_R'] = round(float(det_results[1]), 3) if len(det_results) > 1 else 0.0
            row_data['Box_mAP50'] = round(float(det_results[2]), 3) if len(det_results) > 2 else 0.0
            row_data['Box_mAP75'] = round(float(det_results[3]), 3) if len(det_results) > 3 else 0.0
            row_data['Box_mAP50-95'] = round(float(det_results[4]), 3) if len(det_results) > 4 else 0.0
            
            # 第9-10列：State Prediction Results (Accuracy, Macro Accuracy)
            if self.state_metrics is not None and self.state_metrics.total > 0:
                row_data['State_Accuracy'] = round(float(self.state_metrics.state_accuracy), 4)
                support = self.state_metrics.state_total_counts > 0
                row_data['State_Macro_Accuracy'] = round(float(self.state_metrics.per_state_accuracy[support].mean()), 4) if support.any() else 0.0
            else:
                row_data['State_Accuracy'] = 0.0
                row_data['State_Macro_Accuracy'] = 0.0
            
            # 第11-13列：State Prediction Metrics (P, R, F1)
            if self.state_metrics is not None and self.state_metrics.total > 0:
                self.state_metrics.update_formatted_metrics()
                state_pred_results = self.state_metrics.mean_results()  # [mp, mr, mf1, 0, 0]
                row_data['State_Pred_P'] = round(float(state_pred_results[0]), 3) if len(state_pred_results) > 0 else 0.0
                row_data['State_Pred_R'] = round(float(state_pred_results[1]), 3) if len(state_pred_results) > 1 else 0.0
                row_data['State_Pred_F1'] = round(float(state_pred_results[2]), 3) if len(state_pred_results) > 2 else 0.0
            else:
                row_data['State_Pred_P'] = 0.0
                row_data['State_Pred_R'] = 0.0
                row_data['State_Pred_F1'] = 0.0
            
            # 第14-18列：State Detection Metrics (pre, rec, mAP50, mAP75, mAP50-95)
            if hasattr(self, "state_det_metrics") and hasattr(self, "state_det_stats") and len(self.state_det_stats) > 0:
                state_det_stats = {k: torch.cat(v, 0).cpu().numpy() if v else np.array([]) 
                                      for k, v in self.state_det_stats.items()}
                if len(state_det_stats) > 0 and state_det_stats["tp"].size > 0 and state_det_stats["tp"].any():
                    state_det_stats.pop("target_img", None)
                    self.state_det_metrics.process(**state_det_stats)
                    state_det_results = self.state_det_metrics.mean_results()  # [P, R, mAP50, mAP75, mAP50-95]
                    row_data['State_Det_P'] = round(float(state_det_results[0]), 3) if len(state_det_results) > 0 else 0.0
                    row_data['State_Det_R'] = round(float(state_det_results[1]), 3) if len(state_det_results) > 1 else 0.0
                    row_data['State_Det_mAP50'] = round(float(state_det_results[2]), 3) if len(state_det_results) > 2 else 0.0
                    row_data['State_Det_mAP75'] = round(float(state_det_results[3]), 3) if len(state_det_results) > 3 else 0.0
                    row_data['State_Det_mAP50-95'] = round(float(state_det_results[4]), 3) if len(state_det_results) > 4 else 0.0
                else:
                    row_data['State_Det_P'] = 0.0
                    row_data['State_Det_R'] = 0.0
                    row_data['State_Det_mAP50'] = 0.0
                    row_data['State_Det_mAP75'] = 0.0
                    row_data['State_Det_mAP50-95'] = 0.0
            else:
                row_data['State_Det_P'] = 0.0
                row_data['State_Det_R'] = 0.0
                row_data['State_Det_mAP50'] = 0.0
                row_data['State_Det_mAP75'] = 0.0
                row_data['State_Det_mAP50-95'] = 0.0
            
            # 读取或创建Excel文件
            if excel_path.exists():
                try:
                    # 读取现有Excel文件
                    df = pd.read_excel(excel_path, sheet_name=0, engine='openpyxl')
                    
                    # 检查是否已存在该模型名称的行
                    if 'Model' in df.columns:
                        model_idx = df[df['Model'] == model_name].index
                        if len(model_idx) > 0:
                            # 更新现有行
                            for col, val in row_data.items():
                                df.at[model_idx[0], col] = val
                            LOGGER.info(f"📝 更新Excel中模型 '{model_name}' 的数据")
                        else:
                            # 添加新行
                            df = pd.concat([df, pd.DataFrame([row_data])], ignore_index=True)
                            LOGGER.info(f"📝 在Excel中添加新模型 '{model_name}' 的数据")
                    else:
                        # 如果Model列不存在，添加新行
                        df = pd.concat([df, pd.DataFrame([row_data])], ignore_index=True)
                        LOGGER.info(f"📝 Excel文件缺少Model列，添加新行")
                except Exception as e:
                    LOGGER.warning(f"⚠️ 读取Excel文件时出错: {e}")
                    # 如果读取失败，创建新的DataFrame
                    df = pd.DataFrame([row_data])
                    LOGGER.info(f"📝 创建新的Excel文件")
            else:
                # 创建新的DataFrame
                df = pd.DataFrame([row_data])
                LOGGER.info(f"📝 创建新的Excel文件")
            
            # 保存到Excel
            df.to_excel(excel_path, index=False, engine='openpyxl')
            LOGGER.info(f"✅ 评估结果已成功保存到Excel: {excel_path}")
            
        except ImportError as e:
            LOGGER.warning(f"⚠️ 导入库时出错: {e}")
            import traceback
            traceback.print_exc()
        except Exception as e:
            LOGGER.warning(f"⚠️ 保存Excel时出错: {e}")
            import traceback
            traceback.print_exc()
