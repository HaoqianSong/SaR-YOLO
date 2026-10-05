# Ultralytics YOLO 🚀, AGPL-3.0 license

from copy import copy

from ultralytics.models import yolo
from ultralytics.nn.tasks import JDEModel
from ultralytics.utils import DEFAULT_CFG, RANK
from ultralytics.utils.plotting import plot_images, plot_results


class JDETrainer(yolo.detect.DetectionTrainer):
    """
    A class extending the DetectionTrainer class for training based on a joint detection and embedding model.

    Example:
        ```python
        from ultralytics.models.yolo.jde import JDETrainer

        args = dict(model="yolov8n-jde.pt", data="coco8-seg.yaml", epochs=3)
        trainer = JDETrainer(overrides=args)
        trainer.train()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize a SegmentationTrainer object with given arguments."""
        if overrides is None:
            overrides = {}
        #$#overrides["task"] = "jde"
        super().__init__(cfg, overrides, _callbacks)
        #self.model.person_states = self.data.get("person_states", {})

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return SegmentationModel initialized with specified config and weights."""
        model = JDEModel(cfg, ch=3, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)

        return model

    def get_validator(self):
        """Return an instance of SegmentationValidator for validation of YOLO model."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss", "emb_loss", "state_loss"  # 添加state_loss #￥#添加人员状态预测评估指标
        return yolo.jde.JDEValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """构建dataloader; train模式下扫描全数据集统计各类姿态样本数, 注入到loss供CB权重和自适应margin使用(静态全局统计).
        state_counts_dynamic=True(动态EMA计数)时跳过扫描注入, 由loss内部每batch跟踪实际采样分布."""
        loader = super().get_dataloader(dataset_path, batch_size, rank, mode)
        if mode == "train" and not getattr(self.args, 'state_counts_dynamic', False):
            self._inject_state_class_counts(loader.dataset)
        return loader

    def _inject_state_class_counts(self, dataset):
        """扫描训练集labels统计每类state样本数, 调用criterion.set_class_counts()注入(静态, 训练前一次性)."""
        import numpy as np
        import torch
        from ultralytics.utils.torch_utils import de_parallel

        model = de_parallel(self.model)
        criterion = getattr(model, "criterion", None)
        if criterion is None or not hasattr(criterion, "set_class_counts") or getattr(criterion, "state_classes", None) is None:
            return
        counts = torch.zeros(criterion.state_classes, dtype=torch.long)
        labels = getattr(dataset, "labels", None)
        if labels:
            for lb in labels:
                s = lb.get("states")
                if s is None:
                    continue
                s = np.asarray(s).reshape(-1).astype(int)
                s = s[(s >= 0) & (s < criterion.state_classes)]  # 过滤越界值
                if len(s):
                    counts += torch.bincount(torch.as_tensor(s), minlength=criterion.state_classes)
        criterion.set_class_counts(counts)
    
    def label_loss_items(self, loss_items=None, prefix="train"): #￥#添加人员状态预测评估指标
        """返回带标签的损失项字典，包括状态损失"""
        keys = [f"{prefix}/{x}" for x in self.loss_names] #￥#添加人员状态预测评估指标
        if loss_items is not None: #￥#添加人员状态预测评估指标
            loss_items = [round(float(x), 5) for x in loss_items] #￥#添加人员状态预测评估指标
            return dict(zip(keys, loss_items)) #￥#添加人员状态预测评估指标
        else: #￥#添加人员状态预测评估指标
            return keys #￥#添加人员状态预测评估指标

    def plot_training_samples(self, batch, ni):
        """Plot training samples with annotations."""
        states_or_cls = batch.get("states", batch["cls"]).squeeze(-1) # 使用get获取states，如果不存在就使用cls
        plot_images(
            images=batch["img"],
            batch_idx=batch["batch_idx"],
            cls=states_or_cls,  # batch["states"].squeeze(-1),
            bboxes=batch["bboxes"],
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )

    def set_model_attributes(self):
        """设置JDE模型属性，包括names和person_states"""
        super().set_model_attributes()  # 调用父类方法
        
        # 确保person_states被正确设置到模型
        if hasattr(self, 'data') and self.data and 'person_states' in self.data:
            person_states = self.data["person_states"]
            self.model.person_states = person_states
            #print(f"JDETrainer: 成功设置person_states到模型: {person_states}")
        else:
            self.model.person_states = {}
            print("JDETrainer: 未找到person_states数据，设置为空字典")
        
        # 如果是DDP包装的模型，也设置到module中
        if hasattr(self.model, 'module'):
            self.model.module.person_states = getattr(self.model, 'person_states', {})
            #print(f"JDETrainer: 也设置person_states到DDP模块: {getattr(self.model, 'person_states', {})}")
