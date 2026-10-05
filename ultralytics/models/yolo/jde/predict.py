# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.engine.results import Results
from ultralytics.engine.predictor import BasePredictor
from ultralytics.utils import DEFAULT_CFG, ops
from ultralytics.data.utils import check_det_dataset
import torch


class JDEPredictor(BasePredictor):
    """
    A class extending the DetectionPredictor class for prediction based on a joint detection and embedding model.

    Example:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.jde import JDEPredictor

        args = dict(model="yolov8n-jde.pt", source=ASSETS)
        predictor = JDEPredictor(overrides=args)
        predictor.predict_cli()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initializes the JDEPredictor with the provided configuration, overrides, and callbacks."""
        super().__init__(cfg, overrides, _callbacks)

    def postprocess(self, preds, img, orig_imgs):
        """Applies non-max suppression and processes detections for each image in an input batch."""
        preds = ops.non_max_suppression(
            preds[0],
            self.args.conf,
            self.args.iou,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=len(self.model.names),
            classes=self.args.classes,
        )

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0]):
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            
            #￥#添加状态id预测 修改：检查是否有状态预测
            if hasattr(self.model.model.model[-1], 'embed_dim') and self.model.model.model[-1].state_classes is not None:        
                # 有状态预测的情况：[bbox(4) + conf(1) + cls(1) + embeds + states]
                embed_dim = self.model.model.model[-1].embed_dim  #256 
                state_classes = self.model.model.model[-1].state_classes  #166 
                # 分离各部分
                boxes_data = pred[:, :6]  # [x1, y1, x2, y2, conf, cls]
                embeds_data = pred[:, 6:6+embed_dim]  # embeddings
                states_data = pred[:, 6+embed_dim:6+embed_dim+state_classes]  # state predictions
                
                # 将状态预测转换为ID（选择概率最高的状态作为ID）
                if len(states_data) > 0:
                    state_ids = states_data.argmax(dim=1).unsqueeze(1)  # 获取最大概率的状态ID
                    # 将状态ID添加到boxes数据中，形成7列格式以启用tracking
                    boxes_with_ids = torch.cat([boxes_data[:, :4], state_ids, boxes_data[:, 4:]], dim=1)
                    results.append(Results(orig_img, path=img_path, names=self.model.names, 
                                         person_states=self.model.person_states, 
                                         boxes=boxes_with_ids, embeds=embeds_data))
                else:
                    results.append(Results(orig_img, path=img_path, names=self.model.names, 
                                         person_states=self.model.person_states, 
                                         boxes=boxes_data, embeds=embeds_data))
            else:
                # 原有逻辑：没有状态预测
                results.append(Results(orig_img, path=img_path, names=self.model.names, 
                                     person_states=self.model.person_states, 
                                     boxes=pred[:, :6], embeds=pred[:, 6:]))#￥#添加状态id预测
        return results
