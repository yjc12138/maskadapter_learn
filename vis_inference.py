import torch
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from torch.nn import functional as F
from maft import (
    MaskFormerSemanticDatasetMapper,
    add_maskformer2_config,
    add_fcclip_config,
    add_mask_adapter_config,
)
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import DatasetCatalog, MetadataCatalog, DatasetMapper
from PIL import Image
import numpy as np
from maft.maft_plus import MAFT_Plus
import matplotlib.pyplot as plt

# 1. 加载配置和模型
cfg = get_cfg()
add_deeplab_config(cfg)
add_maskformer2_config(cfg)
add_fcclip_config(cfg)
add_mask_adapter_config(cfg)
cfg.merge_from_file('./configs/mixed-mask-training/maftp/semantic/train_semantic_large_eval_a150.yaml')
cfg.MODEL.WEIGHTS = './checkpoint/maftp_convnext_large_maskadapter.pth'
cfg.freeze()

model = MAFT_Plus(cfg)
DetectionCheckpointer(model).resume_or_load(cfg.MODEL.WEIGHTS, resume=False)
model.eval()
model.cuda()

# 2. 获取数据集中的一张图片
dataset_dicts = DatasetCatalog.get("openvocab_ade20k_panoptic_val")
sample = dataset_dicts[0]

# 3. 使用 dataset_mapper 做预处理
mapper = MaskFormerSemanticDatasetMapper(cfg, True)
batched_input = mapper(sample)
batched_inputs = [batched_input]

image_path = batched_input["file_name"]
image = Image.open(image_path).convert("RGB")
image_np = np.array(image)
print(f"Image shape: {image_np.shape}")  # 输出形状以确认

mask_pred_storage = {}
def hook_fn(module, input, output):
    mask_pred_storage["mask_pred"] = output["pred_masks"].detach().cpu()

hook_handle = model.sem_seg_head.register_forward_hook(hook_fn)

with torch.no_grad():
    outputs = model(batched_inputs)
    mask_pred = mask_pred_storage["mask_pred"]
    print(f"Mask prediction shape1: {mask_pred.shape}")  # 输出形状以确认
    mask_pred = F.interpolate(
        mask_pred,
        size=(image_np.shape[0], image_np.shape[1]),
        mode="bilinear",
        align_corners=False,
    )
    mask_pred = mask_pred.sigmoid().cpu().numpy().squeeze(0)  # [1, C, H, W] -> [C, H, W]
    print(f"Mask prediction shape2: {mask_pred.shape}")  # 输出形状以确认

hook_handle.remove()

fig, axes = plt.subplots(10, 10, figsize=(20, 20))
for i in range(100):
    row, col = divmod(i, 10)
    axes[row, col].imshow(mask_pred[i])
    axes[row, col].set_title(f"{i}")
    axes[row, col].axis('off')
plt.tight_layout()
plt.savefig("../immediate_vis/semantic_mask_pred.png")

# 5. 可视化/保存分割结果
# 假设 outputs[0]["sem_seg"] 为语义分割结果，形状 [num_classes, H, W]
sem_seg = outputs[0]["sem_seg"].cpu().numpy()  # 转为 NumPy 数组
panoptic_seg = outputs[0]["panoptic_seg"][0].cpu().numpy()  # 转为 NumPy 数组
print(sem_seg.shape)  # 输出形状以确认
print(panoptic_seg.shape)  # 输出形状以确认
sem_pred_mask = np.argmax(sem_seg, axis=0)  # [H, W]
panoptic_pred_mask = panoptic_seg  # [H, W]

plt.figure(figsize=(10,5))
plt.subplot(1, 2, 1)
plt.imshow(image_np)
plt.title("Input Image")
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(sem_pred_mask)
plt.title("Semantic Predicted Mask")
plt.axis('off')
plt.savefig("../immediate_vis/semantic_output.png")

plt.figure(figsize=(10,5))
plt.subplot(1, 2, 1)
plt.imshow(image_np)
plt.title("Input Image")
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(panoptic_pred_mask)
plt.title("Panoptic Predicted Mask")
plt.axis('off')
plt.savefig("../immediate_vis/panoptic_output.png")