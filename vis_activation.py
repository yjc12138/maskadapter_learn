import torch
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.modeling.postprocessing import sem_seg_postprocess
from torch.nn import functional as F
from maft import (
    MaskFormerSemanticDatasetMapper,
    add_maskformer2_config,
    add_fcclip_config,
    add_mask_adapter_config,
)
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import DatasetCatalog
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

activation_map_storage = {}
clip_vis_dense_storage = {}

def clip_vis_dense_hook(module, input, output):
    features = input[0]  # 获取输入特征
    clip_vis_dense_storage["clip_vis_dense"] = features['clip_vis_dense']

def mask_adapter_hook(module, input, output):
    activation_map_storage["activation_map"] = output

activation_map_handle = model.mask_adapter.register_forward_hook(mask_adapter_hook)
clip_vis_dense_handle = model.sem_seg_head.register_forward_hook(clip_vis_dense_hook)

with torch.no_grad():
    outputs = model(batched_inputs)
    activation_map = activation_map_storage["activation_map"]
    clip_feature = clip_vis_dense_storage["clip_vis_dense"]
    clip_vis_dense = model.visual_prediction_forward_convnext(clip_feature)
    activation_map = activation_map.detach().cpu().squeeze(0).numpy()
    clip_vis_dense = clip_vis_dense.detach().cpu().squeeze(0).numpy()
    print(f"Clip vis dense shape: {clip_vis_dense.shape}")
    print(f"Activation map shape: {activation_map.shape}")

activation_map_handle.remove()
clip_vis_dense_handle.remove()

fig, axes = plt.subplots(10, 10, figsize=(40, 40))
for i in range(100):
    row, col = divmod(i, 10)
    axes[row, col].imshow(activation_map[i*16:(i+1)*16].mean(axis=0))
    axes[row, col].set_title(f"{i}")
    axes[row, col].axis('off')
plt.tight_layout()
plt.savefig("../immediate_vis/activation_map_mean1.png")

fig, axes = plt.subplots(10, 10, figsize=(40, 40))
for i in range(100):
    row, col = divmod(i, 10)
    axes[row, col].imshow(activation_map[i*16:(i+1)*16].max(axis=0))
    axes[row, col].set_title(f"{i}")
    axes[row, col].axis('off')
plt.tight_layout()
plt.savefig("../immediate_vis/activation_map_max1.png")