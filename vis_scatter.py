import os
os.environ["DETECTRON2_DATASETS"] = "/home/Tarkiya/project/NLP/code/yjc/data"
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
from sklearn.manifold import TSNE

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

hook_storage = {}

def hook_fn(module, input, output):
    features = input[0]  # 获取输入特征
    hook_storage["clip_vis_dense"] = features['clip_vis_dense']

hook_handle = model.sem_seg_head.register_forward_hook(hook_fn)

with torch.no_grad():
    outputs = model(batched_inputs)
    clip_vis_dense = hook_storage["clip_vis_dense"]
    clip_vis_dense = model.visual_prediction_forward_convnext(clip_vis_dense)
    sem_seg = outputs[0]['sem_seg']
    print(sem_seg.shape)
    clip_vis_dense = F.interpolate(
        clip_vis_dense,
        size=sem_seg.shape[-2:],
        mode='bilinear',
        align_corners=False
    )
    clip_vis_dense = clip_vis_dense.squeeze(0).detach().cpu().numpy()
    sem_seg = sem_seg.cpu().numpy()
    print(f"Clip vis dense shape: {clip_vis_dense.shape}")
    print(f"Semantic segmentation results shape: {sem_seg.shape}")

hook_handle.remove()

C, H, W  = clip_vis_dense.shape
features = clip_vis_dense.reshape(C, -1)

sem_seg = sem_seg.argmax(axis=0)
sem_seg_flat = sem_seg.reshape(-1)
M = int(sem_seg_flat.max()) + 1

onehot = np.eye(M, dtype=np.float32)[sem_seg_flat]

sum_act = features @ onehot
counts = onehot.sum(axis=0, keepdims=True)
avg_act = sum_act / (counts + 1e-6)

labels = avg_act.argmax(axis=1)
print(f"Labels shape: {labels.shape}")

tsne = TSNE(n_components=2)
embeddings = tsne.fit_transform(features)

plt.figure(figsize=(30, 10))
plt.subplot(1, 3, 1)
plt.imshow(image_np)
plt.title("Input Image")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(sem_seg)
plt.title("Semantic Segmentation")
plt.axis('off')

plt.subplot(1, 3, 3)
scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, cmap='tab20', s=20, alpha=0.8)
plt.title("t-SNE Visualization of Clip Vis Dense Features")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.tight_layout()
plt.savefig("../immediate_vis/clip_vis_dense_tsne2.png")