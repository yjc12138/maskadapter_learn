import os
os.environ["DETECTRON2_DATASETS"] = "/home/Tarkiya/project/NLP/code/yjc/data"
import torch
import os
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
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from maft.maft_plus import MAFT_Plus
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.colors as mcolors
from sklearn.preprocessing import StandardScaler
from scipy import ndimage
seed = 42 
torch.manual_seed(seed)
np.random.seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 确保CUDA操作的确定性，可能会影响性能
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

os.makedirs("../immediate_vis/yjc", exist_ok=True)
os.makedirs("../immediate_vis/yjc/gt_new", exist_ok=True)
path1="../immediate_vis/yjc/pixel5.png"
path2="../immediate_vis/yjc/gt_new/mask_visualization5.png"
path4="../immediate_vis/yjc/gt/mask_visualization5.png"
path3="../immediate_vis/yjc/pred/mask_visualization5.png"


# 定义一个函数来获取文本尺寸，兼容不同版本的PIL
def get_text_dimensions(text, font):
    try:
        # 新版PIL使用font.getbbox或font.getsize
        if hasattr(font, 'getbbox'):
            bbox = font.getbbox(text)
            return bbox[2] - bbox[0], bbox[3] - bbox[1]
        elif hasattr(font, 'getsize'):
            return font.getsize(text)
        else:
            # 如果上述方法都失败，使用估计值
            return len(text) * 10, 20
    except:
        # 如果出现任何错误，使用估计值
        return len(text) * 10, 20

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

# 2. 获取数据集中的一张图片或指定图片
use_specific_image = True  # 设置为True以使用指定图片
specific_image_path = "/home/Tarkiya/project/NLP/code/yjc/data/ADEChallengeData2016/images/validation/ADE_val_00000002.jpg"

if use_specific_image:
    dataset_dicts = DatasetCatalog.get("openvocab_ade20k_panoptic_val")
    sample = None
    for s in dataset_dicts:
        if s["file_name"] == specific_image_path:
            sample = s
            break
    
# 3. 使用 dataset_mapper 做预处理
mapper = MaskFormerSemanticDatasetMapper(cfg, True)
batched_input = mapper(sample)
batched_inputs = [batched_input]

# 存储中间结果
storage = {}

def extract_features_hook(module, input, output):
    # 获取backbone提取的特征
    features = input[0] if isinstance(input, tuple) else input
    if isinstance(features, dict) and 'clip_vis_dense' in features:
        storage["clip_vis_dense"] = features['clip_vis_dense']
        storage["text_classifier"] = features.get('text_classifier', None)

# 注册hook
handle = model.sem_seg_head.register_forward_hook(extract_features_hook)

# 4. 前向推理
with torch.no_grad():
    outputs = model(batched_inputs)
    clip_vis_dense = storage.get("clip_vis_dense")
    text_classifier = storage.get("text_classifier")
    clip_vis_dense = model.visual_prediction_forward_convnext(clip_vis_dense)

handle.remove()

print(f"Clip vis dense shape: {clip_vis_dense.shape}")
print(f"Text classifier shape: {text_classifier.shape}")

# 5. 获取ground truth
gt_sem_seg = None
gt_sem_seg = batched_input["sem_seg"]
gt=gt_sem_seg

print(f"Ground truth shape: {gt_sem_seg.shape}")

# 6. 处理数据维度匹配
B, C, H, W = clip_vis_dense.shape
clip_features = clip_vis_dense.squeeze(0).permute(1, 2, 0).reshape(-1, C)  # (H*W, C)

# 调整ground truth尺寸以匹配clip_vis_dense
if gt_sem_seg.shape != (H, W):
    gt_sem_seg = F.interpolate(
        gt_sem_seg.float().unsqueeze(0).unsqueeze(0), 
        size=(H, W), 
        mode='nearest'
    ).squeeze().long()

gt_labels = gt_sem_seg.reshape(-1)  # (H*W,)

print(f"Clip features shape: {clip_features.shape}")
print(f"GT labels shape: {gt_labels.shape}")

# 7. 获取图像中实际存在的类别
unique_labels = torch.unique(gt_labels)
# 排除背景类 (255) 和 void 类别
unique_labels = unique_labels[(unique_labels < text_classifier.shape[0] - 1) & (unique_labels != 255)]

print(f"Class labels: {unique_labels}")

# 获取类别名称
meta = batched_input.get("meta", {"dataname": "openvocab_ade20k_panoptic_val"})
dataname = meta.get('dataname', "openvocab_ade20k_panoptic_val")

# 尝试从元数据中获取类别名称
metadata = model.test_metadata[dataname]
class_names = metadata.stuff_classes if hasattr(metadata, 'stuff_classes') else metadata.thing_classes
class_name_dict = {i: name for i, name in enumerate(class_names)}

# 8. (修改后) 提取数据集中所有类别的平均文本嵌入
meta = batched_input.get("meta", {"dataname": "openvocab_ade20k_panoptic_val"})
dataname = meta.get('dataname', "openvocab_ade20k_panoptic_val")
_, num_templates = model.get_text_classifier(dataname)

# 计算每个类别的起始索引
class_start_indices = [0]
current_idx = 0
for num in num_templates:
    current_idx += num
    class_start_indices.append(current_idx)

# 提取每个类别的文本嵌入（遍历所有类别）
all_text_embeddings = []
num_total_classes = len(num_templates)

for label_idx in range(num_total_classes):
    start_idx = class_start_indices[label_idx]
    end_idx = class_start_indices[label_idx + 1]
    
    # 获取该类别的所有模板嵌入并平均
    class_embeddings = text_classifier[start_idx:end_idx]
    all_text_embeddings.append(class_embeddings.mean(dim=0))

all_text_embeddings = torch.stack(all_text_embeddings) # (num_total_classes, C)
print(f"Extracted all text embeddings, shape: {all_text_embeddings.shape}")

# 图像中实际存在的类别数量（用于后续绘图）
num_classes_in_image = len(unique_labels)

# 9. 采样像素点（如果像素点太多的话）
max_pixels = 5000
if clip_features.shape[0] > max_pixels:
    sample_indices = torch.randperm(clip_features.shape[0])[:max_pixels]
    clip_features_sampled = clip_features[sample_indices]
    gt_labels_sampled = gt_labels[sample_indices]
else:
    clip_features_sampled = clip_features
    gt_labels_sampled = gt_labels

print(f"Sampled clip features shape: {clip_features_sampled.shape}")

# 10. (修改后) 标准化并合并所有特征
scaler_text = StandardScaler()
scaler_pixels = StandardScaler()

# 10.1 对所有文本特征进行标准化
text_features_scaled = scaler_text.fit_transform(all_text_embeddings.cpu().numpy())

# 10.2 对像素特征进行标准化
pixel_features_scaled = scaler_pixels.fit_transform(clip_features_sampled.cpu().numpy())

# 10.3 合并两组已经标准化过的数据
all_features_scaled = np.concatenate([
    text_features_scaled,
    pixel_features_scaled
], axis=0)


# 11. 执行TSNE降维
# 使用标准化后的特征进行TSNE
tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, all_features_scaled.shape[0]-1))
features_2d = tsne.fit_transform(all_features_scaled)

print(f"TSNE result shape: {features_2d.shape}")

# 12. 分离文本嵌入和像素特征的2D坐标
all_text_coords = features_2d[:num_total_classes]   # 所有文本嵌入的2D坐标
pixel_coords = features_2d[num_total_classes:]      # 像素特征的2D坐标

# 13. (修改后) 可视化
plt.figure(figsize=(15, 12))

# 定义颜色 (根据图中出现的类别数量来定)
colors = plt.cm.tab20(np.linspace(0, 1, num_classes_in_image))

# 过滤掉背景类的像素点
valid_pixel_mask = (gt_labels_sampled != 255)

# 绘制文本嵌入（三角形），只画图中出现的类别
for i, (label_idx_tensor, color) in enumerate(zip(unique_labels, colors)):
    label_idx = label_idx_tensor.item()
    class_name = class_name_dict.get(label_idx, f"Class {label_idx}")
    
    # 使用类别ID从 all_text_coords 中获取正确的坐标
    plt.scatter(all_text_coords[label_idx, 0], all_text_coords[label_idx, 1], 
                c=[color], marker='^', s=300, 
                label=f'{class_name}', 
                edgecolors='black', linewidth=2, zorder=3) # zorder确保在像素点之上

# 绘制像素特征（圆点）
for i, (label_idx_tensor, color) in enumerate(zip(unique_labels, colors)):
    label_idx = label_idx_tensor.item()
    # 找到属于当前类别的像素点
    mask = (gt_labels_sampled == label_idx) & valid_pixel_mask
    if mask.sum() > 0:
        class_name = class_name_dict.get(label_idx, f"Class {label_idx}")
        plt.scatter(pixel_coords[mask, 0], pixel_coords[mask, 1], 
                    c=[color], marker='o', s=20, alpha=0.5,
                    label=f'Pixels: {class_name}')

plt.title('TSNE Visualization: Text Embeddings (All Classes) vs Pixel Features (Image)')
plt.xlabel('TSNE Dimension 1')
plt.ylabel('TSNE Dimension 2')

# 优化图例，避免重复标签
handles, labels = plt.gca().get_legend_handles_labels()
from collections import OrderedDict
by_label = OrderedDict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='upper left')

plt.grid(True, alpha=0.3)
plt.tight_layout()

# 保存图像
plt.savefig(path1, dpi=300, bbox_inches='tight')
plt.close()


# 14. 额外的统计信息
print("\n=== Statistics ===")
print(f"Image file: {batched_input.get('file_name', 'Unknown')}")
print(f"Original image size: {batched_input.get('height', 'Unknown')} x {batched_input.get('width', 'Unknown')}")
# 15. 生成掩码可视化图像
# 创建保存目录
os.makedirs("../immediate_vis/yjc/gt", exist_ok=True)
os.makedirs("../immediate_vis/yjc/pred", exist_ok=True)

# 获取原始图像
original_image_path = batched_input["file_name"]
original_image = Image.open(original_image_path).convert("RGB")
original_width, original_height = original_image.size

# 获取ground truth掩码
gt_mask = gt_sem_seg.cpu().numpy()

# 获取模型预测掩码
if "sem_seg" in outputs[0]:
    pred_mask = outputs[0]["sem_seg"].argmax(dim=0).cpu().numpy()
else:
    print("No semantic segmentation prediction available")
    pred_mask = np.zeros_like(gt_mask)

# 定义颜色映射
cmap = plt.cm.get_cmap('tab20', len(unique_labels))
color_map = {label.item(): tuple(int(c * 255) for c in cmap(i)[:3]) for i, label in enumerate(unique_labels)}

# 尝试加载字体，如果失败则使用默认字体
try:
    font = ImageFont.truetype("DejaVuSans.ttf", 20)
except IOError:
    font = ImageFont.load_default()

# 处理ground truth掩码
gt_vis = np.zeros((gt_mask.shape[0], gt_mask.shape[1], 3), dtype=np.uint8)
gt_centers = {}

# 为每个类别创建掩码
for label_idx in unique_labels:
    label = label_idx.item()
    if label == 255:  # 跳过背景类
        continue
    
    # 创建当前类别的二值掩码
    binary_mask = (gt_mask == label)
    if not np.any(binary_mask):
        continue
    
    # 为当前类别的掩码上色
    color = color_map.get(label, (255, 255, 255))
    for c in range(3):
        gt_vis[:, :, c] = np.where(binary_mask, color[c], gt_vis[:, :, c])
    
    # 计算掩码的中心点
    labeled_mask, num_features = ndimage.label(binary_mask)
    if num_features > 0:
        for i in range(1, num_features + 1):
            component = labeled_mask == i
            if np.sum(component) > 100:  # 只处理足够大的连通区域
                y_indices, x_indices = np.where(component)
                center_y, center_x = int(np.mean(y_indices)), int(np.mean(x_indices))
                class_name = class_name_dict.get(label, f"Class {label}")
                gt_centers[(center_y, center_x)] = class_name

# 将NumPy数组转换为PIL图像
gt_vis_img = Image.fromarray(gt_vis)
draw = ImageDraw.Draw(gt_vis_img)


# 保存ground truth掩码图像
gt_vis_img.save(path2)

gt_mask = gt.cpu().numpy()
gt_vis = np.zeros((gt_mask.shape[0], gt_mask.shape[1], 3), dtype=np.uint8)
gt_centers = {}

# 为每个类别创建掩码
for label_idx in unique_labels:
    label = label_idx.item()
    if label == 255:  # 跳过背景类
        continue
    
    # 创建当前类别的二值掩码
    binary_mask = (gt_mask == label)
    if not np.any(binary_mask):
        continue
    
    # 为当前类别的掩码上色
    color = color_map.get(label, (255, 255, 255))
    for c in range(3):
        gt_vis[:, :, c] = np.where(binary_mask, color[c], gt_vis[:, :, c])
    
    # 计算掩码的中心点
    labeled_mask, num_features = ndimage.label(binary_mask)
    if num_features > 0:
        for i in range(1, num_features + 1):
            component = labeled_mask == i
            if np.sum(component) > 100:  # 只处理足够大的连通区域
                y_indices, x_indices = np.where(component)
                center_y, center_x = int(np.mean(y_indices)), int(np.mean(x_indices))
                class_name = class_name_dict.get(label, f"Class {label}")
                gt_centers[(center_y, center_x)] = class_name

# 将NumPy数组转换为PIL图像
gt_vis_img = Image.fromarray(gt_vis)
draw = ImageDraw.Draw(gt_vis_img)


# 保存ground truth掩码图像
gt_vis_img.save(path4)

# 处理预测掩码
pred_vis = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
pred_centers = {}

# 为每个类别创建掩码
for label_idx in unique_labels:
    label = label_idx.item()
    if label == 255:  # 跳过背景类
        continue
    
    # 创建当前类别的二值掩码
    binary_mask = (pred_mask == label)
    if not np.any(binary_mask):
        continue
    
    # 为当前类别的掩码上色
    color = color_map.get(label, (255, 255, 255))
    for c in range(3):
        pred_vis[:, :, c] = np.where(binary_mask, color[c], pred_vis[:, :, c])
    
    # 计算掩码的中心点
    labeled_mask, num_features = ndimage.label(binary_mask)
    if num_features > 0:
        for i in range(1, num_features + 1):
            component = labeled_mask == i
            if np.sum(component) > 100:  # 只处理足够大的连通区域
                y_indices, x_indices = np.where(component)
                center_y, center_x = int(np.mean(y_indices)), int(np.mean(x_indices))
                class_name = class_name_dict.get(label, f"Class {label}")
                pred_centers[(center_y, center_x)] = class_name

pred_vis_img = Image.fromarray(pred_vis)
draw = ImageDraw.Draw(pred_vis_img)

pred_vis_img.save(path3)

