"""
这个文件可能已被字节跳动有限公司及其关联公司修改（"字节跳动的修改"）。
所有字节跳动的修改均为字节跳动有限公司及其关联公司的版权所有。

参考：https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/pixel_decoder/msdeformattn.py
"""

# 导入日志模块，用于记录信息
import logging
# 导入numpy，用于数值计算
import numpy as np
# 导入类型提示相关模块
from typing import Callable, Dict, List, Optional, Tuple, Union

# 导入fvcore中的权重初始化工具
import fvcore.nn.weight_init as weight_init
# 导入PyTorch
import torch
# 导入PyTorch的神经网络模块
from torch import nn
# 导入PyTorch的函数式接口
from torch.nn import functional as F
# 导入PyTorch的初始化函数
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
# 导入PyTorch的自动混合精度训练工具
from torch.cuda.amp import autocast

# 导入Detectron2的配置工具
from detectron2.config import configurable
# 导入Detectron2的层
from detectron2.layers import Conv2d, ShapeSpec, get_norm
# 导入Detectron2的模型注册表
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

# 导入位置编码模块
from ..transformer_decoder.position_encoding import PositionEmbeddingSine
# 导入可变形注意力模块
from .ops.modules import MSDeformAttn
# 导入复制模块
import copy


def _get_clones(module, N):
    # 创建模块的N个深度拷贝，并返回为ModuleList
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """返回给定字符串对应的激活函数"""
    if activation == "relu":
        # 返回ReLU激活函数
        return F.relu
    if activation == "gelu":
        # 返回GELU激活函数
        return F.gelu
    if activation == "glu":
        # 返回GLU激活函数
        return F.glu
    # 如果不是上述激活函数，则抛出运行时错误
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


def build_pixel_decoder(cfg, input_shape):
    """
    从`cfg.MODEL.ONE_FORMER.PIXEL_DECODER_NAME`构建像素解码器。
    """
    # 从配置中获取像素解码器名称
    name = cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME
    # 使用注册表获取模型
    model = SEM_SEG_HEADS_REGISTRY.get(name)(cfg, input_shape)
    # 获取forward_features方法
    forward_features = getattr(model, "forward_features", None)
    # 检查forward_features是否可调用
    if not callable(forward_features):
        # 如果不可调用，则抛出错误
        raise ValueError(
            "Only SEM_SEG_HEADS with forward_features method can be used as pixel decoder. "
            f"Please implement forward_features for {name} to only return mask features."
        )
    # 返回构建的模型
    return model


# 可变形DETR中的MSDeformAttn Transformer编码器
class MSDeformAttnTransformerEncoderOnly(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu",
                 num_feature_levels=4, enc_n_points=4,
        ):
        # 初始化父类
        super().__init__()

        # 模型维度
        self.d_model = d_model
        # 注意力头数
        self.nhead = nhead

        # 创建编码器层
        encoder_layer = MSDeformAttnTransformerEncoderLayer(d_model, dim_feedforward,
                                                            dropout, activation,
                                                            num_feature_levels, nhead, enc_n_points)
        # 创建编码器
        self.encoder = MSDeformAttnTransformerEncoder(encoder_layer, num_encoder_layers)

        # 创建层级嵌入参数
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        # 重置参数
        self._reset_parameters()

    def _reset_parameters(self):
        # 对所有参数进行初始化
        for p in self.parameters():
            if p.dim() > 1:
                # 使用xavier均匀初始化多维参数
                nn.init.xavier_uniform_(p)
        # 对所有MSDeformAttn模块进行参数重置
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        # 对level_embed使用正态分布初始化
        normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        # 获取掩码形状
        _, H, W = mask.shape
        # 计算每个样本在高度方向上的有效像素数
        valid_H = torch.sum(~mask[:, :, 0], 1)
        # 计算每个样本在宽度方向上的有效像素数
        valid_W = torch.sum(~mask[:, 0, :], 1)
        # 计算高度方向上的有效比例
        valid_ratio_h = valid_H.float() / H
        # 计算宽度方向上的有效比例
        valid_ratio_w = valid_W.float() / W
        # 将宽高比例堆叠
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        # 返回有效比例
        return valid_ratio

    def forward(self, srcs, pos_embeds):
        # 为每个特征层创建全零掩码
        masks = [torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool) for x in srcs]
        # 准备编码器的输入
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        # 遍历每个特征层
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            # 获取批次大小、通道数、高度和宽度
            bs, c, h, w = src.shape
            # 记录空间形状
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            # 将特征展平并转置
            src = src.flatten(2).transpose(1, 2)
            # 将掩码展平
            mask = mask.flatten(1)
            # 将位置嵌入展平并转置
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            # 添加层级嵌入
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            # 添加到列表中
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        # 连接所有特征层的展平结果
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        # 将空间形状转换为张量
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        # 计算每个层级的起始索引
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        # 计算有效比例
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # 通过编码器处理
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)

        # 返回结果
        return memory, spatial_shapes, level_start_index


class MSDeformAttnTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        # 初始化父类
        super().__init__()

        # 自注意力层
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        # dropout层
        self.dropout1 = nn.Dropout(dropout)
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)

        # 前馈网络
        self.linear1 = nn.Linear(d_model, d_ffn)
        # 激活函数
        self.activation = _get_activation_fn(activation)
        # dropout层
        self.dropout2 = nn.Dropout(dropout)
        # 线性层
        self.linear2 = nn.Linear(d_ffn, d_model)
        # dropout层
        self.dropout3 = nn.Dropout(dropout)
        # 层归一化
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        # 如果位置嵌入为None，则直接返回张量；否则返回张量与位置嵌入的和
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        # 前馈网络的前向传播
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        # 残差连接
        src = src + self.dropout3(src2)
        # 层归一化
        src = self.norm2(src)
        # 返回结果
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # 自注意力
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        # 残差连接
        src = src + self.dropout1(src2)
        # 层归一化
        src = self.norm1(src)

        # 前馈网络
        src = self.forward_ffn(src)

        # 返回结果
        return src


class MSDeformAttnTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        # 初始化父类
        super().__init__()
        # 创建多个编码器层
        self.layers = _get_clones(encoder_layer, num_layers)
        # 记录层数
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        # 创建参考点列表
        reference_points_list = []
        # 遍历每个特征层
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            # 创建网格
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            # 归一化y坐标
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            # 归一化x坐标
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            # 堆叠x和y坐标
            ref = torch.stack((ref_x, ref_y), -1)
            # 添加到列表中
            reference_points_list.append(ref)
        # 连接所有层的参考点
        reference_points = torch.cat(reference_points_list, 1)
        # 根据有效比例调整参考点
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        # 返回参考点
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        # 初始化输出为输入
        output = src
        # 获取参考点
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        # 遍历每一层
        for _, layer in enumerate(self.layers):
            # 通过当前层处理
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)

        # 返回最终输出
        return output


@SEM_SEG_HEADS_REGISTRY.register()
class MSDeformAttnPixelDecoder(nn.Module):
    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        transformer_dropout: float,
        transformer_nheads: int,
        transformer_dim_feedforward: int,
        transformer_enc_layers: int,
        conv_dim: int,
        mask_dim: int,
        norm: Optional[Union[str, Callable]] = None,
        # 可变形transformer编码器参数
        transformer_in_features: List[str],
        common_stride: int,
    ):
        """
        注意：这个接口是实验性的。
        参数:
            input_shape: 输入特征的形状（通道和步长）
            transformer_dropout: transformer中的dropout概率
            transformer_nheads: transformer中的头数
            transformer_dim_feedforward: 前馈网络的维度
            transformer_enc_layers: transformer编码器层数
            conv_dims: 中间卷积层的输出通道数
            mask_dim: 最终卷积层的输出通道数
            norm (str或callable): 所有卷积层的归一化
        """
        # 初始化父类
        super().__init__()
        # 获取transformer输入形状
        transformer_input_shape = {
            k: v for k, v in input_shape.items() if k in transformer_in_features
        }

        # 这是像素解码器的输入形状
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        # 获取特征名称，从"res2"到"res5"
        self.in_features = [k for k, v in input_shape]
        # 获取特征步长
        self.feature_strides = [v.stride for k, v in input_shape]
        # 获取特征通道数
        self.feature_channels = [v.channels for k, v in input_shape]
        
        # 这是transformer编码器的输入形状（可能使用比像素解码器更少的特征）
        transformer_input_shape = sorted(transformer_input_shape.items(), key=lambda x: x[1].stride)
        # 获取transformer特征名称，从"res2"到"res5"
        self.transformer_in_features = [k for k, v in transformer_input_shape]
        # 获取transformer输入通道数
        transformer_in_channels = [v.channels for k, v in transformer_input_shape]
        # 获取transformer特征步长，用于决定额外的FPN层
        self.transformer_feature_strides = [v.stride for k, v in transformer_input_shape]

        # 获取transformer特征层数
        self.transformer_num_feature_levels = len(self.transformer_in_features)
        # 如果有多个特征层
        if self.transformer_num_feature_levels > 1:
            input_proj_list = []
            # 从低分辨率到高分辨率（res5 -> res2）
            for in_channels in transformer_in_channels[::-1]:
                # 创建投影层
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, conv_dim, kernel_size=1),
                    nn.GroupNorm(32, conv_dim),
                ))
            # 创建投影层列表
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            # 只有一个特征层时的投影
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(transformer_in_channels[-1], conv_dim, kernel_size=1),
                    nn.GroupNorm(32, conv_dim),
                )])

        # 初始化投影层权重
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # 创建transformer
        self.transformer = MSDeformAttnTransformerEncoderOnly(
            d_model=conv_dim,
            dropout=transformer_dropout,
            nhead=transformer_nheads,
            dim_feedforward=transformer_dim_feedforward,
            num_encoder_layers=transformer_enc_layers,
            num_feature_levels=self.transformer_num_feature_levels,
        )
        # 设置位置编码步数
        N_steps = conv_dim // 2
        # 创建位置编码层
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        # 设置掩码维度
        self.mask_dim = mask_dim
        # 使用1x1卷积
        self.mask_features = Conv2d(
            conv_dim,
            mask_dim,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        # 初始化权重
        weight_init.c2_xavier_fill(self.mask_features)
        
        # 始终使用3个尺度
        self.maskformer_num_feature_levels = 3
        # 设置公共步长
        self.common_stride = common_stride

        # 额外的FPN层
        stride = min(self.transformer_feature_strides)
        # 计算FPN层数
        self.num_fpn_levels = int(np.log2(stride) - np.log2(self.common_stride))

        # 创建侧向卷积和输出卷积列表
        lateral_convs = []
        output_convs = []

        # 设置是否使用偏置
        use_bias = norm == ""
        # 遍历每个FPN层
        for idx, in_channels in enumerate(self.feature_channels[:self.num_fpn_levels]):
            # 创建侧向卷积的归一化层
            lateral_norm = get_norm(norm, conv_dim)
            # 创建输出卷积的归一化层
            output_norm = get_norm(norm, conv_dim)

            # 创建侧向卷积
            lateral_conv = Conv2d(
                in_channels, conv_dim, kernel_size=1, bias=use_bias, norm=lateral_norm
            )
            # 创建输出卷积
            output_conv = Conv2d(
                conv_dim,
                conv_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
                norm=output_norm,
                activation=F.relu,
            )
            # 初始化权重
            weight_init.c2_xavier_fill(lateral_conv)
            weight_init.c2_xavier_fill(output_conv)
            # 添加模块
            self.add_module("adapter_{}".format(idx + 1), lateral_conv)
            self.add_module("layer_{}".format(idx + 1), output_conv)

            # 添加到列表中
            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
        # 将卷积放入自顶向下的顺序（从低到高分辨率）
        # 使前向计算更清晰
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        # 从配置创建返回字典
        ret = {}
        # 设置输入形状
        ret["input_shape"] = {
            k: v for k, v in input_shape.items() if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
        }
        # 设置卷积维度
        ret["conv_dim"] = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        # 设置掩码维度
        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        # 设置归一化类型
        ret["norm"] = cfg.MODEL.SEM_SEG_HEAD.NORM
        # 设置transformer dropout
        ret["transformer_dropout"] = cfg.MODEL.MASK_FORMER.DROPOUT
        # 设置transformer头数
        ret["transformer_nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
        # 使用1024作为可变形transformer编码器的前馈维度
        ret["transformer_dim_feedforward"] = 1024
        # 设置transformer编码器层数
        ret[
            "transformer_enc_layers"
        ] = cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS
        # 设置transformer输入特征
        ret["transformer_in_features"] = cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES
        # 设置公共步长
        ret["common_stride"] = cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE
        # 返回配置字典
        return ret

    @torch.amp.autocast('cuda',enabled=False)
    def forward_features(self, features):
        # 创建源特征列表
        srcs = []
        # 创建位置编码列表
        pos = []
        # 将特征图反转为自顶向下顺序（从低到高分辨率）
        for idx, f in enumerate(self.transformer_in_features[::-1]):
            # 获取特征并转换为float类型（可变形detr不支持半精度）
            x = features[f].float()
            # 应用投影
            srcs.append(self.input_proj[idx](x))
            # 计算位置编码
            pos.append(self.pe_layer(x))

        # 通过transformer处理
        y, spatial_shapes, level_start_index = self.transformer(srcs, pos)
        # 获取批次大小
        bs = y.shape[0]

        # 准备分割大小或部分
        split_size_or_sections = [None] * self.transformer_num_feature_levels
        # 计算每个层级的分割大小
        for i in range(self.transformer_num_feature_levels):
            if i < self.transformer_num_feature_levels - 1:
                split_size_or_sections[i] = level_start_index[i + 1] - level_start_index[i]
            else:
                split_size_or_sections[i] = y.shape[1] - level_start_index[i]
        # 按维度1分割y
        y = torch.split(y, split_size_or_sections, dim=1)

        # 创建输出列表
        out = []
        # 创建多尺度特征列表
        multi_scale_features = []
        # 初始化当前层级数
        num_cur_levels = 0
        # 遍历每个分割后的y
        for i, z in enumerate(y):
            # 重塑并添加到输出列表
            out.append(z.transpose(1, 2).view(bs, -1, spatial_shapes[i][0], spatial_shapes[i][1]))

        # 使用额外的FPN层扩展`out`
        # 将特征图反转为自顶向下顺序（从低到高分辨率）
        for idx, f in enumerate(self.in_features[:self.num_fpn_levels][::-1]):
            # 获取特征并转换为float类型
            x = features[f].float()
            # 获取侧向卷积
            lateral_conv = self.lateral_convs[idx]
            # 获取输出卷积
            output_conv = self.output_convs[idx]
            # 应用侧向卷积
            cur_fpn = lateral_conv(x)
            # 按照FPN实现，这里使用最近邻上采样
            y = cur_fpn + F.interpolate(out[-1], size=cur_fpn.shape[-2:], mode="bilinear", align_corners=False)
            # 应用输出卷积
            y = output_conv(y)
            # 添加到输出列表
            out.append(y)

        # 收集多尺度特征
        for o in out:
            if num_cur_levels < self.maskformer_num_feature_levels:
                multi_scale_features.append(o)
                num_cur_levels += 1

        # 返回掩码特征、第一个输出和多尺度特征
        return self.mask_features(out[-1]), out[0], multi_scale_features
