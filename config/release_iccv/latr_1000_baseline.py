import numpy as np
from mmcv.utils import Config

_base_ = [
    '../_base_/base_res101_bs16xep100.py',
    '../_base_/optimizer.py'
]

mod = 'release_iccv/latr_1000_baseline'
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

dataset = '1000'
dataset_dir = '/nvmedata/lizhiqi/lane_3d_mini/images/'
data_dir = '/nvmedata/lizhiqi/lane_3d_mini/lane3d_1000/'

batch_size = 8
nworkers = 8
num_category = 21
pos_threshold = 0.3
top_view_region = np.array([
    [-10, 103], [10, 103], [-10, 3], [10, 3]])
enlarge_length = 20
position_range = [
    top_view_region[0][0] - enlarge_length,
    top_view_region[2][1] - enlarge_length,
    -5,
    top_view_region[1][0] + enlarge_length,
    top_view_region[0][1] + enlarge_length,
    5.]
anchor_y_steps = np.linspace(3, 103, 20)
num_y_steps = len(anchor_y_steps)

# extra aug
photo_aug = dict(
    brightness_delta=32 // 2,
    contrast_range=(0.5, 1.5),
    saturation_range=(0.5, 1.5),
    hue_delta=9)

clip_grad_norm = 20.0

_dim_ = 256
num_query = 40
num_pt_per_line = 20
latr_cfg = dict(
    fpn_dim = _dim_,
    num_query = num_query,
    num_group = 1,
    sparse_num_group = 4,
    encoder = dict(
        type='ResNet',#type： 模型类型，这里指定为 'ResNet'，表示要构建一个 ResNet 模型。
        depth=50,#depth： ResNet 的深度，这里设置为 50，表示使用 ResNet-50 结构。
        num_stages=4,#num_stages： ResNet 的阶段数，即残差块的数量，默认为 4。
        out_indices=(1, 2, 3),#out_indices： 输出的阶段索引，这里设置为 (1, 2, 3)，表示输出 ResNet 的第 2、3、4 个阶段的特征图。
        frozen_stages=1,# frozen_stages： 冻结的阶段数，即前几个阶段的参数是否冻结不更新。这里设置为 1，表示只冻结 ResNet 的第一个阶段。
        norm_cfg=dict(type='BN2d', requires_grad=False),#norm_cfg： 归一化层的配置，这里指定为 Batch Normalization，并且设置为不需要梯度更新。
        norm_eval=True,# norm_eval： 归一化层是否在评估模式下，这里设置为 True，表示在评估模式下使用 Batch Normalization。
        style='caffe',#style： ResNet 的风格，可以选择 'caffe' 或 'pytorch'，这里选择了 'caffe' 风格。
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),#dcn： 双线性插值卷积（Deformable Convolution）的配置，这里指定为 DCNv2 类型，设置了 deformable groups 为 1，并且在 stride 不为 1 时不使用 Fallback。这表示在 ResNet 的第 3 和第 4 个阶段使用了双线性插值卷积。
        stage_with_dcn=(False, False, True, True),#stage_with_dcn： 指示每个阶段是否使用双线性插值卷积，这里设置为 (False, False, True, True)，表示在 ResNet 的第 3 和第 4 个阶段使用了双线性插值卷积。
        init_cfg=dict(type='Pretrained', checkpoint='/home/lizhiqi/LATR/resnet50-19c8e357.pth')#init_cfg： 模型初始化的配置，这里指定了使用预训练的模型参数，通过 'torchvision://resnet50' 指定了预训练模型的地址，即使用 PyTorch 中 torchvision 提供的预训练的 ResNet-50 模型参数。
    ),
    neck = dict(
        type='FPN',
        in_channels=[512, 1024, 2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=4,
        relu_before_extra_convs=True
    ),
    head=dict(
        pt_as_query=True,
        num_pt_per_line=num_pt_per_line,
        xs_loss_weight=2.0,
        zs_loss_weight=10.0,
        vis_loss_weight=1.0,
        cls_loss_weight=10,
        project_loss_weight=1.0,
    ),
    trans_params=dict(init_z=0, bev_h=150, bev_w=70),
)

ms2one=dict(
    type='DilateNaive',
    inc=_dim_, outc=_dim_, num_scales=4,
    dilations=(1, 2, 5, 9))

transformer=dict(
    type='LATRTransformer',
    decoder=dict(
        type='LATRTransformerDecoder',
        embed_dims=_dim_,
        num_layers=6,
        enlarge_length=enlarge_length,
        M_decay_ratio=1,
        num_query=num_query,
        num_anchor_per_query=num_pt_per_line,
        anchor_y_steps=anchor_y_steps,
        transformerlayers=dict(
            type='LATRDecoderLayer',
            attn_cfgs=[
                dict(
                    type='MultiheadAttention',
                    embed_dims=_dim_,
                    num_heads=4,
                    dropout=0.1),
                dict(
                    type='MSDeformableAttention3D',
                    embed_dims=_dim_,
                    num_heads=4,
                    num_levels=1,
                    num_points=8,
                    batch_first=False,
                    num_query=num_query,
                    num_anchor_per_query=num_pt_per_line,
                    anchor_y_steps=anchor_y_steps,
                    dropout=0.1),
                ],
            ffn_cfgs=dict(
                type='FFN',
                embed_dims=_dim_,
                feedforward_channels=_dim_*8,
                num_fcs=2,
                ffn_drop=0.1,
                act_cfg=dict(type='ReLU', inplace=True),
            ),
            feedforward_channels=_dim_ * 8,
            operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                            'ffn', 'norm')),
))

sparse_ins_decoder=Config(
    dict(
        encoder=dict(
            out_dims=_dim_),
        decoder=dict(
            num_query=latr_cfg['num_query'],
            num_group=latr_cfg['num_group'],
            sparse_num_group=latr_cfg['sparse_num_group'],
            hidden_dim=_dim_,
            kernel_dim=_dim_,
            num_classes=num_category,
            num_convs=4,
            output_iam=True,
            scale_factor=1.,
            ce_weight=2.0,
            mask_weight=5.0,
            dice_weight=2.0,
            objectness_weight=1.0,
        ),
        sparse_decoder_weight=5.0,
))

nepochs = 20
resize_h = 720
resize_w = 960

eval_freq = 1
optimizer_cfg = dict(
    type='AdamW',
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            'sampling_offsets': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)