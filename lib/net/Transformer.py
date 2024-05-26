# ------------------------------------------------------------------------------------
# Enhancing Transformers
# Copyright (c) 2022 Thuan H. Nguyen. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------------
# Modified from ViT-Pytorch (https://github.com/lucidrains/vit-pytorch)
# Copyright (c) 2020 Phil Wang. All Rights Reserved.
# ------------------------------------------------------------------------------------

import math
import numpy as np
from typing import Union, Tuple, List, Optional
from functools import partial
import pytorch_lightning as pl

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

def get_2d_sincos_pos_embed(embed_dim, grid_size):#获取余弦正弦位置嵌入
    """
    grid_size: int or (int, int) of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_size = (grid_size, grid_size) if type(grid_size) != tuple else grid_size
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)

    return pos_embed

#从网格坐标中生成二维的正弦余弦位置嵌入
def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

#生成一维的余弦和正弦列表
def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0#检查是否符合条件
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)#拼接
    return emb


def init_weights(m):#初始化权重
    if isinstance(m, nn.Linear):#判断是否是指定类
        # we use xavier_uniform following official JAX ViT:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        w = m.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))


class PreNorm(nn.Module):#它实现了一个预归一化（Pre-Normalization）的模块，用于在输入数据上应用层归一化后再应用其他操作
    def __init__(self, dim: int, fn: nn.Module) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):#实现了一个简单的前馈神经网络（Feedforward Neural Network），用于对输入数据进行非线性变换
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64) -> None:
        super().__init__()
        inner_dim = dim_head *  heads#注意力头的维度和注意力头
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Linear(inner_dim, dim) if project_out else nn.Identity()

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)




class CrossAttention(nn.Module):#交叉注意力
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64) -> None:
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.norm = nn.LayerNorm(dim)

        self.to_out = nn.Linear(inner_dim, dim) if project_out else nn.Identity()
        self.multi_head_attention=PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head))


    def forward(self, x: torch.FloatTensor, q_x:torch.FloatTensor) -> torch.FloatTensor:
        
        q_in = self.multi_head_attention(q_x)+q_x
        q_in = self.norm(q_in)

        q = rearrange(self.to_q(q_in),'b n (h d) -> b h n d', h = self.heads)       
        kv = self.to_kv(x).chunk(2, dim = -1)
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), kv)
        
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out),q_in


class Transformer(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, dim_head: int, mlp_dim: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([])
        for idx in range(depth):
            layer = nn.ModuleList([PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head)),
                                   PreNorm(dim, FeedForward(dim, mlp_dim))])
            self.layers.append(layer)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class CrossTransformer(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, dim_head: int, mlp_dim: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([])
        for idx in range(depth):
            layer = nn.ModuleList([CrossAttention(dim, heads=heads, dim_head=dim_head),
                                   PreNorm(dim, FeedForward(dim, mlp_dim))])
            self.layers.append(layer)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x: torch.FloatTensor, q_x:torch.FloatTensor) -> torch.FloatTensor:
        encoder_output=x
        for attn, ff in self.layers:
            x,q_in = attn(encoder_output, q_x)
            x = x + q_in
            x = ff(x) + x
            q_x=x

        return self.norm(q_x)

class ViTEncoder(nn.Module):
    def __init__(self, image_size: Union[Tuple[int, int], int], patch_size: Union[Tuple[int, int], int],
                 dim: int, depth: int, heads: int, mlp_dim: int, channels: int = 3, dim_head: int = 64, output_dim=768) -> None:
        super().__init__()
        image_height, image_width = image_size if isinstance(image_size, tuple) else (image_size, image_size)
        patch_height, patch_width = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        en_pos_embedding = get_2d_sincos_pos_embed(dim, (image_height // patch_height, image_width // patch_width))

        self.num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.patch_dim = channels * patch_height * patch_width
        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(channels, dim, kernel_size=patch_size, stride=patch_size),
            Rearrange('b c h w -> b (h w) c'),
        )
        self.en_pos_embedding = nn.Parameter(torch.from_numpy(en_pos_embedding).float().unsqueeze(0), requires_grad=False)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        # 新增卷积层，用于调整输出通道数
        self.output_conv = nn.Conv2d(dim, output_dim, 1)  # 1x1 卷积用于调整通道数

        self.apply(init_weights)

    def forward(self, img: torch.FloatTensor) -> torch.FloatTensor:
        x = self.to_patch_embedding(img)  # 将输入图像通过 to_patch_embedding 方法转换为嵌入序列
        x = x + self.en_pos_embedding
        x = self.transformer(x)
        x = x.permute(0, 2, 1).view(x.shape[0], -1, int(x.shape[1]**0.5), int(x.shape[1]**0.5))  # 转换回 [B, C, H, W] 格式
        x = self.output_conv(x)  # 调整通道数
        
        return x



class ViTDecoder(nn.Module):#ViT 解码器的作用是将经过编码器处理的图像嵌入序列转换回原始图像的尺寸，从而实现图像的重建。
    def __init__(self, image_size: Union[Tuple[int, int], int], patch_size: Union[Tuple[int, int], int],
                 dim: int, depth: int, heads: int, mlp_dim: int, channels: int = 32, dim_head: int = 64) -> None:
        super().__init__()
        image_height, image_width = image_size if isinstance(image_size, tuple) \
                                    else (image_size, image_size)
        patch_height, patch_width = patch_size if isinstance(patch_size, tuple) \
                                    else (patch_size, patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        de_pos_embedding = get_2d_sincos_pos_embed(dim, (image_height // patch_height, image_width // patch_width))

        self.num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.patch_dim = channels * patch_height * patch_width

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)#Transformer 层，用于对图像块的嵌入序列进行编码。
        self.de_pos_embedding = nn.Parameter(torch.from_numpy(de_pos_embedding).float().unsqueeze(0), requires_grad=False)
        self.to_pixel = nn.Sequential(
            Rearrange('b (h w) c -> b c h w', h=image_height // patch_height),
            nn.ConvTranspose2d(dim, channels, kernel_size=4, stride=4)
        )#编码器处理的图像嵌入序列转换回原始图像的尺寸。
        self.apply(init_weights)
    #将加了位置编码的嵌入序列输入到 Transformer 层中进行编码。
    def forward(self, token: torch.FloatTensor) -> torch.FloatTensor:
        x = token + self.de_pos_embedding
        x = self.transformer(x)
        x = self.to_pixel(x)#重建图像

        return x
    #返回的是解码器中最后一层转置卷积层的权重参数
    def get_last_layer(self) -> nn.Parameter:
        return self.to_pixel[-1].weight


class CrossAttDecoder(nn.Module):
    def __init__(self, image_size: Union[Tuple[int, int], int], patch_size: Union[Tuple[int, int], int],
                 dim: int, depth: int, heads: int, mlp_dim: int, channels: int = 32, dim_head: int = 64) -> None:
        super().__init__()
        image_height, image_width = image_size if isinstance(image_size, tuple) \
                                    else (image_size, image_size)
        patch_height, patch_width = patch_size if isinstance(patch_size, tuple) \
                                    else (patch_size, patch_size)


        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
            Rearrange('b c h w -> b (h w) c'),
        )

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        de_pos_embedding = get_2d_sincos_pos_embed(dim, (image_height // patch_height, image_width // patch_width))

        self.num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.patch_dim = channels * patch_height * patch_width

        self.transformer = CrossTransformer(dim, depth, heads, dim_head, mlp_dim)
        
        self.de_pos_embedding = nn.Parameter(torch.from_numpy(de_pos_embedding).float().unsqueeze(0), requires_grad=False)
        self.to_pixel = nn.Sequential(
            Rearrange('b (h w) c -> b c h w', h=image_height // patch_height),
            nn.ConvTranspose2d(dim, channels, kernel_size=4, stride=4)
        )

        self.apply(init_weights)

    def forward(self, token: torch.FloatTensor, query_img:torch.FloatTensor) -> torch.FloatTensor:
        # batch_size=token.shape[0]
        # query=self.query.repeat(batch_size,1,1)+self.de_pos_embedding
        query=self.to_patch_embedding(query_img)+self.de_pos_embedding
        x = token + self.de_pos_embedding
        x = self.transformer(x,query)
        x = self.to_pixel(x)

        return x

    def get_last_layer(self) -> nn.Parameter:
        return self.to_pixel[-1].weight


class BaseQuantizer(nn.Module):
    def __init__(self, embed_dim: int, n_embed: int, straight_through: bool = True, use_norm: bool = True,
                 use_residual: bool = False, num_quantizers: Optional[int] = None) -> None:
        super().__init__()
        self.straight_through = straight_through
        self.norm = lambda x: F.normalize(x, dim=-1) if use_norm else x

        self.use_residual = use_residual
        self.num_quantizers = num_quantizers

        self.embed_dim = embed_dim
        self.n_embed = n_embed

        self.embedding = nn.Embedding(self.n_embed, self.embed_dim)
        self.embedding.weight.data.normal_()
        
    def quantize(self, z: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.LongTensor]:
        pass
    
    def forward(self, z: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.LongTensor]:
        if not self.use_residual:
            z_q, loss, encoding_indices = self.quantize(z)
        else:
            z_q = torch.zeros_like(z)
            residual = z.detach().clone()

            losses = []
            encoding_indices = []

            for _ in range(self.num_quantizers):
                z_qi, loss, indices = self.quantize(residual.clone())
                residual.sub_(z_qi)
                z_q.add_(z_qi)

                encoding_indices.append(indices)
                losses.append(loss)

            losses, encoding_indices = map(partial(torch.stack, dim = -1), (losses, encoding_indices))
            loss = losses.mean()

        # preserve gradients with straight-through estimator
        if self.straight_through:
            z_q = z + (z_q - z).detach()

        return z_q, loss, encoding_indices


class VectorQuantizer(BaseQuantizer):
    def __init__(self, embed_dim: int, n_embed: int, beta: float = 0.25, use_norm: bool = True,
                 use_residual: bool = False, num_quantizers: Optional[int] = None, **kwargs) -> None:
        super().__init__(embed_dim, n_embed, True,
                         use_norm, use_residual, num_quantizers)
        
        self.beta = beta

    def quantize(self, z: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.LongTensor]:
        z_reshaped_norm = self.norm(z.view(-1, self.embed_dim))
        embedding_norm = self.norm(self.embedding.weight)
        
        d = torch.sum(z_reshaped_norm ** 2, dim=1, keepdim=True) + \
            torch.sum(embedding_norm ** 2, dim=1) - 2 * \
            torch.einsum('b d, n d -> b n', z_reshaped_norm, embedding_norm)

        encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        encoding_indices = encoding_indices.view(*z.shape[:-1])
        
        z_q = self.embedding(encoding_indices).view(z.shape)
        z_qnorm, z_norm = self.norm(z_q), self.norm(z)
        
        # compute loss for embedding
        loss = self.beta * torch.mean((z_qnorm.detach() - z_norm)**2) +  \
               torch.mean((z_qnorm - z_norm.detach())**2)

        return z_qnorm, loss, encoding_indices


class ViTVQ(pl.LightningModule):
    def __init__(self,image_size=512, patch_size=16,channels=3,face_image_size=512) -> None:
        super().__init__()
        
        self.encoder = ViTEncoder(image_size=image_size, patch_size=patch_size, dim=256,depth=8,heads=8,mlp_dim=2048,channels=channels)
        self.facerio = FaceRoI(feat_dim=768,upscale=4)
        self.F_decoder = ViTDecoder(image_size=image_size, patch_size=patch_size, dim=256,depth=3,heads=8,mlp_dim=2048)
        self.B_decoder= CrossAttDecoder(image_size=image_size, patch_size=patch_size, dim=256,depth=3,heads=8,mlp_dim=2048)
        self.R_decoder= CrossAttDecoder(image_size=image_size, patch_size=patch_size, dim=256,depth=3,heads=8,mlp_dim=2048)
        self.L_decoder= CrossAttDecoder(image_size=image_size, patch_size=patch_size, dim=256,depth=3,heads=8,mlp_dim=2048)
        # self.quantizer = VectorQuantizer(embed_dim=32,n_embed=8192)
        # self.pre_quant = nn.Linear(512, 32)
        # self.post_quant = nn.Linear(32, 512)


    def forward(self,bbox, x: torch.FloatTensor,smpl_normal) -> torch.FloatTensor:    
        enc_out,img_feat = self.encode(x)
        y=self.facerio(img_feat,bbox)
        enc_out+=y

        dec = self.decode(enc_out,smpl_normal)
        
        return dec

    def encode(self, x: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        img_feat = self.encoder(x)
        return img_feat, img_feat  # 返回相同的特征作为两个不同的输出，如果确实需要的话。


    def facerio1(self,img_feat, face_bbox):
        face_feat=self.facerio(img_feat,face_bbox)
        return face_feat

        


    def decode(self, enc_out: torch.FloatTensor,smpl_normal) -> torch.FloatTensor:
        back_query=smpl_normal['T_normal_B']
        right_query=smpl_normal['T_normal_R']
        left_query=smpl_normal['T_normal_L']
        # quant = self.post_quant(quant)
        dec_F = self.F_decoder(enc_out)
        dec_B = self.B_decoder(enc_out,back_query)
        dec_R = self.R_decoder(enc_out,right_query)
        dec_L = self.L_decoder(enc_out,left_query)
        
        return (dec_F,dec_B,dec_R,dec_L)

    # def encode_codes(self, x: torch.FloatTensor) -> torch.LongTensor:
    #     h = self.encoder(x)
    #     h = self.pre_quant(h)
    #     _, _, codes = self.quantizer(h)
        
    #     return codes

    # def decode_codes(self, code: torch.LongTensor) -> torch.FloatTensor:
    #     quant = self.quantizer.embedding(code)
    #     quant = self.quantizer.norm(quant)
        
    #     if self.quantizer.use_residual:
    #         quant = quant.sum(-2)  
            
    #     dec = self.decode(quant)
        
    #     return dec


from kornia.geometry.transform import get_affine_matrix2d, warp_affine
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import mediapipe as mp
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from lib.pymafx.core import constants
from main.config import cfg
from common.nets.module_wo_decoder import PositionNet, HandRotationNet, FaceRegressor, BoxNet, HandRoI, BodyRotationNet
from rembg import remove
from common.nets.module_wo_decoder import PositionNet, HandRotationNet, FaceRegressor, BoxNet, HandRoI, BodyRotationNet
# from rembg.session_factory import new_session
from torchvision import transforms

from common.nets.layer import make_conv_layers, make_linear_layers, make_deconv_layers
from common.utils.transforms import sample_joint_features, soft_argmax_2d, soft_argmax_3d

#可以裁剪出脸部边框
def transform_to_tensor(res, mean=None, std=None, is_tensor=False):
    all_ops = []
    if res is not None:
        all_ops.append(transforms.Resize(size=res))
    if not is_tensor:
        all_ops.append(transforms.ToTensor())
    if mean is not None and std is not None:
        all_ops.append(transforms.Normalize(mean=mean, std=std))
    return transforms.Compose(all_ops)


def get_affine_matrix_wh(w1, h1, w2, h2):
    "根据输入和输出图像的宽度和高度，生成一个仿射变换矩阵，用于将一个矩形区域从一个尺寸调整到另一个尺寸。"

    transl = torch.tensor([(w2 - w1) / 2.0, (h2 - h1) / 2.0]).unsqueeze(0)
    center = torch.tensor([w1 / 2.0, h1 / 2.0]).unsqueeze(0)
    scale = torch.min(torch.tensor([w2 / w1, h2 / h1])).repeat(2).unsqueeze(0)
    M = get_affine_matrix2d(transl, center, scale, angle=torch.tensor([0.]))

    return M


def get_affine_matrix_box(boxes, w2, h2):

    # boxes [left, top, right, bottom]
    width = boxes[:, 2] - boxes[:, 0]    #(N,)
    height = boxes[:, 3] - boxes[:, 1]    #(N,)
    center = torch.tensor(
        [(boxes[:, 0] + boxes[:, 2]) / 2.0, (boxes[:, 1] + boxes[:, 3]) / 2.0]
    ).T    #(N,2)
    scale = torch.min(torch.tensor([w2 / width, h2 / height]),
                      dim=0)[0].unsqueeze(1).repeat(1, 2) * 0.9    #(N,2)
    transl = torch.cat([w2 / 2.0 - center[:, 0:1], h2 / 2.0 - center[:, 1:2]], dim=1)   #(N,2)
    M = get_affine_matrix2d(transl, center, scale, angle=torch.tensor([0.,]*transl.shape[0]))

    return M


def load_img(img_file):

    if img_file.endswith("exr"):
        img = cv2.imread(img_file, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    else :
        img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)

    # considering non 8-bit image
    if img.dtype != np.uint8 :
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if not img_file.endswith("png"):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

    return torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float(), img.shape[:2]

#从图像中提取关键点（keypoints），包括人体姿势、左手、右手和面部关键点。
def get_keypoints(image):
    def collect_xyv(x, body=True):
        lmk = x.landmark
        all_lmks = []
        for i in range(len(lmk)):
            visibility = lmk[i].visibility if body else 1.0
            all_lmks.append(torch.Tensor([lmk[i].x, lmk[i].y, lmk[i].z, visibility]))
        return torch.stack(all_lmks).view(-1, 4)

    mp_holistic = mp.solutions.holistic

    with mp_holistic.Holistic(
        static_image_mode=True,
        model_complexity=2,
    ) as holistic:
        results = holistic.process(image)

    fake_kps = torch.zeros(33, 4)

    result = {}
    result["body"] = collect_xyv(results.pose_landmarks) if results.pose_landmarks else fake_kps
    result["lhand"] = collect_xyv(
        results.left_hand_landmarks, False
    ) if results.left_hand_landmarks else fake_kps
    result["rhand"] = collect_xyv(
        results.right_hand_landmarks, False
    ) if results.right_hand_landmarks else fake_kps
    result["face"] = collect_xyv(
        results.face_landmarks, False
    ) if results.face_landmarks else fake_kps

    return result

#根据检测到的关键点（landmarks）对图像进行裁剪和变换，以准备用于后续的分割任务
def get_pymafx(image, landmarks):

    # image [3,512,512]

    item = {
        'img_body':
            F.interpolate(image.unsqueeze(0), size=224, mode='bicubic', align_corners=True)[0]
    }

    for part in ['lhand', 'rhand', 'face']:
        kp2d = landmarks[part]
        kp2d_valid = kp2d[kp2d[:, 3] > 0.]
        if len(kp2d_valid) > 0:
            bbox = [
                min(kp2d_valid[:, 0]),
                min(kp2d_valid[:, 1]),
                max(kp2d_valid[:, 0]),
                max(kp2d_valid[:, 1])
            ]
            center_part = [(bbox[2] + bbox[0]) / 2., (bbox[3] + bbox[1]) / 2.]
            scale_part = 2. * max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2

        # handle invalid part keypoints
        if len(kp2d_valid) < 1 or scale_part < 0.01:
            center_part = [0, 0]
            scale_part = 0.5
            kp2d[:, 3] = 0

        center_part = torch.tensor(center_part).float()

        theta_part = torch.zeros(1, 2, 3)
        theta_part[:, 0, 0] = scale_part
        theta_part[:, 1, 1] = scale_part
        theta_part[:, :, -1] = center_part

        grid = F.affine_grid(theta_part, torch.Size([1, 3, 224, 224]), align_corners=False)
        img_part = F.grid_sample(image.unsqueeze(0), grid, align_corners=False).squeeze(0).float()

        item[f'img_{part}'] = img_part

        theta_i_inv = torch.zeros_like(theta_part)
        theta_i_inv[:, 0, 0] = 1. / theta_part[:, 0, 0]
        theta_i_inv[:, 1, 1] = 1. / theta_part[:, 1, 1]
        theta_i_inv[:, :, -1] = -theta_part[:, :, -1] / theta_part[:, 0, 0].unsqueeze(-1)
        item[f'{part}_theta_inv'] = theta_i_inv[0]

    return item


def remove_floats(mask):

    # 1. find all the contours
    # 2. fillPoly "True" for the largest one
    # 3. fillPoly "False" for its childrens

    new_mask = np.zeros(mask.shape)
    cnts, hier = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnt_index = sorted(range(len(cnts)), key=lambda k: cv2.contourArea(cnts[k]), reverse=True)
    body_cnt = cnts[cnt_index[0]]
    childs_cnt_idx = np.where(np.array(hier)[0, :, -1] == cnt_index[0])[0]
    childs_cnt = [cnts[idx] for idx in childs_cnt_idx]
    cv2.fillPoly(new_mask, [body_cnt], 1)
    cv2.fillPoly(new_mask, childs_cnt, 0)

    return new_mask


def econ_process_image(img_file, hps_type, single, input_res,image, detector):

    img_raw, (in_height, in_width) = load_img(img_file)
    tgt_res = input_res * 2
    M_square = get_affine_matrix_wh(in_width, in_height, tgt_res, tgt_res)
    img_square = warp_affine(
        img_raw,
        M_square[:, :2], (tgt_res, ) * 2,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    )
    #使用的decoder对人体进行检测，得到人体的边框和掩码，
    # detection for bbox
    predictions = detector(img_square / 255.)[0]

    if single:
        top_score = predictions["scores"][predictions["labels"] == 1].max()
        human_ids = torch.where(predictions["scores"] == top_score)[0]
    else:
        human_ids = torch.logical_and(predictions["labels"] == 1,
                                      predictions["scores"] > 0.9).nonzero().squeeze(1)

    boxes = predictions["boxes"][human_ids, :].detach().cpu().numpy()
    masks = predictions["masks"][human_ids, :, :].permute(0, 2, 3, 1).detach().cpu().numpy()
        #根据边框值计算仿射变换矩阵，并用该矩阵对原始图像进行裁剪，得到人体部分的图像
    M_crop = get_affine_matrix_box(boxes, input_res, input_res)

    img_icon_lst = []
    img_crop_lst = []
    img_hps_lst = []
    img_mask_lst = []
    landmark_lst = []
    hands_visibility_lst = []
    img_pymafx_lst = []

    uncrop_param = {
        "ori_shape": [in_height, in_width],
        "box_shape": [input_res, input_res],
        "square_shape": [tgt_res, tgt_res],
        "M_square": M_square,
        "M_crop": M_crop
    }

    for idx in range(len(boxes)):


        # mask out the pixels of others
        if len(masks) > 1:
            mask_detection = (masks[np.arange(len(masks)) != idx]).max(axis=0)
        else:
            mask_detection = masks[0] * 0.

        img_square_rgba = torch.cat(
            [img_square.squeeze(0).permute(1, 2, 0),
             torch.tensor(mask_detection < 0.4) * 255],
            dim=2
        )

        img_crop = warp_affine(
            img_square_rgba.unsqueeze(0).permute(0, 3, 1, 2),
            M_crop[idx:idx + 1, :2], (input_res, ) * 2,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        ).squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)
        #将背景移除，得到准确的mask
        # get accurate person segmentation mask
        img_rembg = remove(img_crop) #post_process_mask=True)
        img_mask = remove_floats(img_rembg[:, :, [3]])

        mean_icon = std_icon = (0.5, 0.5, 0.5)
        img_np = (img_rembg[..., :3] * img_mask).astype(np.uint8)
        img_icon = transform_to_tensor(512, mean_icon, std_icon)(
            Image.fromarray(img_np)
        ) * torch.tensor(img_mask).permute(2, 0, 1)
        img_hps = transform_to_tensor(224, constants.IMG_NORM_MEAN,
                                      constants.IMG_NORM_STD)(Image.fromarray(img_np))

        landmarks = get_keypoints(img_np)

        # get hands visibility判断手的可见性
        hands_visibility = [True, True]
        if landmarks['lhand'][:, -1].mean() == 0.:
            hands_visibility[0] = False
        if landmarks['rhand'][:, -1].mean() == 0.:
            hands_visibility[1] = False
        hands_visibility_lst.append(hands_visibility)

        if hps_type == 'pymafx':#hps_type是处理关键点的类型
            img_pymafx_lst.append(
                get_pymafx(
                    transform_to_tensor(512, constants.IMG_NORM_MEAN,
                                        constants.IMG_NORM_STD)(Image.fromarray(img_np)), landmarks
                )
            )

        img_crop_lst.append(torch.tensor(img_crop).permute(2, 0, 1) / 255.0)
        img_icon_lst.append(img_icon)
        img_hps_lst.append(img_hps)
        img_mask_lst.append(torch.tensor(img_mask[..., 0]))
        landmark_lst.append(landmarks['body'])
        face_landmarks = landmarks["face"]

        # 计算脸部边界框
        x_min = face_landmarks[:, 0].min()
        y_min = face_landmarks[:, 1].min()
        x_max = face_landmarks[:, 0].max()
        y_max = face_landmarks[:, 1].max()
        face_bbox = [x_min, y_min, x_max, y_max]


        face_img = image[:, :, face_bbox[1]:face_bbox[3], face_bbox[0]:face_bbox[2]]
        resized_img = cv2.resize(face_img, (512,512), interpolation=cv2.INTER_LINEAR)
        height, width, _ = face_img.shape
        face_img_size = (width, height)




    return face_img,face_img_size
from mmcv.ops.roi_align import roi_align
class FaceRoI(nn.Module):
    def __init__(self, feat_dim=768, upscale=4):
        super(FaceRoI, self).__init__()
        self.deconv = nn.ModuleList([])
        for i in range(int(math.log2(upscale))+1):
            if i==0:
                self.deconv.append(make_conv_layers([feat_dim, feat_dim], kernel=1, stride=1, padding=0, bnrelu_final=False))
            elif i==1:
                self.deconv.append(make_deconv_layers([feat_dim, feat_dim//2]))
            elif i==2:
                self.deconv.append(make_deconv_layers([feat_dim, feat_dim//2, feat_dim//4]))
            elif i==3:
                self.deconv.append(make_deconv_layers([feat_dim, feat_dim//2, feat_dim//4, feat_dim//8]))

    def forward(self, img_feat, face_bbox):
        face_bbox = torch.cat((torch.arange(face_bbox.shape[0]).float().cuda()[:, None], face_bbox),
                               1)  # batch_idx, xmin, ymin, xmax, ymax
        face_img_feats = []
        for i, deconv in enumerate(self.deconv):
            scale = 2**i
            img_feat_i = deconv(img_feat)
            face_bbox_roi = face_bbox.clone()
            face_bbox_roi[:, 1] = face_bbox_roi[:, 1] / cfg.input_body_shape[1] * cfg.output_hm_shape[2] * scale
            face_bbox_roi[:, 2] = face_bbox_roi[:, 2] / cfg.input_body_shape[0] * cfg.output_hm_shape[1] * scale
            face_bbox_roi[:, 3] = face_bbox_roi[:, 3] / cfg.input_body_shape[1] * cfg.output_hm_shape[2] * scale
            face_bbox_roi[:, 4] = face_bbox_roi[:, 4] / cfg.input_body_shape[0] * cfg.output_hm_shape[1] * scale
            assert (cfg.output_hm_shape[1]*scale, cfg.output_hm_shape[2]*scale) == (img_feat_i.shape[2], img_feat_i.shape[3])
            face_img_feat = roi_align(img_feat_i, face_bbox_roi,
                                                       (cfg.output_face_hm_shape[1]*scale//2,
                                                        cfg.output_face_hm_shape[2]*scale//2),
                                                       1.0, 0, 'avg', False)
            face_img_feats.append(face_img_feat)
        return face_img_feats[::-1]   # high resolution -> low resolution


def blend_rgb_norm(norms, data):

    # norms [N, 3, res, res]
    masks = (norms.sum(dim=1) != norms[0, :, 0, 0].sum()).float().unsqueeze(1)
    norm_mask = F.interpolate(
        torch.cat([norms, masks], dim=1).detach(),
        size=data["uncrop_param"]["box_shape"],
        mode="bilinear",
        align_corners=False
    )
    final = data["img_raw"].type_as(norm_mask)

    for idx in range(len(norms)):

        norm_pred = (norm_mask[idx:idx + 1, :3, :, :] + 1.0) * 255.0 / 2.0
        mask_pred = norm_mask[idx:idx + 1, 3:4, :, :].repeat(1, 3, 1, 1)

        norm_ori = unwrap(norm_pred, data["uncrop_param"], idx)
        mask_ori = unwrap(mask_pred, data["uncrop_param"], idx)

        final = final * (1.0 - mask_ori) + norm_ori * mask_ori

    return final.detach().cpu()

#将经过裁剪的图像恢复到原始尺寸
def unwrap(image, uncrop_param, idx):

    device = image.device

    img_square = warp_affine(
        image,
        torch.inverse(uncrop_param["M_crop"])[idx:idx + 1, :2].to(device),
        uncrop_param["square_shape"],
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    )

    img_ori = warp_affine(
        img_square,
        torch.inverse(uncrop_param["M_square"])[:, :2].to(device),
        uncrop_param["ori_shape"],
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    )
    from main.config import cfg
    import math
    from mmcv.ops.roi_align import roi_align
    return img_ori



