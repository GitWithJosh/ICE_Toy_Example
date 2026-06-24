# Minimal, eigenstaendige Neuimplementierung der ICE-Modellklasse fuer das
# Toy Example dieser Seminararbeit.
#
# Architektur und Forward-Pass sind funktional identisch zu
# models_ICE.py aus dem offiziellen Repository von Choi et al. (ICE):
# https://github.com/Hanyang-HCC-Lab/ICE
# (imagenet_segmentation/baselines/ViT/models_ICE.py, MIT License),
# das wiederum auf der DeiT-Implementierung von Touvron et al. und der
# timm-Bibliothek aufbaut. Um das Toy Example ohne die zusaetzliche
# Abhaengigkeit von timm lauffaehig zu halten, werden die wenigen dort
# verwendeten Bausteine (Attention, MLP, Gewichtsinitialisierung) hier
# mit reinem PyTorch nachgebildet. Die fuer den Forward-Pass und das
# Laden des Original-Checkpoints relevante Struktur, insbesondere der
# zusaetzliche patchweise Klassifikationskopf mlp_head mit
# num_classes + 1 Ausgaben (1000 ImageNet-Klassen plus eine eigens
# eingefuehrte Hintergrundklasse), ist unveraendert uebernommen.

import math
from functools import partial

import torch
import torch.nn as nn


def trunc_normal_(tensor, std=0.02):
    return nn.init.trunc_normal_(tensor, std=std, a=-2 * std, b=2 * std)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    """ Bild zu Patch-Embedding, analog zu Dosovitskiy et al. """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer (DeiT-artig) mit zusaetzlichem ICE-Patchklassifikationskopf. """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dim=384, depth=12, num_heads=6, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=None, act_layer=None):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim) * .02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                  drop=drop_rate, attn_drop=attn_drop_rate, norm_layer=norm_layer, act_layer=act_layer)
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        ## ICE: patchweiser Klassifikationskopf mit zusaetzlicher Hintergrundklasse
        self.mlp_head = nn.Linear(self.num_features, num_classes + 1, bias=False)
        ## ICE

        self.fc_norm = nn.Identity()
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        nn.init.normal_(self.cls_token, std=1e-6)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        # Patchweise Klassifikation (ohne CLS-Token) ueber den ICE-Kopf
        att = x[:, 1:].clone()
        att = self.mlp_head(att)

        x = self.norm(x)
        return x, att

    def forward_head(self, x):
        x = x[:, 0]
        x = self.fc_norm(x)
        return self.head(x)

    def forward(self, x):
        x, patch_logits = self.forward_features(x)
        cls_logits = self.forward_head(x)
        return cls_logits, patch_logits, patch_logits


def deit_small_patch16_224(num_classes=1000, img_size=224):
    return VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=num_classes, img_size=img_size)
