# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Mostly copy-paste from timm library.
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
import math
from functools import partial

import numpy as np
import paddle
import paddle.nn as nn
from paddle.nn.initializer import Constant
from paddle import ParamAttr


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + paddle.rand(shape, dtype=x.dtype)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Layer):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Layer):
    """Code referenced from https://aistudio.baidu.com/aistudio/projectdetail/5022985?forkThirdPart=1"""
    def __init__(self, in_features, hidden_features, out_features=None, act_layer=nn.GELU, drop=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        #fc1
        w_attr_1, b_attr_1 = self._init_weights()
        self.fc1 = nn.Linear(in_features, 
                            hidden_features, 
                            weight_attr = w_attr_1, 
                            bias_attr = b_attr_1)
        #fc2
        w_attr_2, b_attr_2 = self._init_weights()
        self.fc2 = nn.Linear(hidden_features,
                            out_features, 
                            weight_attr = w_attr_2, 
                            bias_attr = b_attr_2)

        self.act = act_layer() #GELU > ELU > ReLU > sigmod
        self.dropout1 = nn.Dropout(drop)
        self.dropout2 = nn.Dropout(drop)

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.XavierUniform())
            #XavierNormal正态分布的所有层梯度一致，XavierUniform均匀分布的所有成梯度一致。
        bias_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.Normal(std=1e-6)) #正态分布的权值和偏置
        return weight_attr, bias_attr

    def forward(self, x):
        x = self.fc1(x)         #[N, ~, embed_dim]
        x = self.act(x)
        x = self.dropout1(x)
        x = self.fc2(x)         #[N, ~, embed_dim]
        x = self.dropout2(x)
        return x


class Attention(nn.Layer):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.attn_head_size = int(dim / self.num_heads)
        self.all_head_size = self.attn_head_size * self.num_heads
        self.scales = qk_scale or self.attn_head_size ** -0.5

        #calculate qkv
        w_attr_1, b_attr_1 = self._init_weights()
        self.qkv = nn.Linear(
            dim, self.all_head_size * 3, # weight for Q K V
            weight_attr = w_attr_1,
            bias_attr = b_attr_1 if qkv_bias else False)

        #calculate proj
        w_attr_2, b_attr_2 = self._init_weights()
        self.proj = nn.Linear(dim, dim, weight_attr=w_attr_2, bias_attr=b_attr_2)

        self.attn_dropout = nn.Dropout(attn_drop)
        self.proj_dropout = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(axis=-1)

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(initializer=nn.initializer.KaimingUniform())
        bias_attr = paddle.ParamAttr(initializer=nn.initializer.KaimingUniform())
        return weight_attr, bias_attr

    def transpose_multihead(self, x):
        #input size  [N, ~, embed_dim]
        new_shape = x.shape[0:2] + [self.num_heads, self.attn_head_size]
        #reshape size[N, ~, head, head_size]
        x = x.reshape(new_shape)
        x = x.transpose([0, 2, 1, 3])
        #transpose   [N, head, ~, head_size]
        return x

    def forward(self, x):
        #input x = [N, H * W + 1, embed_dim]
        qkv = self.qkv(x).chunk(3, axis = -1)           #[N, ~, embed_dim * 3]  list
        q, k, v = map(self.transpose_multihead, qkv)    #[N, head, ~, head_size]
        
        attn = paddle.matmul(q, k, transpose_y = True)  #[N, head, ~, ~]
        attn = self.softmax(attn * self.scales)         #softmax(Q*K/(dk^0.5))
        attn = self.attn_dropout(attn)                  #[N, head, ~, ~]
        
        z = paddle.matmul(attn, v)                      #[N, head, ~, head_size]
        z = z.transpose([0, 2, 1, 3])                   #[N, ~, head, head_size]
        new_shape = z.shape[0:2] + [self.all_head_size]
        z = z.reshape(new_shape)                        #[N, ~, embed_dim]
        z = self.proj(z)                                #[N, ~, embed_dim]
        z = self.proj_dropout(z)                        #[N, ~, embed_dim]

        return z, attn


class Block(nn.Layer):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Layer):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2D(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose((0, 2, 1))
        return x


class VisionTransformer(nn.Layer):
    """ Vision Transformer """
    def __init__(self, img_size=[224], patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        p_attr = ParamAttr(initializer=Constant(0.0))
        self.cls_token = paddle.create_parameter(
            shape=[1, 1, embed_dim],
            dtype='float32',
            attr=p_attr,
            default_initializer=paddle.nn.initializer.TruncatedNormal(std=.02))

        self.pos_embed = paddle.create_parameter(
            shape=[1, 1 + num_patches, embed_dim],
            dtype='float32',
            attr=p_attr,
            default_initializer=paddle.nn.initializer.TruncatedNormal(std=.02))

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in paddle.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.LayerList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            m.weight = paddle.create_parameter(
                shape=m.weight.shape,
                dtype='float32',
                default_initializer=paddle.nn.initializer.TruncatedNormal(std=.02)
            )
            # trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.set_value(np.zeros(m.bias.shape, dtype="float32"))
                # nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            m.bias.set_value(np.zeros(m.bias.shape, dtype="float32"))
            m.weight.set_value(np.ones(m.weight.shape, dtype="float32"))
            # nn.init.constant_(m.bias, 0)
            # nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return paddle.concat((class_pos_embed.unsqueeze(0), patch_pos_embed), axis=1)

    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand((B, -1, -1))
        x = paddle.concat((cls_tokens, x), axis=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)

        return self.pos_drop(x)

    def forward(self, x):
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        # temp
        return x[:, 0]

    def get_last_selfattention(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)

    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output


def vit_small(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs)
    return model


class DINOHead(nn.Layer):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1D(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1D(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias_attr=False), dim=1)
        self.last_layer.weight_g.set_value(
            np.ones(self.last_layer.weight_g.shape, dtype="float32")
        )
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            m.weight = paddle.create_parameter(
                shape=m.weight.shape,
                dtype='float32',
                default_initializer=paddle.nn.initializer.TruncatedNormal(std=.02)
            )
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.set_value(np.zeros(m.bias.shape, dtype="float32"))

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, axis=-1, p=2)
        x = self.last_layer(x)
        return x
