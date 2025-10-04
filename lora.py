import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
from safetensors import safe_open
from safetensors.torch import save_file
import yaml
class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]
class LoRA_qkv(nn.Module):

    def __init__(
            self,
            qkv,
            linear_a_q: nn.Module,
            linear_b_q: nn.Module,
            linear_a_v: nn.Module,
            linear_b_v: nn.Module,
            rank,
            alpha,
            dropout_rate
    ):
        
        super(LoRA_qkv, self).__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.d_model = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)
        self.dim = qkv.in_features
        # self.conv = GhostModule(16, self.dim)
        self.rank = rank
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout_rate)
    def forward(self, x: Tensor):
        qkv = self.qkv(x)


        q_ba = (self.alpha / self.rank) * self.linear_b_q(self.dropout(self.linear_a_q(x)))
        v_ba = (self.alpha / self.rank) * self.linear_b_v(self.dropout(self.linear_a_v(x)))


        qkv[:, :, :, :self.d_model] += q_ba #q part
        qkv[:, :, :, -self.d_model:] += v_ba #v part

        return qkv


class LoRA_linear0(nn.Module):

    def __init__(
            self,
            linear0,
            w_a_linear_linear1: nn.Module,
            w_b_linear_linear1: nn.Module,
            rank,
            alpha,
            dropout_rate

    ):
        super(LoRA_linear0, self).__init__()
        self.mlp_linear0 = linear0
        self.linear_a_q = w_a_linear_linear1
        self.linear_b_q = w_b_linear_linear1
        self.d_model = linear0.in_features
        self.w_identity = torch.eye(linear0.in_features)
        self.dim = linear0.in_features
        # self.conv = GhostModule(16, self.dim)
        self.rank = rank
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: Tensor):
        mlp_linear0 = self.mlp_linear0(x)

        q_ba = (self.alpha / self.rank) * self.linear_b_q(self.dropout(self.linear_a_q(x)))

        mlp_linear0[:, :, :, :self.d_model] += q_ba  # q part


        return mlp_linear0
# class LoRA_linear1(nn.Module):
#
#     def __init__(
#             self,
#             linear1,
#             w_a_linear_linear2: nn.Module,
#             w_b_linear_linear2: nn.Module,
#
#     ):
#         super(LoRA_linear1, self).__init__()
#         self.mlp_linear0 = linear1
#         self.linear_a_q = w_a_linear_linear2
#         self.linear_b_q = w_b_linear_linear2
#         self.w_identity = torch.eye(linear1.in_features)
#         self.dim = linear1.in_features
#         # self.conv = GhostModule(16, self.dim)
#
#     def forward(self, x: Tensor):
#         mlp_linear0 = self.mlp_linear0(x)
#
#         q_ba = self.linear_b_q(self.linear_a_q(x))
#
#         mlp_linear0[:, :, :, :self.dim] += q_ba  # q part
#
#
#         return mlp_linear0
class LoRA_sam(nn.Module):
    """
    Class that takes the image encoder of SAM and add the lora weights to the attentions blocks

    Arguments:
        sam_model: Sam class of the segment anything model
        rank: Rank of the matrix for LoRA
        lora_layer: List of weights exisitng for LoRA
    
    Return:
        None

    """

    def __init__(self, sam_model, rank: int, lora_layer=None):
        super(LoRA_sam, self).__init__()
        self.rank = rank
        self.alpha = 32
        self.dropout_rate = 0.1
        assert rank > 0
        # base_vit_dim = sam_model.image_encoder.patch_embed.proj.out_channels

        if lora_layer:
            self.lora_layer = lora_layer
        else:
            # In each block, you have an attention block => total blocks -> nb lora layers
            self.lora_layer = list(range(len(sam_model.encoder.blocks)))
        
        self.A_weights = []
        self.B_weights = []

        self.A_weights_linear = []
        self.B_weights_linear = []
        # freeze parameters of the image encoder
        for param in sam_model.encoder.parameters():
            param.requires_grad = False

        for t_layer_i, blk in enumerate(sam_model.encoder.blocks):
            # if only lora on few layers
            if t_layer_i % 2 == 0:
                if t_layer_i not in self.lora_layer:
                    continue

                w_qkv_linear = blk.attn.qkv

                mlp_linear0 = blk.mlp.layers[0]
                # mlp_linear1 = blk.mlp.layers[1]
                self.d_linear0 = mlp_linear0.in_features
                # self.d_linear1 = mlp_linear1.in_features

                self.d_model = w_qkv_linear.in_features

                w_a_linear_q = nn.Linear(self.d_model, self.rank, bias=False)
                w_b_linear_q = nn.Linear(self.rank, self.d_model, bias=False)
                w_a_linear_v = nn.Linear(self.d_model, self.rank, bias=False)
                w_b_linear_v = nn.Linear(self.rank, self.d_model, bias=False)

                w_a_linear_linear1 = nn.Linear(self.d_linear0, self.rank, bias=False)
                w_b_linear_linear1 = nn.Linear(self.rank, self.d_linear0, bias=False)
                # w_a_linear_linear2 = nn.Linear(self.d_linear1, self.rank, bias=False)
                # w_b_linear_linear2 = nn.Linear(self.rank, self.d_linear1, bias=False)


                self.A_weights.append(w_a_linear_q)
                self.B_weights.append(w_b_linear_q)
                self.A_weights.append(w_a_linear_v)
                self.B_weights.append(w_b_linear_v)

                self.A_weights.append(w_a_linear_linear1)
                self.B_weights.append(w_b_linear_linear1)

                # self.A_weights.append(w_a_linear_linear2)
                # self.B_weights.append(w_b_linear_linear2)



                blk.attn.qkv = LoRA_qkv(
                    w_qkv_linear,
                    w_a_linear_q,
                    w_b_linear_q,
                    w_a_linear_v,
                    w_b_linear_v,
                    self.rank,
                    self.alpha,
                    self.dropout_rate
                )

                blk.mlp.layers[0] = LoRA_linear0(
                    mlp_linear0,
                    w_a_linear_linear1,
                    w_b_linear_linear1,
                    self.rank,
                    self.alpha,
                    self.dropout_rate
                )
                # blk.mlp.layers[1] = LoRA_linear1(
                #     mlp_linear1,
                #     w_a_linear_linear2,
                #     w_b_linear_linear2,
                # )


        self.reset_parameters()
        self.sam = sam_model
        self.lora_vit = sam_model.encoder


    def reset_parameters(self):
        """
        Initialize the LoRA A and B matrices like in the paper
        """
        # Initalisation like in the paper
        for w_A in self.A_weights:
            nn.init.kaiming_uniform_(w_A.weight, a=np.sqrt(5))
        for w_B in self.B_weights:
            nn.init.zeros_(w_B.weight)

        for w_A in self.A_weights_linear:
            nn.init.kaiming_uniform_(w_A.weight, a=np.sqrt(5))
        for w_B in self.B_weights_linear:
            nn.init.zeros_(w_B.weight)


    def save_lora_parameters(self, filename: str):
        """
        Save the LoRA wieghts applied to the attention model as safetensors.

        Arguments:
            filenmame: Name of the file that will be saved
        
        Return:
            None: Saves a safetensors file
        """
        num_layer = len(self.A_weights)
        # sufix 03:d -> allows to have a name 1 instead of 001
        a_tensors = {f"w_a_{i:03d}": self.A_weights[i].weight for i in range(num_layer)}
        b_tensors = {f"w_b_{i:03d}": self.B_weights[i].weight for i in range(num_layer)}
        merged_dict = {**a_tensors, **b_tensors}
        save_file(merged_dict, filename)


    def load_lora_parameters(self, filename: str):
        """
        Load a safetensor file of LoRA weights for the attention modules

        Arguments:
            filename: Name of the file containing the saved weights
        
        Return:
            None: Loads the weights to the LoRA_sam class
        """
        with safe_open(filename, framework="pt") as f:
            for i, w_A_linear in enumerate(self.A_weights):
                saved_key = f"w_a_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                w_A_linear.weight = nn.Parameter(saved_tensor)

            for i, w_B_linear in enumerate(self.B_weights):
                saved_key = f"w_b_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                w_B_linear.weight = nn.Parameter(saved_tensor)

