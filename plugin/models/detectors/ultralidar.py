# Copyright (c) Phigent Robotics. All rights reserved.
import torch
import torch.nn.functional as F
from mmcv.runner import force_fp32
import math
from torch.distributions import Categorical
from pyquaternion import Quaternion
import torch.distributed as dist
# DISTRIBUTIONS
from mmdet.models import DETECTORS
import mmcv
import sys
# from .. import builder
from einops import rearrange
from mmdet3d.models.detectors.centerpoint import CenterPoint
import torch.nn as nn
from mmcv.cnn import build_plugin_layer
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmdet3d.models.builder import build_voxel_encoder
import numpy as np
from plyfile import PlyData, PlyElement
from collections import Counter
import cv2
import matplotlib.pyplot as plt
import os
import kornia
from plugin.datasets.evaluation.jsd_mmd import point_cloud_to_histogram
import open3d as o3d


def dump_ply(save_path, points):
    # Create an Open3D point cloud object
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)

    # Save the point cloud to a PLY file
    o3d.io.write_point_cloud(save_path, point_cloud)
    

def count(idx):
    
    unique_elements, counts = torch.unique(idx, return_counts=True)
    sorted_indices = torch.argsort(counts, descending=True)
    sorted_elements = unique_elements[sorted_indices]
    sorted_counts = counts[sorted_indices]
    return sorted_elements, sorted_counts


def _sample_logistic(shape, out=None):
    U = out.resize_(shape).uniform_() if out is not None else torch.rand(shape)
    return torch.log(U) - torch.log(1 - U)


def _sigmoid_sample(logits, tau=1):
    """
    Implementation of Bernouilli reparametrization based on Maddison et al. 2017
    """
    dims = logits.dim()
    logistic_noise = _sample_logistic(logits.size(), out=logits.data.new())
    y = logits + logistic_noise
    return torch.sigmoid(y / tau)


def gumbel_sigmoid(logits, tau=1, hard=False):
    gumbel_sigmoid_coeff = 1.0
    y_soft = _sigmoid_sample(logits * gumbel_sigmoid_coeff, tau=tau)
    if hard:
        y_hard = torch.where(y_soft > 0.5, torch.ones_like(y_soft), torch.zeros_like(y_soft))
        y = y_hard.data - y_soft.data + y_soft
    else:
        y = y_soft
    return y


def coords2incides(coord, min_val, max_val, size):
    incides = (coord - min_val) / (max_val - min_val) * size
    return incides.int()


def gamma_func(self, mode="cosine"):
    if mode == "linear":
        return lambda r: 1 - r
    elif mode == "cosine":
        return lambda r: np.cos(r * np.pi / 2)
    elif mode == "square":
        return lambda r: 1 - r**2
    elif mode == "cubic":
        return lambda r: 1 - r**3
    else:
        raise NotImplementedError
    
def draw_bev_lidar(voxels, pth):

    cv2.imwrite(
        pth,
        voxels[0].max(dim=0)[0][:, :, None].repeat(1, 1, 3).detach().cpu().numpy() * 255,
    )
    


@DETECTORS.register_module()
class UltraLiDAR(CenterPoint):
    def __init__(
        self,
        model_type=None,
        vis_flag=True,
        dataset_type=None,
        num_classes=None,
        voxelizer=None,
        vector_quantizer=None,
        maskgit_transformer=None,
        lidar_encoder=None,
        lidar_decoder=None,
        bev_encoder=None,
        bev_decoder=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_type = model_type
        self.num_classes = num_classes
        self.voxelizer = build_voxel_encoder(voxelizer)
        self.vector_quantizer = build_plugin_layer(vector_quantizer)[1]
        self.pre_quant = nn.Sequential(nn.Linear(1024, 1024), nn.LayerNorm(1024))
        self.lidar_encoder = build_transformer_layer_sequence(lidar_encoder)
        self.lidar_decoder = build_transformer_layer_sequence(lidar_decoder)

        if self.model_type != "codebook_training":
            for p in self.parameters():
                p.requires_grad = False
            self.maskgit_transformer = build_transformer_layer_sequence(maskgit_transformer)
            

        del self.pts_bbox_head
        self.iter = 0
        self.T = 30
        
        
        self.counter = Counter()
        self.register_buffer("code_age", torch.zeros(self.vector_quantizer.n_e) * 10000)
        self.register_buffer("code_usage", torch.zeros(self.vector_quantizer.n_e))
        self.gamma = gamma_func("cosine")

        self.aug = nn.Sequential(
            kornia.augmentation.RandomVerticalFlip(),
            kornia.augmentation.RandomHorizontalFlip(),
        )
        self.vis_flag = vis_flag
        self.code_dict = {}
        for i in range(self.vector_quantizer.n_e):
            self.code_dict[i] = 0

    def train_codebook(self, points):
        
        voxels = self.voxelizer([[_] for _ in points])
        voxels = self.aug(voxels)

        lidar_feats = self.lidar_encoder(voxels)
        feats = self.pre_quant(lidar_feats)
        lidar_quant, emb_loss, _ = self.vector_quantizer(feats, self.code_age, self.code_usage)

        lidar_rec = self.lidar_decoder(lidar_quant)
        lidar_rec_loss = (F.binary_cross_entropy_with_logits(lidar_rec, voxels, reduction="none") * 100).mean()

        self.iter += 1
        lidar_rec_prob = lidar_rec.sigmoid().detach()
        lidar_rec_diff = (lidar_rec_prob - voxels).abs().sum() / voxels.shape[0]
        lidar_rec_iou = ((lidar_rec_prob >= 0.5) & (voxels >= 0.5)).sum() / (
            (lidar_rec_prob >= 0.5) | (voxels >= 0.5)
        ).sum()
        code_util = (self.code_age < self.vector_quantizer.dead_limit).sum() / self.code_age.numel()
        code_uniformity = self.code_usage.topk(10)[0].sum() / self.code_usage.sum()

        losses = dict()

        losses.update(
            {
                "loss_lidar_rec": lidar_rec_loss,
                "loss_emb": sum(emb_loss) * 10,
                "lidar_rec_diff": lidar_rec_diff,
                "lidar_rec_iou": lidar_rec_iou,
                "code_util": code_util,
                "code_uniformity": code_uniformity,
            }
        )
        return losses


    def train_transformer(self, points):
        with torch.no_grad():
            voxels = self.voxelizer([[_] for _ in points])
            
            voxels = self.aug(voxels)
            lidar_feats = self.lidar_encoder(voxels)
            feats = self.pre_quant(lidar_feats)
            code, _, code_indices = self.vector_quantizer(feats, self.code_age, self.code_usage)



        x, mask, ids_restore = self.mask_code(code, self.maskgit_transformer.mask_token)
        pred = self.maskgit_transformer(x)
        mask = mask.flatten(0, 1)
        

        loss = (
            F.cross_entropy(pred.flatten(0, 1), code_indices, reduction="none", label_smoothing=0.1) * mask
        ).sum() / mask.sum()

        acc = (pred.flatten(0, 1).max(dim=-1)[1] == code_indices)[mask > 0].float().mean()

        losses = dict()
        losses.update(
            {
                "loss_code_pred": loss,
                "acc": acc,
            }
        )
        return losses


    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def mask_code(self, code, mask_token, cond=None, mask_ratio=None):
        if mask_ratio == None:
            mask_ratio = self.gamma(np.random.uniform())

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(code, mask_ratio)
        # append mask tokens to sequence

        mask_tokens = mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x = torch.cat([x, mask_tokens], dim=1)  # no cls token
        x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

        if cond is not None:
            x = torch.cat([x, cond], dim=-1)

        return x, mask, ids_restore


    def forward_train(
        self,
        points=None,
        **kwargs,
    ):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """

        if self.model_type == "codebook_training":
            losses = self.train_codebook(points)
        elif self.model_type == "transformer_training":
            losses = self.train_transformer(points)

        return losses

    def unconditional_generation(self, points):
        self.iter += 1
        voxels = self.voxelizer([[_] for _ in points])

        choice_temperature = 2.0
        folder_name = "ultralidar_samples"

        x = self.maskgit_transformer.mask_token.repeat(1, self.maskgit_transformer.img_size**2, 1)
        code_idx = torch.ones((1, self.maskgit_transformer.img_size**2), dtype=torch.int64, device=x.device) * -1
        num_unknown_code = (code_idx == -1).sum(dim=-1)

        os.makedirs(f"{folder_name}", exist_ok=True)

        for t in range(self.T):
            pred = self.maskgit_transformer(x)

            if t < 10:
                pred[..., self.BLANK_CODE] = -10000
            
            sample_ids = torch.distributions.Categorical(logits=pred).sample()

            prob = torch.softmax(pred, dim=-1)
            prob = torch.gather(prob, -1, sample_ids.unsqueeze(-1)).squeeze(-1)
            sample_ids[code_idx != -1] = code_idx[code_idx != -1]
            prob[code_idx != -1] = 1e10

            ratio = 1.0 * (t + 1) / self.T
            mask_ratio = self.gamma(ratio)

            mask_len = num_unknown_code * mask_ratio
            mask_len = torch.minimum(mask_len, num_unknown_code - 1)
            mask_len = mask_len.clamp(min=1).long()

            temperature = choice_temperature * (1.0 - ratio)
            gumbels = -torch.empty_like(prob, memory_format=torch.legacy_contiguous_format).exponential_().log()
            confidence = prob.log() + temperature * gumbels



            cutoff = torch.sort(confidence, dim=-1)[0][
                torch.arange(mask_len.shape[0], device=mask_len.device), mask_len
            ].unsqueeze(1)
            
            mask = confidence < cutoff

            x = self.vector_quantizer.get_codebook_entry(sample_ids)

            code_idx = sample_ids.clone()

            if t != self.T - 1:

                
                code_idx[mask] = -1
                x[mask] = self.maskgit_transformer.mask_token

        lidar_rec = self.lidar_decoder(x)
        

        rank = 0
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
        denoised_generated_sample = self.denoise(gumbel_sigmoid(lidar_rec, hard=True))
        if self.vis_flag:
            draw_bev_lidar(voxels, "{}/Rank{}_No{}_gt.png".format(folder_name, rank, self.iter))
            draw_bev_lidar(denoised_generated_sample, "{}/Rank{}_No{}_generated.png".format(folder_name, rank, self.iter))

        geneted_points = self.voxels2points(denoised_generated_sample)
        gt_points = self.voxels2points(voxels)
        
        dump_ply("{}/Rank_{}_No{}_denoised_gene.ply".format(folder_name, rank, self.iter), geneted_points.detach().cpu().numpy())

        return gt_points, geneted_points


    def denoise(self, voxels, mask_ratio=0.5):
        for _ in range(8):
            voxels = self.conditional_generation(voxels)

        return voxels

    def conditional_generation(self, voxels):
        lidar_feats = self.lidar_encoder(voxels)
        feats = self.pre_quant(lidar_feats)
        gt_code, _, code_indices = self.vector_quantizer(feats, self.code_age, self.code_usage)


        masked_code, mask, ids_restore = self.mask_code(gt_code, self.maskgit_transformer.mask_token, mask_ratio=0.5)

        predicted_tokens = self.maskgit_transformer(masked_code)

        sampled_ids = predicted_tokens.max(dim=-1)[1]
        masked_code[mask != 0] = self.vector_quantizer.get_codebook_entry(sampled_ids)[mask != 0]
        lidar_rec = self.lidar_decoder(masked_code)

        return gumbel_sigmoid(lidar_rec, hard=True)


    def voxels2points(self, voxels):

        non_zero_indices = torch.nonzero(voxels)
        xy = (non_zero_indices[:, 2:] * self.voxelizer.step) + self.voxelizer.y_min
        z = (non_zero_indices[:, 1] * self.voxelizer.z_step) + self.voxelizer.z_min
        xyz = torch.cat([xy, z.unsqueeze(1)], dim=1)

        return xyz
    
    def static_blank_code(self):
        
        x = self.maskgit_transformer.mask_token.repeat(1, self.maskgit_transformer.img_size**2, 1)
        code_idx = torch.ones((1, self.maskgit_transformer.img_size**2), dtype=torch.int64, device=x.device) * -1
        num_unknown_code = (code_idx == -1).sum(dim=-1)
        self.iter += 1
        pred = self.maskgit_transformer(x)
        sample_ids = torch.distributions.Categorical(logits=pred).sample()
        codes, counts = count(sample_ids)
        for code_idx, conut_nbs in zip(codes, counts):
            self.code_dict[int(code_idx.data)] += conut_nbs
        blank_code = [k for k, v in sorted(self.code_dict.items(), key=lambda item: item[1], reverse=True)[:20]]
        if self.iter == 100:
            print("\nThe blank code has already been successfully counted, and will exit automatically soon")
            mmcv.dump(blank_code, "blank_code.pkl")
            sys.exit(0)

            

    def forward_test(
        self,
        points=None,
        **kswargs,
    ):  
        if self.model_type == 'static_blank_code':
            self.static_blank_code()
        else:
            self.BLANK_CODE = mmcv.load("blank_code.pkl")
            gt_points, generated_points = self.unconditional_generation(points)
        

        return [0]









