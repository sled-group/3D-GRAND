from PIL import Image
from io import BytesIO
import base64

import torch
from transformers import StoppingCriteria
from llava.constants import IMAGE_TOKEN_INDEX
import numpy as np

from llava.model.data_3d_util import (
    compute_max_extent_and_centroid,
    unit_cube_normalization_in_place,
)


def prepare_3d_input_minkowski(pcd_file: str, input_color: bool, voxelizer):
    pcd_data_np = np.load(pcd_file)  # (num_points, 6)
    locs_in = pcd_data_np[:, 3:]
    rgb = np.rint(pcd_data_np[:, :3] * 255).astype(int)
    # common trick to change the range of [-1, 1]
    feats_in = rgb / 127.5 - 1
    # keep this operation if we have other data format
    feats_in = (feats_in + 1.0) * 127.5
    labels_in = torch.ones(locs_in.shape[0]).int()

    locs, feats, labels, inds_reconstruct = voxelizer.voxelize(locs_in, feats_in, labels_in)
    coords = torch.from_numpy(locs).int()
    coords = torch.cat((torch.zeros(coords.shape[0], 1, dtype=torch.int), coords), dim=1)

    if input_color:
        feats = torch.from_numpy(feats).float() / 127.5 - 1.0
    else:
        feats = torch.ones(coords.shape[0], 3)

    return coords, feats, inds_reconstruct


def prepare_3d_input(
    pcd_file: str,
    max_num_points: int,
    is_normalize_points_to_unit_cube: bool,
    mm_vision_tower: str,
) -> torch.Tensor:
    # TODO: add support for MinkNet
    if mm_vision_tower == "pointcloud-perceiver":
        pcd_data = np.load(pcd_file)  # (num_points, 768)
        # note that the convention of the last 3 dimension of any npy flle is x, y, z
        pcd_data_xyz = pcd_data[:, -3:]
        max_extent, centroid = compute_max_extent_and_centroid(pcd_data_xyz, epsilon=1e-4)
        if is_normalize_points_to_unit_cube:
            unit_cube_normalization_in_place(pcd_data_xyz, max_extent, centroid)
        pcd_data = torch.from_numpy(pcd_data).float()
        pcd_attention_mask = torch.ones(pcd_data.shape[0], dtype=pcd_data.dtype)

        # truncate or pad to NUM_POINTS datapoints along dim 0
        if pcd_data.shape[0] > max_num_points:
            pcd_data = pcd_data[:max_num_points, :]
            pcd_attention_mask = pcd_attention_mask[:max_num_points]
        elif pcd_data.shape[0] < max_num_points:
            padding = torch.zeros(
                (
                    max_num_points - pcd_data.shape[0],
                    pcd_data.shape[1],
                )
            )
            pcd_data = torch.cat((pcd_data, padding), dim=0)

            # extend the attention mask with zeros for the padding points
            attention_mask_padding = torch.zeros(padding.shape[0], dtype=torch.bool)
            pcd_attention_mask = torch.cat((pcd_attention_mask, attention_mask_padding))

        # output shape: (num_points, 768 + 1) where the last dimension is the attention mask
        output_tensor = torch.cat((pcd_data, pcd_attention_mask.unsqueeze(1)), dim=1)
    return output_tensor.unsqueeze(0)  # add batch dimension


def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


def process_images(images, image_processor, model_cfg):
    return image_processor(images, return_tensors="pt")["pixel_values"]


def tokenizer_image_token(
    prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None
):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split("<image>")]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if (
        len(prompt_chunks) > 0
        and len(prompt_chunks[0]) > 0
        and prompt_chunks[0][0] == tokenizer.bos_token_id
    ):
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == "pt":
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f"Unsupported tensor type: {return_tensors}")
    return input_ids


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith("checkpoint-"):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        assert output_ids.shape[0] == 1, "Only support batch size 1 (yet)"  # TODO
        offset = min(output_ids.shape[1] - self.start_len, 3)
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            if output_ids[0, -keyword_id.shape[0] :] == keyword_id:
                return True
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False
