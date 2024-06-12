#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import time

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    LlamaConfig,
    LlamaModel,
    LlamaForCausalLM,
)

from transformers.modeling_outputs import CausalLMOutputWithPast

from llava.constants import GROUND_TOKEN, PROFILE_RUNTIME
from llava.model.iou_3d_loss import distance_box_iou_loss_3d
from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM



from transformers.utils import logging

logger = logging.get_logger("transformers")


class LlavaConfig(LlamaConfig):
    model_type = "llava"

    def __init__(self, **kwargs):
        self.lm_loss_weight = kwargs.pop("lm_loss_weight", 1.0)
        self.use_bbox_iou_loss = kwargs.pop("use_bbox_iou_loss", None)
        self.bbox_iou_loss_weight = kwargs.pop("bbox_iou_loss_weight", None)
        self.use_bbox_mse_loss = kwargs.pop("use_bbox_mse_loss", None)
        self.bbox_mse_loss_weight = kwargs.pop("bbox_mse_loss_weight", None)
        self.use_bbox_ce_loss = kwargs.pop("use_bbox_ce_loss", None)
        self.bbox_ce_loss_weight = kwargs.pop("bbox_ce_loss_weight", None)
        self.num_latents = kwargs.pop("num_latents", None)
        self.d_latents = kwargs.pop("d_latents", None)
        self.vision_tower = kwargs.pop("vision_tower", None)
        super().__init__(**kwargs)


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


@dataclass
class CausalLMOutputWithPastWithBbox(CausalLMOutputWithPast):
    total_loss: Optional[torch.FloatTensor] = None
    lm_loss: Optional[torch.FloatTensor] = None
    bbox_iou_loss: Optional[torch.FloatTensor] = None
    bbox_mse_loss: Optional[torch.FloatTensor] = None
    bbox_ce_loss: Optional[torch.FloatTensor] = None
    bbox_iou: Optional[torch.FloatTensor] = None

    @classmethod
    def ignore_keys_for_eval(cls):
        # only keep the losses values for validation during training
        # keys left: 0: "total_loss", 1: "lm_loss", 2: "bbox_iou_loss", 3: "bbox_mse_loss", 4: "bbox_iou"
        return [
            "logits",
            "past_key_values",
            "hidden_states",
            "attentions",
        ]


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config, **kwargs):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # a MLP bbox regression head
        if (
            self.config.use_bbox_iou_loss
            or self.config.use_bbox_mse_loss
            or self.config.use_bbox_ce_loss
        ):
            if self.config.vision_tower == "bbox-ground-truth":
                self.bbox_head = BBoxHeadForGroundTruthBboxSelectionMLPFusionBoxCoordsAndClassID(
                    lm_feat_dim_in=config.hidden_size,
                    vision_feat_dim_in=config.d_latents,
                    num_vision_feat=config.num_latents,
                )
            else:
                # self.bbox_head = BBoxHead(lm_feat_dim_in=config.hidden_size, vision_feat_dim_in=d_latents)
                self.bbox_head = SimpleBBoxHead(
                    lm_feat_dim_in=config.hidden_size,
                    vision_feat_dim_in=config.d_latents,
                    num_vision_feat=config.num_latents,
                )

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        coords_minknet: Optional[torch.Tensor] = None,
        feats_minknet: Optional[torch.Tensor] = None,
        inds_reconstruct_minknet: Optional[torch.LongTensor] = None,
        bbox_labels: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        forward function

        Args:
            input_ids (torch.LongTensor, optional): Tensor of token indices to be processed by the model.
            attention_mask (Optional[torch.Tensor], optional): Mask to avoid performing attention on padding token indices.
            past_key_values (Optional[List[torch.FloatTensor]], optional): List of tensors containing past key values for attention layers.
            inputs_embeds (Optional[torch.FloatTensor], optional): Inputs embeddings for model processing.
            labels (Optional[torch.LongTensor], optional): Labels for supervised training.
            use_cache (Optional[bool], optional): Whether to use caching for faster generation of sequences.
            output_attentions (Optional[bool], optional): Whether to return attentions weights.
            output_hidden_states (Optional[bool], optional): Whether to return hidden states of the model.
            images (Optional[torch.FloatTensor], optional): Tensor for image inputs if the model is configured for vision tasks.
            return_dict (Optional[bool], optional): Whether to return a `ModelOutput` instead of a plain tuple.
            coords_minknet (Optional[torch.Tensor], optional): Coordinates tensor for Minkowski network, detailing spatial structure. (N, 4)
            feats_minknet (Optional[torch.Tensor], optional): Features tensor for Minkowski network, specifying attributes at each coordinate. (N, 3)
            inds_reconstruct_minknet (Optional[torch.LongTensor], optional): Index tensor to map Minkowski network outputs back to original point cloud. (N_origin,)
            bbox_labels (Optional[torch.FloatTensor], optional): Bounding box labels for supervised training.

        Returns:
            Union[Tuple, CausalLMOutputWithPast]
        """
        ########################################
        # profile the time cost of each forward pass
        start_time_foward = time.time()

        # data preprocessing for MinkowskiEngine
        if images is None and coords_minknet is not None:
            # this is the input to the model for MinkowskiEngine,
            # we need to convert it to SparseTensor and put it into `images`
            sparse_tensor_minknet_input = SparseTensor(
                features=feats_minknet.to(dtype=torch.float32).squeeze(),
                coordinates=coords_minknet.squeeze(),
            )  # MinkowskiEngine only supports float32, so we need to convert the input to float32, note that .to() is also differentiable
            images = sparse_tensor_minknet_input
        ########################################
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        start_time_prepare_inputs_labels_for_multimodal = time.time()
        (
            input_ids,
            attention_mask,
            past_key_values,
            inputs_embeds,
            labels,
            vision_features_before_mm_projection,  # (B, num_latents, d_latents)
        ) = self.prepare_inputs_labels_for_multimodal(
            input_ids, attention_mask, past_key_values, labels, images
        )
        if PROFILE_RUNTIME:
            logger.info(
                f"prepare_inputs_labels_for_multimodal time: {time.time() - start_time_prepare_inputs_labels_for_multimodal}"
            )

        start_time_llm_forward = time.time()
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        lm_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if PROFILE_RUNTIME:
            logger.info(f"llm_forward time: {time.time() - start_time_llm_forward}")

        hidden_states = lm_outputs[0]
        logits = self.lm_head(hidden_states)  # (B, L, V)

        # compute bbox loss
        start_time_bbox_loss = time.time()
        if (
            self.config.use_bbox_iou_loss
            or self.config.use_bbox_mse_loss
            or self.config.use_bbox_ce_loss
        ):
            assert labels is not None and bbox_labels is not None
            shifted_hidden_states = hidden_states[
                ..., :-1, :
            ]  # (B, L-1, D), -1 to remove the last token
            shifted_labels = labels[..., 1:]  # (B, L-1), -1 to remove the first token
            grd_token_pos = shifted_labels.eq(
                self.config.added_special_token_to_input_id[GROUND_TOKEN]
            )  # (B, L-1) # ground token positions
            # Get the hidden states of the ground tokens
            grd_token_hidden_states_list = (
                []
            )  # each element contain the hidden states of the ground tokens in one sample
            for i in range(shifted_hidden_states.size(0)):  # iterate over the batch dimension
                grd_token_hidden_states_list.append(shifted_hidden_states[i, grd_token_pos[i]])

            assert sum([e.shape[0] for e in grd_token_hidden_states_list]) == bbox_labels.shape[0]

            bbox_scores = self.bbox_head(
                grd_token_hidden_states_list,
                vision_features_before_mm_projection,
            )  # (N, num_boxes)

            # calculate CE loss for bbox
            # first get which box is the ground truth box
            bbox_idx = 0
            gt_bbox_idx_list = []
            bbox_pred_list = []
            for i, hidden_states_in_one_sample in enumerate(
                grd_token_hidden_states_list
            ):  # iterate over the batch dimension
                for j in range(hidden_states_in_one_sample.shape[0]):
                    min_diff, min_idx = torch.min(
                        (images[i, :, 0:6] - bbox_labels[bbox_idx]).norm(dim=-1), dim=0
                    )
                    gt_bbox_idx_list.append(min_idx)
                    assert (
                        min_diff < 1e-1
                    ), f"min_diff: {min_diff}, min_idx: {min_idx}, bbox_labels[bbox_idx]: {bbox_labels[bbox_idx]}, images[i, :, 0:6]: {images[i, :, 0:6]}"
                    # get the bbox prediction
                    bbox_pred_idx = bbox_scores[bbox_idx].argmax()  # (1,)
                    bbox_pred = images[i, bbox_pred_idx][0:6]  # (6,)
                    bbox_pred_list.append(bbox_pred)
                    bbox_idx += 1

            gt_bbox_idx = torch.stack(gt_bbox_idx_list)  # (N,)
            bbox_preds = torch.stack(bbox_pred_list)  # (N, 6)

            # then calculate CE loss
            bbox_ce_loss_fct = nn.CrossEntropyLoss()
            bbox_ce_loss = bbox_ce_loss_fct(bbox_scores, gt_bbox_idx)

            bbox_iou_loss_fct = distance_box_iou_loss_3d
            bbox_mse_loss_fct = nn.MSELoss()
            assert bbox_preds.shape[0] == bbox_labels.shape[0]
            _, bbox_iou = bbox_iou_loss_fct(bbox_preds, bbox_labels, return_iou=True)
            bbox_iou_loss = 1 - bbox_iou  # range: [0, 1]
            bbox_mse_loss = bbox_mse_loss_fct(bbox_preds, bbox_labels)

            # log one bbox prediction for debugging
            logger.info(f"DEBUG: bbox_labels[0]: {bbox_labels[0]}")
            logger.info(f"DEBUG: bbox_preds[0]: {bbox_preds[0]}")
            logger.info(f"DEBUG: bbox_iou for batch: {bbox_iou}")
        else:
            bbox_iou_loss = None
            bbox_iou = None
            bbox_mse_loss = None
            bbox_ce_loss = None
        if PROFILE_RUNTIME:
            logger.info(f"bbox_loss time: {time.time() - start_time_bbox_loss}")

        # compute language modeling loss
        total_loss = None
        lm_loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            lm_loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            lm_loss = lm_loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + lm_outputs[1:]
            return (lm_loss,) + output if lm_loss is not None else output

        if lm_loss is not None:
            total_loss = lm_loss * self.config.lm_loss_weight
        if bbox_iou_loss is not None:
            total_loss = total_loss + bbox_iou_loss * self.config.bbox_iou_loss_weight
        if bbox_mse_loss is not None:
            total_loss = total_loss + bbox_mse_loss * self.config.bbox_mse_loss_weight
        if bbox_ce_loss is not None:
            total_loss = total_loss + bbox_ce_loss * self.config.bbox_ce_loss_weight

        if PROFILE_RUNTIME:
            logger.info(f"foward time: {time.time() - start_time_foward}")

        return CausalLMOutputWithPastWithBbox(
            total_loss=total_loss,
            lm_loss=lm_loss,
            bbox_iou_loss=bbox_iou_loss,
            bbox_mse_loss=bbox_mse_loss,
            bbox_ce_loss=bbox_ce_loss,
            bbox_iou=bbox_iou,
            logits=logits,
            past_key_values=lm_outputs.past_key_values,
            hidden_states=lm_outputs.hidden_states,
            attentions=lm_outputs.attentions,
        )

    def predict_bboxes(
        self,
        input_ids: torch.LongTensor,
        lm_hidden_states: torch.FloatTensor,
    ) -> dict[str, torch.Tensor]:
        """
        predict bounding boxes

        Args:
            input_ids (torch.LongTensor): tokenized input, shape (B, L)
            lm_hidden_states (torch.FloatTensor): hidden states from the language model, shape (B, L, D)

        Returns:
            dict[str, torch.Tensor]: dictionary of tensors:
                1. predicted bounding boxes
                2. number of ground phrases
        """
        grd_token_pos = input_ids.eq(
            self.self.config.added_special_token_to_input_id[GROUND_TOKEN]
        )  # (B, L)
        num_grd_phrases = grd_token_pos.sum(dim=1).long()  # (B,)
        grd_token_hs = lm_hidden_states[grd_token_pos]  # (N, D), N is the number of ground tokens

        # compute the bbox predictions
        bbox_preds = self.bbox_head(grd_token_hs)  # (N, 6)

        ret = {
            "bbox_preds": bbox_preds,
            "num_grd_phrases": num_grd_phrases,
        }
        return ret

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
                "coords_minknet": kwargs.get("coords_minknet", None),
                "feats_minknet": kwargs.get("feats_minknet", None),
                "inds_reconstruct_minknet": kwargs.get("inds_reconstruct_minknet", None),
            }
        )
        return model_inputs


AutoConfig.register("llava", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
