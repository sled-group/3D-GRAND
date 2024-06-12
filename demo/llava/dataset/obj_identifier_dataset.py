from collections import OrderedDict
import torch
import os
import copy
from dataclasses import dataclass
import json
import re
from typing import Dict, Optional, Sequence


import transformers

from llava.constants import (
    IGNORE_INDEX,
)
from torch.utils.data import Dataset

from llava.util.tokenization import (
    preprocess_llama_2,
    preprocess_llama_2_obj_identifier,
    preprocess_multimodal,
    preprocess,
)

from llava import conversation as conversation_lib


class ObjIdentifierDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        data_path: str | list,
        scene_to_obj_mapping: str,
        obj_context_feature_type: str = "text",
        mode: str = "train",
        **kwargs,
    ):
        super(ObjIdentifierDataset, self).__init__()

        self.tokenizer = tokenizer
        self.scene_to_obj_mapping = json.load(open(scene_to_obj_mapping, "r"))
        self.update_data(data_path)
        self.obj_context_feature_type = obj_context_feature_type
        self.mode = mode

    def __len__(self):
        return len(self.list_data_dict)

    def update_data(self, data_path: str):
        assert self.scene_to_obj_mapping is not None, "scene_to_obj_mapping needs to be set first."
        if isinstance(data_path, str):
            self.list_data_dict = json.load(open(data_path, "r"))
        elif isinstance(data_path, list):
            self.list_data_dict = data_path

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = copy.deepcopy(self.list_data_dict[i])
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        ############
        scene_id = sources[0]["scene_id"]
        input_obj_dict = copy.deepcopy(self.scene_to_obj_mapping[scene_id])

        # prepare the object-centric features
        # we want the LLM to see:
        # "%%%% Object-centric context: <obj_0>: <obj_0_feat>, <obj_1>: <obj_1_feat>, ..."
        # where <obj_i_feat> will later be replaced by the actual feature in vector form,
        # everything else is pure text string.
        # 1. We need to first change the object_id to a new object_id, e.g., 'obj_0', 'obj_1', ...,
        #   and replace the old object_id with new object_id in the text conversation
        # 2. Tokenize the conversation, and add object context to the tokenized conversation
        # 3. Gather and return the necessary information for each object,
        #   so that it can be later embeded into vector

        # 1. change the object_id to a new object_id
        # original object_id: 'wardrobe-0', 'three-seat/multi-seat sofa-1', ...
        # convert to obj_id: 'obj_0', 'obj_1', ...
        # and remember the mapping
        old_id_to_new_id_mapping = {}
        result_obj_dict = OrderedDict()
        # first pass, map the old object_id to new object_id
        for old_id, obj_info in input_obj_dict.items():
            # make sure old_id doesn't contain < or >
            assert (
                "<" not in old_id and ">" not in old_id
            ), "object_id in scene graph should not contain < or >"
            new_id = f"obj_{len(old_id_to_new_id_mapping)}"
            old_id_to_new_id_mapping[old_id] = new_id
        # second pass, create the new object-centric context, modify the object_id in the text content
        for old_id, obj_info in input_obj_dict.items():
            new_id = old_id_to_new_id_mapping[old_id]
            # TODO: Determine what information to include in the object-centric context
            result_obj_info_dict = {}
            result_obj_info_dict["category"] = obj_info["category"]
            # result_obj_info_dict["category_id"] = obj_info["category_id"]
            # for relations, we need to replace the old object_id with new object_id
            # result_obj_info_dict["relations"] = []
            # for relation in obj_info["relations"]:
            #     for local_old_id, local_new_id in old_id_to_new_id_mapping.items():
            #         if local_old_id in relation:
            #             result_obj_info_dict["relations"].append(
            #                 re.sub(rf"<{local_old_id}>", f"<{local_new_id}>", relation)
            #             )
            if "description" in obj_info:
                result_obj_info_dict["description"] = obj_info["description"]
            else:
                # print(f"WARNING: Object {old_id} does not have a description.")
                pass

            # use two decimal places for the centroid and extent
            result_obj_info_dict["centroid"] = (
                f"[{obj_info['centroid'][0]:.2f}, {obj_info['centroid'][1]:.2f}, {obj_info['centroid'][2]:.2f}]"
            )
            result_obj_info_dict["extent"] = (
                f"[{obj_info['extent'][0]:.2f}, {obj_info['extent'][1]:.2f}, {obj_info['extent'][2]:.2f}]"
            )
            result_obj_dict[new_id] = result_obj_info_dict

        # replace the old object_id with new object_id in the text content
        # text conversation example:
        #     {
        #     "id": "55f2b905-d367-443d-8f88-ef71b958c81f@LivingRoom-3973@1",
        #     "scene_id": "55f2b905-d367-443d-8f88-ef71b958c81f@LivingRoom-3973",
        #     "conversations": [
        #         {
        #             "from": "human",
        #             "value": "Can you describe the ambiance of this room?"
        #         },
        #         {
        #             "from": "gpt",
        #             "value": "In this Living Room, the arrangement of furniture caters to both style and function. The <p>warm wooden hue wardrobe</p>[<wardrobe-0>] stands with a retro flair, while the <p>neutral grey, sleek rectangular form multi-seat sofa</p>[<three-seat/multi-seat sofa-1>] and <p>neutral grey, sleek rectangular form three-seat</p>[<three-seat/multi-seat sofa-8>] provide modern and comfortable seating options. The <p>sleek black, dark grey, brown, rectangular coffee table</p>[<coffee table-2>] and <p>sleek black, dark grey, brown coffee table</p>[<coffee table-3>] in minimalist style serve as focal points and functional pieces for gatherings. The <p>light grey and dark grey (two-tone), shell-like backrest, smooth armchair</p>[<armchair-4>] and <p>light grey and dark grey (two-tone), smooth armchair</p>[<armchair-5>] add additional seating, complemented by the <p>rich walnut brown top and contrasting light grey base side table</p>[<corner/side table-6>] for convenience. Suspended above, the <p>gradient of grey to bronze hues, cylindrical with abstract cityscape cutouts pendant lamp</p>[<pendant lamp-7>] offers a decorative element with its unique Chinoiserie design. The room's setup is ideal for hosting guests or enjoying quiet evenings, with thoughtful placement of each piece to enhance the living experience."
        #         }
        #     ]
        # },
        for conv in sources[0]["conversations"]:
            for old_id, new_id in old_id_to_new_id_mapping.items():
                conv["value"] = re.sub(rf"<{old_id}>", f"<{new_id}>", conv["value"])

        # if in generate mode, shave off the last conversation if it is from the assistant
        if self.mode == "generate" and sources[0]["conversations"][-1]["from"] == "gpt":
            sources[0]["conversations"] = sources[0]["conversations"][:-1]

        # 2. Tokenize the conversation, and add object context to the tokenized conversation
        sources = preprocess_multimodal(
            copy.deepcopy([e["conversations"] for e in sources]),
            is_multimodal=True,
            mm_use_im_start_end=False,
        )
        data_dict = preprocess_llama_2_obj_identifier(
            sources=sources,
            tokenizer=self.tokenizer,
            obj_dict=result_obj_dict,
            obj_context_feature_type=self.obj_context_feature_type,
            mode=self.mode,
        )

        # if in generate mode, add the obj context and bbox label to the data_dict
        # so that we can use them later to compute the metrics
        if self.mode == "generate":
            data_dict["obj_context"] = result_obj_dict
            if "bbox" in self.list_data_dict[i]:
                data_dict["bbox_label"] = self.list_data_dict[i]["bbox"]
            # full_info_dict is the full information of this data sample
            # {'id': 'scene0643_00$desk-0@0', 'scene_id': 'scene0643_00', 'conversations': [{...}],
            # 'referred_object_id': '0', 'referred_object_text': 'desk',
            # 'grounded_object_reference': 'a brown wooden office desk on the left to the gray shelf.',
            # 'bbox': [0.3769365990161897, -0.06906220592784873, -0.020513275327205656, 1.1370925301275254, 1.5355764355778696, 0.8130822017173767]
            # }
            data_dict["full_info_dict"] = self.list_data_dict[i]

        return data_dict


@dataclass
class DataCollatorForObjIdentifierDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, **kwargs):
        self.tokenizer = tokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        return batch


@dataclass
class DataCollatorForBatchDecodingObjIdentifierDataset(object):
    """Collate examples for batch decoding."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, **kwargs):
        self.tokenizer = tokenizer

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "right":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=batch_first, padding_value=padding_value
        )
        if self.tokenizer.padding_side == "right":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [instance["input_ids"] for instance in instances]
        input_ids = self.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )

        batch = dict(input_ids=input_ids)
        if "bbox_label" in instances[0].keys():
            batch["bbox_label"] = [instance["bbox_label"] for instance in instances]

        if "obj_context" in instances[0].keys():
            batch["obj_context"] = [instance["obj_context"] for instance in instances]

        if "full_info_dict" in instances[0].keys():
            batch["full_info_dict"] = [instance["full_info_dict"] for instance in instances]

        return batch


# test the dataset
if __name__ == "__main__":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "/data/jianingy/3d-llama/checkpoints/llava-llama-2-7b-chat-lightning-preview",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    conversation_lib.default_conversation = conversation_lib.conv_templates["llava_llama_2"]

    dataset = ObjIdentifierDataset(
        tokenizer,
        data_path="/home/jianingy/research/LLaVA-original/llava/dataset/3dfront/grounded_scene_description_gpt_format.json",
        scene_to_obj_mapping="/home/jianingy/research/LLaVA-original/llava/dataset/3dfront/compressed_organized_data.json",
    )
    print(len(dataset))
    print(dataset[0])
