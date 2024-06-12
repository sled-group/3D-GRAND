# model.py
import spaces
import json
import torch
from torch.utils.data import DataLoader

from llava.dataset.obj_identifier_dataset import (
    ObjIdentifierDataset,
    DataCollatorForBatchDecodingObjIdentifierDataset,
)
from llava.mm_utils import get_model_name_from_path
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

def load_model_and_dataloader(model_path, model_base, scene_to_obj_mapping, obj_context_feature_type="text", load_8bit=False, load_4bit=False, load_bf16=False, device_map='auto'):

    model_name = get_model_name_from_path(model_path)

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=model_base,
        model_name=model_name,
        load_8bit=load_8bit,
        load_4bit=load_4bit,
        load_bf16=load_bf16,
        device_map=device_map,
    )

    dataset = ObjIdentifierDataset(
        tokenizer=tokenizer,
        data_path=[],
        scene_to_obj_mapping=scene_to_obj_mapping,
        obj_context_feature_type=obj_context_feature_type,
        mode="generate",
    )
    collator = DataCollatorForBatchDecodingObjIdentifierDataset(tokenizer=tokenizer)
    data_loader = DataLoader(
        dataset, collate_fn=collator, batch_size=1, num_workers=0, shuffle=False
    )

    return tokenizer, model, data_loader


def get_model_response(model, tokenizer, data_loader, scene_id, user_input, max_new_tokens=50, temperature=0.2, top_p=0.9):
    input_data = [
        {
            "id": f"interactive@{scene_id}@input",
            "scene_id": scene_id,
            "conversations": [{"from": "human", "value": f"Ground these sentences: <refer_expression>{user_input}<refer_expression>\n<image>"}],
            "ground_truth": "User provided description for interactive session.",
        }
    ]

    data_loader.dataset.update_data(input_data)

    for batch in data_loader:
        input_ids = batch["input_ids"].squeeze(dim=1).cuda()
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                use_cache=True,
            )
        prompt = tokenizer.batch_decode(input_ids)[0]
        outputs = tokenizer.batch_decode(output_ids[:, input_ids.shape[-1]:], skip_special_tokens=True)
    
    return prompt, outputs[0]
