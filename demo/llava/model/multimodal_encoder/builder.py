from .clip_encoder import CLIPVisionTower



def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(
        vision_tower_cfg,
        "mm_vision_tower",
        getattr(vision_tower_cfg, "vision_tower", None),
    )
    if vision_tower.startswith("openai") or vision_tower.startswith("laion"):
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif vision_tower == "pointcloud-perceiver":
        return PointCloudPerceiverVisionTower(vision_tower, args=vision_tower_cfg)
    elif vision_tower == "pointcloud-minkowski":
        return PointCloudMinkowskiVisionTower(vision_tower, args=vision_tower_cfg)
    elif vision_tower == "pointcloud-minkowski-mask3d":
        return PointCloudMinkowskiMask3DVisionTower(vision_tower, args=vision_tower_cfg)
    elif vision_tower == "bbox-ground-truth":
        return None  # return None so that there is no vision input to the LLM, see prepare_inputs_labels_for_multimodal() in llava_arch.py

    raise ValueError(f"Unknown vision tower: {vision_tower}")
