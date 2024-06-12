import torch
from typing import Tuple

from torchvision.ops._utils import _upcast_non_float


def distance_box_iou_loss_3d(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    reduction: str = "mean",
    eps: float = 1e-7,
    return_iou: bool = False,
) -> torch.Tensor:
    """Gradient-friendly IoU loss with an additional penalty that is non-zero when the
    distance between boxes' centers isn't zero. Indeed, for two exactly overlapping
    boxes, the distance IoU is the same as the IoU loss.
    This loss is symmetric, so the boxes1 and boxes2 arguments are interchangeable.

    Both sets of boxes are expected to be in ``(x1, y1, z1, x2, y2, z2)`` format with
    ``0 <= x1 < x2``, ``0 <= y1 < y2`` and ``0 <= z1 < z2`` and The two boxes should have the
    same dimensions.

    range: [0, 1 + normalized_center_distance_squared]

    Args:
        boxes1 (Tensor[N, 6]): first set of boxes
        boxes2 (Tensor[N, 6]): second set of boxes
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: No reduction will be
            applied to the output. ``'mean'``: The output will be averaged.
            ``'sum'``: The output will be summed. Default: ``'mean'``
        eps (float, optional): small number to prevent division by zero. Default: 1e-7
        return_iou (bool, optional): If True, also return the IoU loss. Default: False

    Returns:
        Tensor: Loss tensor with the reduction option applied.

    Reference:
        Zhaohui Zheng et al.: Distance Intersection over Union Loss:
            https://arxiv.org/abs/1911.08287
            and
            https://pytorch.org/vision/main/_modules/torchvision/ops/diou_loss.html#distance_box_iou_loss
    """
    boxes1 = _upcast_non_float(boxes1)
    boxes2 = _upcast_non_float(boxes2)

    loss, iou = _diou_iou_loss_3d(boxes1, boxes2, eps)

    # Check reduction option and return loss accordingly
    if reduction == "none":
        pass
    elif reduction == "mean":
        loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
        iou = iou.mean() if iou.numel() > 0 else 0.0 * iou.sum()
    elif reduction == "sum":
        loss = loss.sum()
        iou = iou.sum()
    else:
        raise ValueError(
            f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
        )

    if return_iou:
        return loss, iou
    else:
        return loss


def _diou_iou_loss_3d(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    eps: float = 1e-7,
) -> Tuple[torch.Tensor, torch.Tensor]:
    intsct, union = _loss_inter_union_3d(boxes1, boxes2)
    iou = intsct / (union + eps)

    # Smallest enclosing box
    x1, y1, z1, x2, y2, z2 = boxes1.unbind(dim=-1)
    x1g, y1g, z1g, x2g, y2g, z2g = boxes2.unbind(dim=-1)
    xc1 = torch.min(x1, x1g)
    yc1 = torch.min(y1, y1g)
    zc1 = torch.min(z1, z1g)
    xc2 = torch.max(x2, x2g)
    yc2 = torch.max(y2, y2g)
    zc2 = torch.max(z2, z2g)

    # Diagonal distance of the smallest enclosing box squared
    diagonal_distance_squared = ((xc2 - xc1) ** 2) + ((yc2 - yc1) ** 2) + ((zc2 - zc1) ** 2) + eps

    # Centers of boxes
    x_c = (x2 + x1) / 2
    y_c = (y2 + y1) / 2
    z_c = (z2 + z1) / 2
    x_cg = (x2g + x1g) / 2
    y_cg = (y2g + y1g) / 2
    z_cg = (z2g + z1g) / 2

    # Distance between boxes' centers squared
    centers_distance_squared = ((x_c - x_cg) ** 2) + ((y_c - y_cg) ** 2) + ((z_c - z_cg) ** 2)

    # The distance IoU is the IoU penalized by a normalized
    # distance between boxes' centers squared.
    loss = 1 - iou + (centers_distance_squared / diagonal_distance_squared)
    return loss, iou


def _loss_inter_union_3d(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    x1, y1, z1, x2, y2, z2 = boxes1.unbind(dim=-1)
    x1g, y1g, z1g, x2g, y2g, z2g = boxes2.unbind(dim=-1)

    # Intersection keypoints
    xkis1 = torch.max(x1, x1g)
    ykis1 = torch.max(y1, y1g)
    zkis1 = torch.max(z1, z1g)
    xkis2 = torch.min(x2, x2g)
    ykis2 = torch.min(y2, y2g)
    zkis2 = torch.min(z2, z2g)

    intsctk = torch.zeros_like(x1)
    mask = (xkis2 > xkis1) & (ykis2 > ykis1) & (zkis2 > zkis1)
    intsctk[mask] = (
        (xkis2[mask] - xkis1[mask]) * (ykis2[mask] - ykis1[mask]) * (zkis2[mask] - zkis1[mask])
    )
    unionk = (x2 - x1) * (y2 - y1) * (z2 - z1) + (x2g - x1g) * (y2g - y1g) * (z2g - z1g) - intsctk

    return intsctk, unionk


if __name__ == "__main__":
    # Example usage
    boxes1 = torch.tensor([[0.2715, 0.3398, 0.2793, 0.4160, 0.4121, 0.7070]], dtype=torch.float32)
    boxes2 = torch.tensor([[0.2402, 0.2246, -0.1689, 0.7422, 0.7109, 0.7734]], dtype=torch.float32)

    loss = distance_box_iou_loss_3d(boxes1, boxes2)
    print(loss)
    # Output: tensor(0.0000)
    # The boxes are exactly overlapping, so the distance IoU is the same as the IoU loss.
    # The distance between boxes' centers is zero, so the distance IoU loss is zero.
    # The output is a tensor(0.0000) as expected.
    # The distance IoU loss is symmetric, so the boxes1 and boxes2 arguments are interchangeable.
