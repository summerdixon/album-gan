import torch
import torch.nn.functional as F


def normalize_embedding(emb: "torch.Tensor | list | tuple") -> torch.Tensor:
    if not torch.is_tensor(emb):
        emb = torch.tensor(emb)

    emb = emb.float()

    emb = emb.squeeze()

    if emb.dim() > 1:
        emb = emb.view(-1)

    return emb


def resize_image_tensor(image_tensor: torch.Tensor, size: int | tuple[int, int], mode: str = "bicubic") -> torch.Tensor:
    is_single_image = image_tensor.dim() == 3

    if is_single_image:
        image_tensor = image_tensor.unsqueeze(0)
    elif image_tensor.dim() != 4:
        raise ValueError(f"Expected a 3D or 4D image tensor, got shape {tuple(image_tensor.shape)}")

    if mode in {"bilinear", "bicubic", "trilinear"}:
        resized = F.interpolate(image_tensor, size=size, mode=mode, align_corners=False)
    else:
        resized = F.interpolate(image_tensor, size=size, mode=mode)

    return resized.squeeze(0) if is_single_image else resized
