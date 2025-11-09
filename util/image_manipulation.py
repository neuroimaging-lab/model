import torch


def cut_images_and_masks( #TODO: move to train_support.py and remove this file
    images: torch.Tensor, masks: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Reduces voxel dimensions for the images and corresponding masks, by center-cropping.
    Used as a test tool for the local development - (e.g. 160x240x240 → 80x160x160) - to be able to fit an MRI image inside a model with full channels.
    """
    depth = 80  # base 160
    height = 160  # base 240
    width = 160  # base 240

    start_d = (160 - depth) // 2
    start_h = (240 - height) // 2
    start_w = (240 - width) // 2

    return images[
        :,
        :,
        start_d : start_d + depth,
        start_h : start_h + height,
        start_w : start_w + width,
    ], masks[
        :,
        :,
        start_d : start_d + depth,
        start_h : start_h + height,
        start_w : start_w + width,
    ]
