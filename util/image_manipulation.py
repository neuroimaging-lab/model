def cut_images_and_masks(images, masks):
    depth = 64  # base 160
    height = 112  # base 240
    width = 112  # base 240

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
