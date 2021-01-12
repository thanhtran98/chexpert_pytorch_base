import numpy as np
import cv2

def border_pad(image, pixel_mean=128.0, long_side=512):
    h, w, c = image.shape
    image = np.pad(image, ((0, long_side - h),
                    (0, long_side - w), (0, 0)),
                    mode='constant',
                    constant_values=pixel_mean)
    return image

def fix_ratio(image, pixel_mean=128.0, long_side=512):
    h, w, c = image.shape
    if h >= w:
        ratio = h * 1.0 / w
        h_ = long_side
        w_ = round(h_ / ratio)
    else:
        ratio = w * 1.0 / h
        w_ = long_side
        h_ = round(w_ / ratio)
    image = cv2.resize(image, dsize=(w_, h_), interpolation=cv2.INTER_LINEAR)
    image = border_pad(image, pixel_mean, long_side)
    return image

def transform(image, use_equalizeHist=True, gaussian_blur=3, pixel_mean=128.0, pixel_std=64.0, long_side=512):
    assert image.ndim == 2, "image must be gray image"
    if use_equalizeHist:
        image = cv2.equalizeHist(image)

    if gaussian_blur > 0:
        image = cv2.GaussianBlur(
            image,
            (gaussian_blur, gaussian_blur), 0)

    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    image = fix_ratio(image, pixel_mean, long_side)
    # augmentation for train or co_train

    # normalization
    image = image.astype(np.float32) - cfg.pixel_mean
    # vgg and resnet do not use pixel_std, densenet and inception use.
    if pixel_std:
        image /= pixel_std
    # normal image tensor :  H x W x C
    # torch image tensor :   C X H X W
    image = image.transpose((2, 0, 1))

    return image