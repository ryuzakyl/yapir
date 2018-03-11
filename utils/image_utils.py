import numpy as np


# determines if a pixel is valid in certain image
def valid_pixel(arr, x, y):
    height, width = arr.shape
    return 0 <= x < width and 0 <= y < height


#ToDo: Python code here. Optimize it with cython or anything like that.
# masks an grayscale image
def mask_image(img, mask):
    # both must be valid arrays
    if img is None or mask is None:
        return None

    img_height, img_width = img.shape
    mask_height, mask_width = mask.shape

    # must be of same size
    if img_height != mask_height or img_width != mask_width:
        return None

    # creating masked image array
    masked_image = np.empty(img.shape, np.uint8)

    for i in range(img_height):
        for j in range(img_width):
            # if pixel is ok
            if mask[i, j]:
                pixel_value = img[i, j]
            else:
                pixel_value = 0 if i % 2 == 0 or j % 2 == 0 else 200

            masked_image[i, j] = pixel_value

    return masked_image