import os
import cv2
import numpy as np

from utils.error_utils import SUCCESS
from utils.recognition_definitions import *


UPOL = 1
CASIA_1 = 2
MMU = 3
UBIRIS = 4

UPOL_STR = "UPOL"
CASIA_1_STR = "CASIA 1"
MMU_STR = "MMU"
UBIRIS_STR = "UBIRIS"

UPOL_PATH = "./databases/upol/"
CASIA_1_PATH = "./databases/casia1/"
MMU_PATH = "./databases/mmu/"
UBIRIS_PATH = "./databases/ubiris/"

IMAGES_PATH = "images/"
MASKS_PATH = "masks/"
CODES_PATH = "codes/"

STD_RADII = 32
STD_ANGLES = 128


gabor_prefix = "gab"
log_gabor_prefix = "log"
zern_circ_prefix = "zcp"
zern_annu_prefix = "zap"
fourier_prefix = "fou"

mask_prefix = "msk"
mask_ext = "npy"
code_ext = "npy"


def load_code(img_name, db_type, encoding_method, use_mask, alg):
    # getting database root path
    base_path = get_base_path(db_type)

    # if already computed, the load it from HD
    encoding_prefix = get_proper_prefix(encoding_method)
    img_without_ext = img_name[0:len(img_name) - 4]
    code_name = "%s_%s.%s" % (encoding_prefix, img_without_ext, code_ext)
    code_path = base_path + CODES_PATH + code_name

    code_mask_name = "%s_%s_%s.%s" % (encoding_prefix, img_without_ext, mask_prefix, code_ext)
    code_mask_path = base_path + CODES_PATH + code_mask_name

    # if code and it's mask exists, then load them
    if os.access(code_path, os.F_OK) and os.access(code_mask_path, os.F_OK):
        return np.load(code_path), np.load(code_mask_path)

    # ---------------------------------------------------------------------------

    # loading the image
    img_path = base_path + IMAGES_PATH + img_name
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    # creating proper mask
    if use_mask:
        mask_name = "%s.%s" % (img_without_ext, mask_ext)
        mask_path = base_path + MASKS_PATH + mask_name
        img_mask = np.load(mask_path)
    else:
        img_mask = np.ones(img.shape, np.uint8)

    #encoding image
    result, code, mask = alg.encode(img, img_mask)

    if result != SUCCESS:
        return None, None

    # saving computed code
    np.save(code_path, code)
    np.save(code_mask_path, mask)

    return code, mask


def get_base_path(db_type):
    if db_type == UPOL:
        return UPOL_PATH

    elif db_type == CASIA_1:
        return CASIA_1_PATH

    elif db_type == MMU:
        return MMU_PATH

    elif db_type == UBIRIS:
        return UBIRIS_PATH

    else:
        return None


def get_proper_prefix(encoding_method):
    if encoding_method == GABOR_FILTERS_ENCODING:
        return gabor_prefix

    elif encoding_method == LOG_GABOR_ENCODING:
        return log_gabor_prefix

    elif encoding_method == ZCP_ENCODING:
        return zern_circ_prefix

    elif encoding_method == ZAP_ENCODING:
        return zern_annu_prefix

    elif encoding_method == FOURIER_ENCODING:
        return fourier_prefix

    else:
        return None


def get_image_class(image_name, db_type):
    if db_type == UPOL:
        return image_name[0:4]      # four first characters

    elif db_type == CASIA_1:
        return image_name[0:3]      # five three characters

    elif db_type == MMU:
        return image_name[0:len(image_name) - 5]    # removing last 5 characters

    elif db_type == UBIRIS:
        return image_name[0: len(image_name) - 6]

    else:
        return None


def compute_far_percent(false_accepted, total):
    return float(false_accepted) / total * 100


def compute_frr_percent(false_rejected, total):
    return float(false_rejected) / total * 100


def compute_accuracy(accepted, total):
    return float(accepted) / total * 100


def compute_eer():
    pass