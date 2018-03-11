import numpy as np


# hamming distance with masks
def hamming_distance(bit_code_x, mask_x, bit_code_y, mask_y):
    # getting code size (assuming all have the same size)
    code_size = len(bit_code_x)

    # performing logical xor between the two iris codes
    xor_result = np.bitwise_xor(bit_code_x, bit_code_y)

    # performing logical and between the two masks
    and_result = np.bitwise_and(mask_x, mask_y)

    # performing logical and between the two previous results
    result = np.bitwise_and(xor_result, and_result)

    # counting disagreeing bits and normalizing
    return np.count_nonzero(result) / float(code_size)