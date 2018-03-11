from _testcapi import DBL_MAX

import segmentation.projectiris_segmentation as proj_iris_segm
import segmentation.vasir_segmentation as vasir_segm

import normalization.projectiris_normalization as proj_iris_norm
import normalization.rubbersheet_normalization as rubb_sheet_norm

import encoding.projectiris_encoding as gab_filt_enc
import encoding.vasir_encoding as log_gab_filt_enc
import encoding.zcp_encoding as zcp_enc
import encoding.zap_encoding as zap_enc
import encoding.fourier_encoding as fou_enc

import matching.hamming_matching as hamm_match
import matching.lineal_algebra_matching as linalg_match

from utils.error_utils import *
from utils.iris_data_definitions import *
from utils.math_utils import fit_parabola_coords

from utils.recognition_definitions import *

# ----------------------------------------------------------------------------------


def generates_binary_template(encoding_method):
    return \
        encoding_method == GABOR_FILTERS_ENCODING or \
        encoding_method == LOG_GABOR_ENCODING or \
        encoding_method == FOURIER_ENCODING


def generates_vector_template(encoding_method):
    return \
        encoding_method == ZCP_ENCODING or \
        encoding_method == ZAP_ENCODING

# ----------------------------------------------------------------------------------


class RecognitionAlgorithm(object):

    # iris segmentation method
    segment_iris_method = None

    # iris segmentation function
    segment_iris_func = None

    #iris normalization method
    normalize_iris_method = None

    # iris normalization function
    normalize_iris_func = None

    # angular resolution in the normalization process
    angles = 180

    # radial resolution in the normalization process
    radii = 32

    # iris encoding method
    encode_iris_method = None

    # iris encoding function
    encode_iris_func = None

    # order of the zernike polynomials (annular or circular)
    polynomial_order = 16

    # internal epsilon (for annular polynomial)
    internal_eps = 0.25

    # external epsilon (for annular polynomial)
    external_eps = 1.0

    # template matching method
    template_matching_method = None

    # template matching function
    match_templates_func = None

    def __init__(self,
                 segmentation_method=PROJECT_IRIS_SEGMENTATION,
                 normalization_method=RUBBERSHEET_NORMALIZATION,
                 encoding_method=LOG_GABOR_ENCODING,
                 matching_method=HAMMING_DISTANCE
                 ):
        # calling parent initializer
        super(RecognitionAlgorithm, self).__init__()

        # setting segmentation method
        self.set_segmentation_method(segmentation_method)

        # setting normalization method, angular_resolution and radial_resolution
        self.set_normalization_method(normalization_method)

        # setting encoding method
        self.set_encoding_method(encoding_method)

        # setting matching method
        self.set_template_matching_method(matching_method)

    # ----------------------------------------------------------------------------

    def get_segmentation_method(self):
        return self.segment_iris_method

    def set_segmentation_method(self, method):
        if method == PROJECT_IRIS_SEGMENTATION:
            self.segment_iris_func = proj_iris_segm.segment_iris

        elif method == VASIR_SEGMENTATION:
            self.segment_iris_func = vasir_segm.segment_iris

        else:
            return

        self.segment_iris_method = method

    def get_normalization_method(self):
        return self.normalize_iris_method

    def set_normalization_method(self, method):
        if method == PROJECT_IRIS_NORMALIZATION:
            self.normalize_iris_func = proj_iris_norm.normalize_iris

        elif method == RUBBERSHEET_NORMALIZATION:
            self.normalize_iris_func = rubb_sheet_norm.normalize_iris

        else:
            return

        self.normalize_iris_method = method

    def get_encoding_method(self):
        return self.encode_iris_method

    def set_encoding_method(self, method):
        if method == GABOR_FILTERS_ENCODING:
            self.encode_iris_func = gab_filt_enc.encode_iris

        elif method == LOG_GABOR_ENCODING:
            self.encode_iris_func = log_gab_filt_enc.encode_iris

        elif method == ZCP_ENCODING:
            self.encode_iris_func = zcp_enc.encode_iris

        elif method == ZAP_ENCODING:
            self.encode_iris_func = zap_enc.encode_iris

        elif method == FOURIER_ENCODING:
            self.encode_iris_func = fou_enc.encode_iris

        else:
            return

        self.encode_iris_method = method

        # setting template matching method
        if generates_binary_template(method):
            self.match_templates_func = hamm_match.hamming_distance

        elif generates_vector_template(method):
            self.match_templates_func = linalg_match.euclidean_distance

    def get_template_matching_method(self):
        return self.template_matching_method

    def set_template_matching_method(self, method):
        if method == HAMMING_DISTANCE and generates_binary_template(self.encode_iris_method):
            self.match_templates_func = hamm_match.hamming_distance

        elif method == EUCLIDEAN_DISTANCE and generates_vector_template(self.encode_iris_method):
            self.match_templates_func = linalg_match.euclidean_distance

        else:
            return

    def get_angular_resolution(self):
        return self.angles

    def set_angular_resolution(self, angular_resolution):
        if MIN_ANGULAR_RESOLUTION <= angular_resolution <= MAX_ANGULAR_RESOLUTION:
            self.angles = angular_resolution

    def get_radial_resolution(self, ):
        return self.radii

    def set_radial_resolution(self, radial_resolution):
        if MIN_RADIAL_RESOLUTION <= radial_resolution <= MAX_RADIAL_RESOLUTION:
            self.radii = radial_resolution

    def get_polynomial_order(self):
        return self.polynomial_order

    def set_polynomial_order(self, order):
        if order > 0:
            self.polynomial_order = order

    def get_internal_epsilon(self):
        return self.internal_eps

    def set_internal_epsilon(self, eps_int):
        if 0 <= eps_int <= 1.0 and eps_int <= self.external_eps:
            self.internal_eps = eps_int

    def get_external_epsilon(self):
        return self.external_eps

    def set_external_epsilon(self, eps_ext):
        if 0 <= eps_ext <= 1.0 and eps_ext >= self.internal_eps:
            self.external_eps = eps_ext

    # ----------------------------------------------------------------------------

    def match(self, original, query):
        # getting the code for the original image
        result, code_1, mask_1 = self.get_template(original)

        # if there was an error of some kind
        if result != SUCCESS:
            return result, None

        # getting the code for the query image
        result, code_2, mask_2 = self.get_template(query)

        # if there was an error of some kind
        if result != SUCCESS:
            return result, None

        # getting the distance between the templates
        d = self.get_distance(code_1, mask_1, code_2, mask_2)

        # True if distance is lower than threshold, False otherwise
        return SUCCESS, d

    def get_distance(self, code_1, mask_1, code_2, mask_2):
        if generates_binary_template(self.encode_iris_method):
            return self.match_templates_func(code_1, mask_1, code_2, mask_2)

        elif generates_vector_template(self.encode_iris_method):
            return self.match_templates_func(code_1, code_2)

        else:
            return DBL_MAX

    def get_template(self, eye_img):
        # ---------------segmenting iris---------------
        result, data = self.segment_iris_func(eye_img)

        # if there was a segmentation error
        if result != SUCCESS:
            return result, None, None

        # getting segmentation data
        pupil_data = data[PUPIL_DATA]
        iris_data = data[IRIS_DATA]
        eyelids_data = data[EYELIDS_DATA]

        pupil_center = pupil_data[CENTER]
        pupil_radius = pupil_data[RADIUS]

        iris_center = iris_data[CENTER]
        iris_radius = iris_data[RADIUS]

        #ToDo: Take eyelids in consideration
        upper_coeff = None
        lower_coeff = None
        # # getting eyelids data
        # upper_eyelid_data, lower_eyelid_data = eyelids_data
        # p1, p2, p3 = upper_eyelid_data
        # p4, p5, p6 = lower_eyelid_data
        # upper_coeff = fit_parabola_coords(p1[X], p1[Y], p2[X], p2[Y], p3[X], p3[Y])
        # lower_coeff = fit_parabola_coords(p4[X], p4[Y], p5[X], p5[Y], p6[X], p6[Y])

        # ---------------normalize iris---------------

        angles = self.angles
        radii = self.radii

        result, norm_image, mask_image = self.normalize_iris_func(eye_img, angles, radii, pupil_center, pupil_radius, iris_center, iris_radius, upper_coeff, lower_coeff)

        # if there was a normalization error
        if result != SUCCESS:
            return result, None, None

        # ---------------encode iris---------------

        return self.__encode(norm_image, mask_image, angles, radii)

    def encode(self, norm_imag, norm_mask):
        if norm_imag.shape != norm_mask.shape:
            return None, None

        radii, angles = norm_imag.shape
        return self.__encode(norm_imag, norm_mask, angles, radii)

    def __encode(self, norm_image, mask_image, angles, radii):
        if self.encode_iris_method == GABOR_FILTERS_ENCODING:
            return self.encode_iris_func(norm_image, mask_image, angles, radii)

        elif self.encode_iris_method == LOG_GABOR_ENCODING:
            return self.encode_iris_func(norm_image, mask_image)

        elif self.encode_iris_method == ZCP_ENCODING:
            # getting polynomial order
            order = self.polynomial_order
            return self.encode_iris_func(norm_image, order)

        elif self.encode_iris_method == ZAP_ENCODING:
            # getting polynomial data
            order = self.polynomial_order
            eps_lb = self.internal_eps
            eps_ub = self.external_eps
            return self.encode_iris_func(norm_image, mask_image, order, eps_lb, eps_ub)

        elif self.encode_iris_method == FOURIER_ENCODING:
            return self.encode_iris_func(norm_image, mask_image, angles, radii)

        else:
            return UNKNOWN_ENCODING_METHOD, None, None
