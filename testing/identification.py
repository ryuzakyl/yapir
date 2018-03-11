import os

from PyQt5 import QtCore

from utils.math_utils import DBL_MAX

from utils.testing_utils import *

from recognition.iris_recognition_algorithm import RecognitionAlgorithm


# performs 1-to-many comparisons
class IdentificationTest(QtCore.QThread):

    # type of database in which the test will be performed
    _db_type = None

    # type of iris encoding method to use
    _encoding_method = None

    # determines wether a mask is used or not in the encoding process
    _use_mask = None

    # flag that determines wether the thread must finish or not
    end_flag = 0

    # order for zernike polynomials
    polynomial_order = 16

    # internal epsilon for pupil
    eps_int = 0.50

    # signal throwed when the test has started (db_type, encoding_method)
    identification_started = QtCore.pyqtSignal(int, int)

    # signal throwed when an item is finished (curr_item, total_items, item_name, nearest_item_name, result)
    item_finished = QtCore.pyqtSignal(int, int, str, str, int)

    # signal throwed when the identification test is finished (accepted, total)
    identification_finished = QtCore.pyqtSignal(int, int)

    def __init__(self, parent):
        # calling base initializer
        super(IdentificationTest, self).__init__(parent)

    @property
    def db_type(self):
        return self._db_type

    @db_type.setter
    def db_type(self, value):
        self._db_type = value

    @property
    def encoding_method(self):
        return self._encoding_method

    @encoding_method.setter
    def encoding_method(self, value):
        self._encoding_method = value

    @property
    def use_mask(self):
        return self._use_mask

    @use_mask.setter
    def use_mask(self, value):
        self._use_mask = value

    def run(self):
        # emitting the identification started signal
        self.identification_started.emit(self.db_type, self.encoding_method)

        # reading image names from database
        db_path = get_base_path(self._db_type)
        images_path = db_path + IMAGES_PATH
        db_images = os.listdir(images_path)

        # obtaining total amount of test images of such database
        db_length = len(db_images)
        total = db_length

        # creating the recognition algorithm
        alg = RecognitionAlgorithm()
        alg.set_encoding_method(self._encoding_method)
        alg.set_polynomial_order(self.polynomial_order)
        alg.set_internal_epsilon(self.eps_int)

        # counters
        accepted = 0
        failed = 0
        for i in range(db_length):
            if self.end_flag:
                return

            # getting source image name
            src_img = db_images[i]

            # getting source image class or subject
            src_class = get_image_class(src_img, self._db_type)

            # encoding image
            code_1, mask_1 = load_code(src_img, self._db_type, self._encoding_method, self._use_mask, alg)

            # minimum distance
            best_distance = DBL_MAX
            best_index = -1
            for j in range(db_length):
                # test not valid for the same image
                if i == j:
                    continue

                # getting destination image name
                dst_img = db_images[j]

                # encoding image
                code_2, mask_2 = load_code(dst_img, self._db_type, self._encoding_method, self._use_mask, alg)

                # computing distance
                d = alg.get_distance(code_1, mask_1, code_2, mask_2)

                # storing best distance and corresponding index
                if d < best_distance:
                    best_distance = d
                    best_index = j

            # checking if the identification was successfull
            dst_class = get_image_class(db_images[best_index], self._db_type)

            # determining wether it was a success or not
            if src_class == dst_class:
                accepted += 1
                ok = 1
            else:
                failed += 1
                ok = 0

            # emitting the item finished signal
            self.item_finished.emit(i + 1, total, src_img, db_images[best_index], ok)

        # emitting the finished signal
        self.identification_finished.emit(accepted, total)