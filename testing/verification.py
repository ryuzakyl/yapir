import os

from PyQt5 import QtCore

from utils.testing_utils import *

from recognition.iris_recognition_algorithm import RecognitionAlgorithm


# performs 1-to-1 comparisons
class VerificationTest(QtCore.QThread):
    # type of database in which the test will be performed
    _db_type = None

    # type of iris encoding method to use
    _encoding_method = None

    # determines wether a mask is used or not in the encoding process
    _use_mask = None

    # threshold of the verification test
    _thres = 0.0

    # flag that determines wether the thread must finish or not
    end_flag = 0

    # order for zernike polynomials
    polynomial_order = 16

    # internal epsilon for pupil
    eps_int = 0.50

    # signal throwed when the test has started (db_type, encoding_method, threshold as string)
    verification_started = QtCore.pyqtSignal(int, int, str)

    # signal throwed when an item is finished (curr_item, total_items, item_name, nearest_item_name, result)
    comparison_finished = QtCore.pyqtSignal(int, int, str, str, int, str)

    # signal throwed when the verification test is finished (fa, fr, accepted, total)
    verification_finished = QtCore.pyqtSignal(int, int, int, int)

    def __init__(self, parent):
        # calling base initializer
        super(VerificationTest, self).__init__(parent)

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

    @property
    def threshold(self):
        return self._thres

    @threshold.setter
    def threshold(self, value):
        self._thres = value

    def run(self):
        # emitting the verification started signal
        self.verification_started.emit(self.db_type, self.encoding_method, str(self._thres))

        # reading image names from database
        db_path = get_base_path(self._db_type)
        images_path = db_path + IMAGES_PATH
        db_images = os.listdir(images_path)

        # obtaining total amount of test images of such database
        db_length = len(db_images)
        total = db_length * (db_length - 1) // 2

        # creating the recognition algorithm
        alg = RecognitionAlgorithm()
        alg.set_encoding_method(self._encoding_method)
        alg.set_polynomial_order(self.polynomial_order)
        alg.set_internal_epsilon(self.eps_int)

        cont = 0
        fa = 0
        fr = 0
        accepted = 0
        for i in range(0, db_length - 1):
            # getting source image name
            src_img = db_images[i]

            # getting source image class or subject
            src_class = get_image_class(src_img, self._db_type)

            # encoding image
            code_1, mask_1 = load_code(src_img, self._db_type, self._encoding_method, self._use_mask, alg)

            for j in range(i + 1, db_length):
                if self.end_flag:
                    return

                # incrementing comparison counter
                cont += 1

                # getting destination image name
                dst_img = db_images[j]

                # getting source image class or subject
                dst_class = get_image_class(dst_img, self._db_type)

                # encoding image
                code_2, mask_2 = load_code(dst_img, self._db_type, self._encoding_method, self._use_mask, alg)

                # computing distance
                d = alg.get_distance(code_1, mask_1, code_2, mask_2)

                # computing answer
                match = 1 if src_class == dst_class else 0

                # FA (False Accept)
                if d < self._thres and not match:
                    fa += 1
                    ok = 0

                # FR (False Reject)
                elif d >= self._thres and match:
                    fr += 1
                    ok = 0

                # no error
                else:
                    accepted += 1
                    ok = 1

                # emitting the item finished signal
                self.comparison_finished.emit(cont, total, src_img, dst_img, ok, str(round(d, 3)))

        #  emitting the finished signal
        self.verification_finished.emit(fa, fr, accepted, total)