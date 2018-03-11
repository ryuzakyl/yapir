
from PyQt5 import QtCore, QtGui, QtWidgets

from ui.airs_window import *

import cv2

from utils.math_utils import euclidean_distance_points, fit_parabola_coords
from utils.image_utils import mask_image
from utils.error_utils import *
from utils.recognition_definitions import *
from utils.testing_utils import *

#segmentation modules
import segmentation.projectiris_segmentation as proj_iris_seg
import segmentation.vasir_segmentation as vasir_seg

# normalization modules
import normalization.projectiris_normalization as proj_iris_norm
import normalization.rubbersheet_normalization as rubb_sheet_norm

# encoding modules
import encoding.projectiris_encoding as proj_iris_enc
import encoding.vasir_encoding as vasir_enc
import encoding.fourier_encoding as fou_enc
import encoding.zcp_encoding as zcp_enc
import encoding.zap_encoding as zap_enc

# iris recognition algorithm
from recognition.iris_recognition_algorithm import *

# database testing methods
from testing.verification import VerificationTest
from testing.identification import IdentificationTest

import time
from threading import Timer

#-----------------------------------------------------------------------------

TEMPLATE_WIDTH = 256
TEMPLATE_HEIGHT = 64

#-----------------------------------------------------------------------------

CHECKED = 2
UNCHECKED = 0

#-----------------------------------------------------------------------------


class AirsWindow(QtWidgets.QMainWindow, Ui_airsWindow):

    def __init__(self, parent=None):

        #calling base initializers
        super(AirsWindow, self).__init__(parent)

        #calling base setupUi for ui configuration
        self.setupUi(self)

        #setting up the gui
        self.setup_gui()

        # normalized iris image
        self.normalized_iris_image = None

        # mask of the normalized iris image
        self.normalized_iris_mask = None

        # original heatmap image
        self.heatmap_image = None

        # identification test worker class
        self.ident_test_worker = None

        # verification test worker class
        self.verif_test_worker = None

    #---------------------------GUI configuration----------------------------

    def setup_gui(self):
        self.irisViewer.setFixedSize(QtCore.QSize(300, 200))
        self.normIrisImageViewer.setFixedSize(QtCore.QSize(256, 10))
        self.heatmapImageViewer.setFixedSize(QtCore.QSize(256, 10))

        # hiding the encode button
        self.encodeButton.setVisible(False)

    #-------------------------------Menu Slots--------------------------------

    def onActionAboutTriggered(self):
        QtWidgets.QMessageBox.about(self,
            "About Automated Iris Recognition System",
            "<h2>Automated Iris Recognition System 0.1</h2>"
            "<p>An application that allows biometric authentication using iris patterns of an individual's eyes.</p>"
            "Copyright &copy; 2013 <br/>"
            "<a href='v.mendiola@lab.matcom.uh.cu'>Victor Manuel Mendiola Lau</a>."
            "<br/>Student at <a href='http://www.uh.cu'>Havana's University</a>"
            "<br/>Under the supervision of <a href='mailto:fjsilva@cenatav.co.cu'>Francisco J. Silva</a> and \
            <a href='mailto:dpmunoz@cenatav.co.cu'>Dania Porro</a>.")

    def onActionQuitTriggered(self):
        self.close()

    def onActionHelpTriggered(self):
        QtWidgets.QMessageBox.about(self, "Help", "<h2>Still no help available.</h2>")

    def onActionConfigurationTriggered(self):
        QtWidgets.QMessageBox.about(self,
            "Configuration",
            "<h2>Configuration not available yet.</h2>"
            "Iris recognition internal parameters configuration here.\n"
            "(db specific parameters-min_radius, etc.-, etc.)")

    #-------------------------Iris Processing Tab Slots--------------------------

    def onLoadIrisButtonClicked(self):
        #load the iris image
        if not self.irisViewer.loadImage():
            QtWidgets.QMessageBox.information(self, "Information", "<h3>%s</h3>" % error_messages[LOAD_IMAGE_ERROR])
            return

        #changing the name label
        new_name = "Iris image - %s" % self.irisViewer.getImageName()
        self.irisImageLabel.setText(new_name)

        #resetting viewers to initial conditions
        self.normIrisImageViewer.unloadImage()
        self.normIrisImageViewer.setFixedSize(QtCore.QSize(256, 10))

        self.heatmapImageViewer.unloadImage()
        self.heatmapImageViewer.setFixedSize(QtCore.QSize(256, 10))

    def onIrisViewerReset(self):
        #disabling all the spinboxes in Segmentation tab
        self.pupilCenterAbscissaSpinBox.setEnabled(False)
        self.pupilCenterOrdinateSpinBox.setEnabled(False)
        self.pupilBorderAbscissaSpinBox.setEnabled(False)
        self.pupilBorderOrdinateSpinBox.setEnabled(False)

        self.irisCenterAbscissaSpinBox.setEnabled(False)
        self.irisCenterOrdinateSpinBox.setEnabled(False)
        self.irisBorderAbscissaSpinBox.setEnabled(False)
        self.irisBorderOrdinateSpinBox.setEnabled(False)

    #-------------------------Segmentation Tab Slots--------------------------

    def onPupilCenterAbscissaSpinBoxValueChanged(self):
        pupil_center = self.irisViewer.getPupilCenter()
        new_x = self.pupilCenterAbscissaSpinBox.value()
        new_center = QtCore.QPointF(new_x, pupil_center.y())
        self.irisViewer.changePupilCenter(new_center)

    def onPupilCenterOrdinateSpinBoxValueChanged(self):
        pupil_center = self.irisViewer.getPupilCenter()
        new_y = self.pupilCenterOrdinateSpinBox.value()
        new_center = QtCore.QPointF(pupil_center.x(), new_y)
        self.irisViewer.changePupilCenter(new_center)

    def onPupilBorderAbscissaSpinBoxValueChanged(self):
        pupil_border = self.irisViewer.getPupilBorder()
        new_x = self.pupilBorderAbscissaSpinBox.value()
        new_border = QtCore.QPointF(new_x, pupil_border.y())
        self.irisViewer.changePupilBorder(new_border)

    def onPupilBorderOrdinateSpinBoxValueChanged(self):
        pupil_border = self.irisViewer.getPupilBorder()
        new_y = self.pupilBorderOrdinateSpinBox.value()
        new_border = QtCore.QPointF(pupil_border.x(), new_y)
        self.irisViewer.changePupilBorder(new_border)

    def onIrisCenterAbscissaSpinBoxValueChanged(self):
        iris_center = self.irisViewer.getIrisCenter()
        new_x = self.irisCenterAbscissaSpinBox.value()
        new_center = QtCore.QPointF(new_x, iris_center.y())
        self.irisViewer.changeIrisCenter(new_center)

    def onIrisCenterOrdinateSpinBoxValueChanged(self):
        iris_center = self.irisViewer.getIrisCenter()
        new_y = self.irisCenterOrdinateSpinBox.value()
        new_center = QtCore.QPointF(iris_center.x(), new_y)
        self.irisViewer.changeIrisCenter(new_center)

    def onIrisBorderAbscissaSpinBoxValueChanged(self):
        iris_border = self.irisViewer.getIrisBorder()
        new_x = self.irisBorderAbscissaSpinBox.value()
        new_border = QtCore.QPointF(new_x, iris_border.y())
        self.irisViewer.changeIrisBorder(new_border)

    def onIrisBorderOrdinateSpinBoxValueChanged(self):
        iris_border = self.irisViewer.getIrisBorder()
        new_y = self.irisBorderOrdinateSpinBox.value()
        new_border = QtCore.QPointF(iris_border.x(), new_y)
        self.irisViewer.changeIrisBorder(new_border)

    #-------------------------IrisViewer Slots--------------------------

    def onPupilDetected(self):
        # self.outputTextEdit.append("Pupil detected")
        pass

    def onIrisDetected(self):
        # self.outputTextEdit.append("Iris detected")
        pass

    def onUpperEyelidDetected(self):
        # self.outputTextEdit.append("Upper eyelid detected")
        pass

    def onLowerEyelidDetected(self):
        # self.outputTextEdit.append("Lower eyelid detected")
        pass

    def onIrisSegmented(self):
        # self.outputTextEdit.append("Iris completely segmented")
        pass

    def onPupilCenterChanged(self):
        pupil_center = self.irisViewer.getPupilCenter()
        cx = pupil_center.x()
        cy = pupil_center.y()

        self.pupilCenterAbscissaSpinBox.setValue(cx)
        self.pupilCenterOrdinateSpinBox.setValue(cy)

        #if the pupil border is already set
        if self.irisViewer.isPupilDetected():
            pupil_border = self.irisViewer.getPupilBorder()
            pupil_radius = int(euclidean_distance_points(pupil_center, pupil_border))
            self.pupilRadiusSpinBox.setValue(pupil_radius)

    def onPupilBorderChanged(self):
        pupil_border = self.irisViewer.getPupilBorder()
        bx = pupil_border.x()
        by = pupil_border.y()

        self.pupilBorderAbscissaSpinBox.setValue(bx)
        self.pupilBorderOrdinateSpinBox.setValue(by)

        pupil_center = self.irisViewer.getPupilCenter()
        pupil_radius = int(euclidean_distance_points(pupil_center, pupil_border))
        self.pupilRadiusSpinBox.setValue(pupil_radius)

    def onIrisCenterChanged(self):
        iris_center = self.irisViewer.getIrisCenter()
        ix = iris_center.x()
        iy = iris_center.y()

        self.irisCenterAbscissaSpinBox.setValue(ix)
        self.irisCenterOrdinateSpinBox.setValue(iy)

        #if the iris border is already set
        if self.irisViewer.isIrisDetected():
            iris_border = self.irisViewer.getIrisBorder()
            iris_radius = int(euclidean_distance_points(iris_center, iris_border))
            self.irisRadiusSpinBox.setValue(iris_radius)

    def onIrisBorderChanged(self):
        iris_border = self.irisViewer.getIrisBorder()
        ix = iris_border.x()
        iy = iris_border.y()

        self.irisBorderAbscissaSpinBox.setValue(ix)
        self.irisBorderOrdinateSpinBox.setValue(iy)

        iris_center = self.irisViewer.getIrisCenter()
        iris_radius = int(euclidean_distance_points(iris_center, iris_border))
        self.irisRadiusSpinBox.setValue(iris_radius)

    def onPupilCenterSet(self):
        self.pupilCenterAbscissaSpinBox.setEnabled(True)
        self.pupilCenterOrdinateSpinBox.setEnabled(True)

    def onPupilBorderSet(self):
        self.pupilBorderAbscissaSpinBox.setEnabled(True)
        self.pupilBorderOrdinateSpinBox.setEnabled(True)

    def onIrisCenterSet(self):
        self.irisCenterAbscissaSpinBox.setEnabled(True)
        self.irisCenterOrdinateSpinBox.setEnabled(True)

    def onIrisBorderSet(self):
        self.irisBorderAbscissaSpinBox.setEnabled(True)
        self.irisBorderOrdinateSpinBox.setEnabled(True)

    #--------------------------Normalization Tab Slots---------------------------

    def onTemplateWidthSpinBoxValueChanged(self):
        self.updateNormalizedImage()

    def onTemplateHeightSpinBoxValueChanged(self):
        self.updateNormalizedImage()

    def onShowMaskStateChanged(self, state):
        self.setImageMask(self.normalized_iris_image, self.normalized_iris_mask)

    #---------------------------Match Tab Slots---------------------------

    def onOriginalLoadIrisButtonClicked(self):
        #load the iris image
        if not self.origIrisImageViewer.loadImage():
            QtWidgets.QMessageBox.information(self, "Information", "<h3>%s</h3>" % error_messages[LOAD_IMAGE_ERROR])
            return

        #changing the name label
        new_name = "Original Iris - %s" % self.origIrisImageViewer.getImageName()
        self.originalIrisLabel.setText(new_name)

    def onQueryLoadIrisButtonClicked(self):
        #load the iris image
        if not self.queryIrisImageViewer.loadImage():
            QtWidgets.QMessageBox.information(self, "Information", "<h3>%s</h3>" % error_messages[LOAD_IMAGE_ERROR])
            return

        #changing the name label
        new_name = "Query Iris - %s" % self.queryIrisImageViewer.getImageName()
        self.queryIrisLabel.setText(new_name)

    def onGoButtonClicked(self):
        original_eye = self.origIrisImageViewer.getImageData()
        query_eye = self.queryIrisImageViewer.getImageData()

        if original_eye is None or query_eye is None:
            QtWidgets.QMessageBox.information(self, "Information", "<h2>%s</h2>" % error_messages[INVALID_IMAGE])
            return

        # creating an iris recognition algorithm
        rec_alg = RecognitionAlgorithm()

        segm_method = self.getSelectedSegmentationMethod()
        rec_alg.set_segmentation_method(segm_method)

        norm_method = self.getSelectedNormalizationMethod()
        rec_alg.set_normalization_method(norm_method)

        enc_method = self.getSelectedEncodingMethod()
        rec_alg.set_encoding_method(enc_method)

        dist_method = self.getAdequateMatchingMethod()
        rec_alg.set_template_matching_method(dist_method)

        # selecting threshold
        thres = self.matchThresholdSpinBox.value()

        # performing matching
        result, d = rec_alg.match(original_eye, query_eye)

        if result != SUCCESS:
            QtWidgets.QMessageBox.about(self, "Error", "<h2>%s</h2>" % error_messages[result])
        else:
            if d < thres:
                self.resultLabel.setText("Match (%.2f)" % d)
                self.resultLabel.setStyleSheet("QLabel {background-color: #33CC00}")
            else:
                self.resultLabel.setText("No Match (%.2f)" % d)
                self.resultLabel.setStyleSheet("QLabel {background-color: #FF3300}")

        # resetting the state of the result label in 1.5 sec
        reset_timer = Timer(1.5, self.resetResultLabel)
        reset_timer.start()

    #---------------------------Test Tab Slots----------------------------

    def onRunTestButtonClicked(self):
        if self.verificationRadioButton.isChecked():
            # setting the verification test
            self.configure_verification_test()

            # running the identification test
            if self.verif_test_worker is not None:
                self.verif_test_worker.quit()
                self.verif_test_worker.start()

                # enabling the stop button
                self.runButton.setEnabled(False)
                self.stopButton.setEnabled(True)
            else:
                QtWidgets.QMessageBox.about(self, "Error", "<h3>There was an error while setting the verification test.</h3>")

        elif self.identificationRadioButton.isChecked():
        # setting the identification test
            self.configure_identification_test()

            # running the identification test
            if self.ident_test_worker is not None:
                self.ident_test_worker.quit()
                self.ident_test_worker.start()

                # enabling the stop button
                self.runButton.setEnabled(False)
                self.stopButton.setEnabled(True)
            else:
                QtWidgets.QMessageBox.about(self, "Error", "<h3>There was an error while setting the identification test.</h3>")

        else:
            QtWidgets.QMessageBox.about(self, "Error", "<h3>Unknown test.</h3>")

    def onStopTestButtonClicked(self):
        if self.verificationRadioButton.isChecked():
            if self.verif_test_worker is not None:
                # setting termination flag
                self.verif_test_worker.end_flag = 1

                # for synchronous termination
                self.verif_test_worker.wait()

        elif self.identificationRadioButton.isChecked():
            if self.ident_test_worker is not None:
                # setting termination flag
                self.ident_test_worker.end_flag = 1

                # for synchronous termination
                self.ident_test_worker.wait()

        # enabling the run button
        self.runButton.setEnabled(True)
        self.stopButton.setEnabled(False)

    def onZapRadioButtonToggled(self, checked):
        if checked:
            self.kComboBox.setEnabled(True)
        else:
            self.kComboBox.setEnabled(False)

    #---------------------------Button's Slots----------------------------

    def onSegmentButtonClicked(self):
        #getting the image as numpy array
        eye_img = self.irisViewer.getImageData()

        start = time.time()
        result, data = self.segmentIris(eye_img)
        end = time.time() - start

        if result != SUCCESS:
            QtWidgets.QMessageBox.about(self, "Error", "<h2>%s</h2>" % error_messages[result])
        else:
            self.outputTextEdit.append("segmentation time = %i ms" % int(end * 1000))

            # updating the iris viewer
            self.updateIrisViewer(result, data)

    def onNormalizeButtonClicked(self):
        #getting the image as numpy array
        eye_img = self.irisViewer.getImageData()

        start = time.time()
        result, norm_iris_data, norm_iris_mask = self.normalizeIris(eye_img)
        end = time.time() - start

        if result != SUCCESS:
            QtWidgets.QMessageBox.about(self, "Error", "<h2>%s</h2>" % error_messages[result])
        else:
            self.outputTextEdit.append("normalization time = %i ms" % int(end * 1000))

            # setting the image in the normalized image viewer
            self.setNormalizedImage(result, norm_iris_data, norm_iris_mask)

    def onComputeHeatmapButtonClicked(self):
        start = time.time()
        # computing heatmap
        result, heatmap = self.generateHeatmap()
        end = time.time() - start

        if result != SUCCESS:
            QtWidgets.QMessageBox.about(self, "Error", "<h2>%s</h2>" % error_messages[result])
        else:
            # saving the heatmap of iris image
            self.heatmap_image = heatmap

            # setting the image data
            self.heatmapImageViewer.setImageData(heatmap)

            self.outputTextEdit.append("heatmap time = %i ms" % int(end * 1000))

    def onEncodeButtonClicked(self):
        start = time.time()
        result, bit_code, bit_code_mask = self.encodeIris()
        end = time.time() - start

        if result != SUCCESS:
            QtWidgets.QMessageBox.about(self, "Error", "<h2>%s</h2>" % error_messages[result])
        else:
            self.outputTextEdit.append("encoding time = %i ms" % int(end * 1000))

    #-----------------------------Methods------------------------------

    def setImageMask(self, norm_iris_data, norm_iris_mask):
        if norm_iris_data is None or norm_iris_mask is None:
            return

        # setting the state
        state = CHECKED if self.showMaskCheckBox.isChecked() else UNCHECKED

        masked = None

        #if the user checked the checkbox
        if state == CHECKED:
            masked = mask_image(norm_iris_data, norm_iris_mask)

        elif state == UNCHECKED:
            masked = norm_iris_data

        # resizing to standard size
        standard_size = (TEMPLATE_WIDTH, TEMPLATE_HEIGHT)
        show_img = cv2.resize(masked, standard_size, interpolation=cv2.INTER_LINEAR)

        # setting the normalized iris image
        self.normIrisImageViewer.setImageData(show_img)

    def segmentIris(self, eye_img):
        #if eye image is not valid
        if eye_img is None:
            return INVALID_IMAGE, None

        #segmenting with the project iris segmentation method
        if self.projIrisRadioButton.isChecked():
            return proj_iris_seg.segment_iris(eye_img)

        #segmenting with the vasir segmentation method
        elif self.vasirRadioButton.isChecked():
            # return NOT_IMPLEMENTED_FEATURE, None
            return vasir_seg.segment_iris(eye_img)

    def updateIrisViewer(self, result, data):
        if result != SUCCESS:
            QtWidgets.QMessageBox.about(self, "Error", "<h2>%s</h2>" % error_messages[result])
        else:
            #resetting the viewer
            self.irisViewer.reset()

            #drawing control points in the iris image viewer
            pupil_data = data[PUPIL_DATA]
            iris_data = data[IRIS_DATA]
            eyelids_data = data[EYELIDS_DATA]

            #setting pupil control points
            xp, yp = pupil_data[CENTER]
            rp = pupil_data[RADIUS]
            p_center = QtCore.QPointF(xp, yp)
            p_border = QtCore.QPointF(xp + rp, yp)
            self.irisViewer.setPupilCenter(p_center)
            self.irisViewer.setPupilBorder(p_border)

            #setting iris control points
            xi, yi = iris_data[CENTER]
            ri = iris_data[RADIUS]
            i_center = QtCore.QPointF(xi, yi)
            i_border = QtCore.QPointF(xi + ri, yi)
            self.irisViewer.setIrisCenter(i_center)
            self.irisViewer.setIrisBorder(i_border)

            #setting eyelids
            upper_eyelid_data, lower_eyelid_data = eyelids_data
            p1_data, p2_data, p3_data = upper_eyelid_data
            p1 = QtCore.QPointF(p1_data[0], p1_data[1])
            p2 = QtCore.QPointF(p2_data[0], p2_data[1])
            p3 = QtCore.QPointF(p3_data[0], p3_data[1])

            p4_data, p5_data, p6_data = lower_eyelid_data
            p4 = QtCore.QPointF(p4_data[0], p4_data[1])
            p5 = QtCore.QPointF(p5_data[0], p5_data[1])
            p6 = QtCore.QPointF(p6_data[0], p6_data[1])

            self.irisViewer.setUpperEyelid(p1, p2, p3)
            self.irisViewer.setLowerEyelid(p4, p5, p6)

    def normalizeIris(self, eye_img):
        #getting rectangle dimensions
        template_width = self.templateWidthSpinBox.value()
        template_height = self.templateHeightSpinBox.value()

        width = TEMPLATE_WIDTH if template_width <= 1 else template_width
        height = TEMPLATE_HEIGHT if template_height <= 1 else template_height

        if eye_img is None:
            return INVALID_IMAGE, None, None

        if not self.irisViewer.isIrisDetected():
            return SEGMENTATION_REQUIRED, None, None

        #getting pupil data
        p_center = self.irisViewer.getPupilCenter()
        p_border = self.irisViewer.getPupilBorder()
        pupil_center = (p_center.x(), p_center.y())
        pupil_radius = euclidean_distance_points(p_center, p_border)

        #getting iris data
        i_center = self.irisViewer.getIrisCenter()
        i_border = self.irisViewer.getIrisBorder()
        iris_center = (i_center.x(), i_center.y())
        iris_radius = euclidean_distance_points(i_center, i_border)

        #getting the eye image height
        img_height = eye_img.shape[0]

        #getting eyelids data
        if self.irisViewer.isUpperEyelidDetected():
            p1, p2, p3 = self.irisViewer.getUpperEyelid()
            x1 = p1.x()
            y1 = img_height - p1.y()

            x2 = p2.x()
            y2 = img_height - p2.y()

            x3 = p3.x()
            y3 = img_height - p3.y()

            upper_eyelid = fit_parabola_coords(x1, y1, x2, y2, x3, y3)
        else:
            upper_eyelid = None

        if self.irisViewer.isLowerEyelidDetected():
            p4, p5, p6 = self.irisViewer.getLowerEyelid()
            x4 = p4.x()
            y4 = img_height - p4.y()

            x5 = p5.x()
            y5 = img_height - p5.y()

            x6 = p6.x()
            y6 = img_height - p6.y()

            lower_eyelid = fit_parabola_coords(x4, y4, x5, y5, x6, y6)
        else:
            lower_eyelid = None

        #performing concentric pupil and iris normalization
        if self.concentricRadioButton.isChecked():
            return proj_iris_norm.normalize_iris(eye_img, width, height, pupil_center, pupil_radius, None, iris_radius, upper_eyelid, lower_eyelid)

        #performing classical rubbersheet model normalization
        elif self.nonconcentricRadioButton.isChecked():
            return rubb_sheet_norm.normalize_iris(eye_img, width, height, pupil_center, pupil_radius, iris_center, iris_radius, upper_eyelid, lower_eyelid)

    def setNormalizedImage(self, result, norm_iris_data, norm_iris_mask):
        if result != SUCCESS:
            QtWidgets.QMessageBox.information(self, "Error", "<h2>%s</h2>" % error_messages[result])
        else:
            # saving normalized iris image and it's mask
            self.normalized_iris_image = norm_iris_data
            self.normalized_iris_mask = norm_iris_mask

            # setting the state
            state = CHECKED if self.showMaskCheckBox.isChecked() else UNCHECKED

            #if the user checked the checkbox
            if state == CHECKED:
                masked = mask_image(norm_iris_data, norm_iris_mask)

                # resizing to standard size
                standard_size = (TEMPLATE_WIDTH, TEMPLATE_HEIGHT)
                show_img = cv2.resize(masked, standard_size, interpolation=cv2.INTER_LINEAR)

                # setting the normalized iris image
                self.normIrisImageViewer.setImageData(show_img)

            #if the user unchecked the checkbox
            elif state == UNCHECKED:
                # resizing to standard size
                standard_size = (TEMPLATE_WIDTH, TEMPLATE_HEIGHT)
                show_img = cv2.resize(norm_iris_data, standard_size, interpolation=cv2.INTER_LINEAR)

                # setting the normalized iris image
                self.normIrisImageViewer.setImageData(show_img)

    def updateNormalizedImage(self):
        # iris image must be segmented
        if not self.irisViewer.isIrisSegmented():
            return

        # normalization should have been performed in the past
        if self.normIrisImageViewer.getImage() is None:
            return

        # normalizing the iris
        result, norm_iris_data, norm_iris_mask = self.normalizeIris(self.irisViewer.getImageData())

        # setting the image in the normalized image viewer
        self.setNormalizedImage(result, norm_iris_data, norm_iris_mask)

    def generateHeatmap(self):
        norm_img = self.normalized_iris_image

        # image must be valid
        if norm_img is None:
            return INVALID_IMAGE, None

        # resizing image to standard size
        standard_size = (TEMPLATE_WIDTH, TEMPLATE_HEIGHT)
        norm_img = cv2.resize(norm_img, standard_size, interpolation=cv2.INTER_LINEAR)

        if self.daugmanRadioButton.isChecked():
            return SUCCESS, proj_iris_enc.generate_heatmap(norm_img)

        elif self.masekRadioButton.isChecked():
            return SUCCESS, vasir_enc.generate_heatmap(norm_img)

        elif self.zernikeCircularRadioButton.isChecked():
            return NOT_IMPLEMENTED_FEATURE, None

        elif self.zernikeAnnularRadioButton.isChecked():
            return NOT_IMPLEMENTED_FEATURE, None

        elif self.fourierRadioButton.isChecked():
            return SUCCESS, fou_enc.generate_heatmap(norm_img)

        else:
            return INVALID_METHOD, None

    def encodeIris(self):
        norm_iris_imag = self.normalized_iris_image
        norm_iris_mask = self.normalized_iris_mask

        # normalized image and it's mask must be valid
        if norm_iris_imag is None or norm_iris_mask is None:
            return INVALID_IMAGE, None, None

        if self.daugmanRadioButton.isChecked():
            # getting angular resolution and radial resolution
            radial_resolution, angular_resolution = norm_iris_imag.shape

            return proj_iris_enc.encode_iris(norm_iris_imag, norm_iris_mask, angular_resolution, radial_resolution)

        elif self.masekRadioButton.isChecked():
            return vasir_enc.encode_iris(norm_iris_imag, norm_iris_mask)

        elif self.zernikeCircularRadioButton.isChecked():
            return zcp_enc.encode_iris(norm_iris_imag, DEFAULT_ZERNIKE_ORDER)

        elif self.zernikeAnnularRadioButton.isChecked():
            return zap_enc.encode_iris(norm_iris_imag, norm_iris_mask, DEFAULT_ZERNIKE_ORDER, DEFAULT_EPS_INT, DEFAULT_EPS_EXT)

        elif self.fourierRadioButton.isChecked():
            # getting angular resolution and radial resolution
            radial_resolution, angular_resolution = norm_iris_imag.shape

            return fou_enc.encode_iris(norm_iris_imag, norm_iris_mask, angular_resolution, radial_resolution)

    def getSelectedSegmentationMethod(self):
        if self.projectIrisSegmRadioButton.isChecked():
            return PROJECT_IRIS_SEGMENTATION

        if self.vasirSegmRadioButton.isChecked():
            return VASIR_SEGMENTATION

        return INVALID_METHOD

    def getSelectedNormalizationMethod(self):
        if self.projectIrisNormRadioButton.isChecked():
            return PROJECT_IRIS_NORMALIZATION

        if self.vasirNormRadioButton.isChecked():
            return RUBBERSHEET_NORMALIZATION

        return INVALID_METHOD

    def getSelectedEncodingMethod(self):
        if self.projectIrisEncRadioButton.isChecked():
            return GABOR_FILTERS_ENCODING

        if self.vasirEncRadioButton.isChecked():
            return LOG_GABOR_ENCODING

        if self.zcpEncRadioButton.isChecked():
            return ZCP_ENCODING

        if self.zapEncRadioButton.isChecked():
            return ZAP_ENCODING

        if self.fourierEncRadioButton.isChecked():
            return FOURIER_ENCODING

    def getAdequateMatchingMethod(self):
        encoding_method = self.getSelectedEncodingMethod()

        if generates_binary_template(encoding_method):
            return HAMMING_DISTANCE

        if generates_vector_template(encoding_method):
            return EUCLIDEAN_DISTANCE

        return INVALID_METHOD

    def resetResultLabel(self):
        self.resultLabel.setText("?")
        self.resultLabel.setStyleSheet("")

    #-----------------------------Verification test------------------------------

    def configure_verification_test(self):
        db_type, encoding_method, use_mask, thres = self.get_test_parameters()

        # creating worker
        self.verif_test_worker = VerificationTest(self)

        # setting parameters
        self.verif_test_worker.db_type = db_type
        self.verif_test_worker.encoding_method = encoding_method
        self.verif_test_worker.use_mask = use_mask
        self.verif_test_worker.threshold = thres
        # setting K
        k = int(str(self.kComboBox.currentText()))
        self.verif_test_worker.polynomial_order = k

        # connecting the signals of the worker thread with slots in GUI
        self.verif_test_worker.verification_started.connect(self.onVerificationStarted)
        self.verif_test_worker.comparison_finished.connect(self.onComparisonFinished)
        self.verif_test_worker.verification_finished.connect(self.onVerificationFinished)

    def onVerificationStarted(self, db_type, enc_type, str_thres):
        self.resultsTextEdit.setTextBackgroundColor(QtGui.QColor(255, 255, 255))

        test_kind = "Verification test (1-to-1 comparisons)"
        db_kind = self.get_db_string(db_type)
        enc_kind = self.get_encoding_string(enc_type)

        # printing information
        self.resultsTextEdit.clear()
        self.resultsTextEdit.append("-------------------------------------------------------------------------------")

        self.resultsTextEdit.append("Selected test:\t\t%s" % test_kind)
        self.resultsTextEdit.append("Testing database:\t\t%s" % db_kind)
        self.resultsTextEdit.append("Encoding technique:\t\t%s" % enc_kind)
        self.resultsTextEdit.append("Selected threshold:\t\t%s" % str_thres)

        self.resultsTextEdit.append("-------------------------------------------------------------------------------")
        self.resultsTextEdit.append("")

    def onComparisonFinished(self, current, total, src_img, dst_img, ok, str_d):
        if ok:
            line = "%i    %s - %s    ->    %s (%s)\t" % (current, src_img, dst_img, "RIGHT", str_d)
            self.resultsTextEdit.setTextBackgroundColor(QtGui.QColor(221, 255, 222))
        else:
            line = "%i    %s - %s    ->    %s (%s)\t" % (current, src_img, dst_img, "WRONG", str_d)
            self.resultsTextEdit.setTextBackgroundColor(QtGui.QColor(254, 223, 220))

        self.resultsTextEdit.append(line)

    def onVerificationFinished(self, fa, fr, accepted, total):
        self.resultsTextEdit.setTextBackgroundColor(QtGui.QColor(255, 255, 255))

        # computing rates
        false_acceptance_rate = compute_far_percent(fa, total)
        false_reject_rate = compute_frr_percent(fr, total)
        accuracy = compute_accuracy(accepted, total)

        # printing final results
        self.resultsTextEdit.append("")
        self.resultsTextEdit.append("-------------------------------------------------------------------------------")
        self.resultsTextEdit.append("")
        line = "Comparisons made:\t%i" % total
        self.resultsTextEdit.append(line)
        line = "False acceptance rate:\t%.2f%%" % false_acceptance_rate
        self.resultsTextEdit.append(line)
        line = "False reject rate:\t%.2f%%" % false_reject_rate
        self.resultsTextEdit.append(line)
        line = "Accuracy:\t\t%.2f%%" % accuracy
        self.resultsTextEdit.append(line)
        self.resultsTextEdit.update()

        # enabling run button
        self.runButton.setEnabled(True)
        self.stopButton.setEnabled(False)

    #----------------------------Identification test-----------------------------

    def configure_identification_test(self):
        db_type, encoding_method, use_mask, _ = self.get_test_parameters()

        # creating worker
        self.ident_test_worker = IdentificationTest(self)

        # setting parameters
        self.ident_test_worker.db_type = db_type
        self.ident_test_worker.encoding_method = encoding_method
        self.ident_test_worker.use_mask = use_mask
        # setting K
        k = int(str(self.kComboBox.currentText()))
        self.ident_test_worker.polynomial_order = k

        # connecting the signals of the worker thread with slots in GUI
        QtCore.QObject.connect(self.ident_test_worker, QtCore.SIGNAL('identification_started(int, int)'), self.onIdentificationStarted)
        QtCore.QObject.connect(self.ident_test_worker, QtCore.SIGNAL('item_finished(int, int, QString, QString, int)'), self.onIdentificationItemFinished)
        QtCore.QObject.connect(self.ident_test_worker, QtCore.SIGNAL('identification_finished(int, int)'), self.onIdentificationFinished)

    def onIdentificationStarted(self, db_type, enc_type):
        self.resultsTextEdit.setTextBackgroundColor(QtGui.QColor(255, 255, 255))

        test_kind = "Identification test (1-to-many comparisons)"
        db_kind = self.get_db_string(db_type)
        enc_kind = self.get_encoding_string(enc_type)

        # printing information
        self.resultsTextEdit.clear()
        self.resultsTextEdit.append("-------------------------------------------------------------------------------")

        self.resultsTextEdit.append("Selected test:\t\t%s" % test_kind)
        self.resultsTextEdit.append("Testing database:\t\t%s" % db_kind)
        self.resultsTextEdit.append("Encoding technique:\t\t%s" % enc_kind)

        self.resultsTextEdit.append("-------------------------------------------------------------------------------")
        self.resultsTextEdit.append("")

    def onIdentificationItemFinished(self, current, total, src_img, dst_img, ok):
        if ok:
            line = "%i - %s    ->    %s\t\t" % (current, src_img, "RIGHT")
            self.resultsTextEdit.setTextBackgroundColor(QtGui.QColor(221, 255, 222))
        else:
            line = "%i - %s    ->    %s (%s)\t" % (current, src_img, "WRONG", dst_img)
            self.resultsTextEdit.setTextBackgroundColor(QtGui.QColor(254, 223, 220))

        self.resultsTextEdit.append(line)

    def onIdentificationFinished(self, accepted, total):
        self.resultsTextEdit.setTextBackgroundColor(QtGui.QColor(255, 255, 255))

        self.resultsTextEdit.append("")
        self.resultsTextEdit.append("-------------------------------------------------------------------------------")
        self.resultsTextEdit.append("")
        accuracy = compute_accuracy(accepted, total)
        line = "Accuracy %.2f%%" % accuracy
        self.resultsTextEdit.append(line)
        self.resultsTextEdit.update()

        # enabling run button
        self.runButton.setEnabled(True)
        self.stopButton.setEnabled(False)

    #----------------------------------------------------------------------------

    def onVerificationRadioButtonToggled(self, checked):
        if checked:
            self.thresholdSpinBox.setEnabled(True)
        else:
            self.thresholdSpinBox.setEnabled(False)

    #-----------------------------Testing auxiliars------------------------------

    def get_test_parameters(self):
        # getting selected encoding method
        if self.gaborEncRadioButton.isChecked():
            encoding_method = GABOR_FILTERS_ENCODING

        elif self.logGaborEncRadioButton.isChecked():
            encoding_method = LOG_GABOR_ENCODING

        elif self.zcpRadioButton.isChecked():
            encoding_method = ZCP_ENCODING

        elif self.zapRadioButton.isChecked():
            encoding_method = ZAP_ENCODING

        elif self.fourEncRadioButton.isChecked():
            encoding_method = FOURIER_ENCODING

        else:
            encoding_method = None

        # ---------------------------------------------------

        # getting selected testing database
        if self.upolRadioButton.isChecked():
            db_type = UPOL

        elif self.casia1RadioButton.isChecked():
            db_type = CASIA_1

        elif self.mmuRadioButton.isChecked():
            db_type = MMU

        elif self.ubirisRadioButton.isChecked():
            db_type = UBIRIS

        else:
            db_type = None

        # ---------------------------------------------------

        # determin if use mask or not
        use_mask = 1 if self.useMaskCheckBox.isChecked() else 0

        # ---------------------------------------------------

        # getting the threshold
        thres = self.thresholdSpinBox.value()

        return db_type, encoding_method, use_mask, thres

    def get_encoding_string(self, encoding_method):
        if encoding_method == GABOR_FILTERS_ENCODING:
            return GABOR_FILTERS_ENCODING_STR

        elif encoding_method == LOG_GABOR_ENCODING:
            return LOG_GABOR_ENCODING_STR

        elif encoding_method == ZCP_ENCODING:
            return ZCP_ENCODING_STR

        elif encoding_method == ZAP_ENCODING:
            return ZAP_ENCODING_STR

        elif encoding_method == FOURIER_ENCODING:
            return FOURIER_ENCODING_STR

        return "Unknown"

    def get_db_string(self, db_type):
        if db_type == UPOL:
            return UPOL_STR

        elif db_type == CASIA_1:
            return CASIA_1_STR

        elif  db_type == MMU:
            return MMU_STR

        elif db_type == UBIRIS:
            return UBIRIS_STR

        return "Unknown"

#-------------------------------------------------------------------------------------------

if __name__ == '__main__':
    import sys

    #creating the application
    app = QtWidgets.QApplication(sys.argv)

    #creting an instance of the main window
    ids_window = AirsWindow()

    #showing the window
    ids_window.show()

    #entering main event loop and returns the value setted via to exit()
    sys.exit(app.exec_())