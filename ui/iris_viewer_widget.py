import os
import pickle
from PyQt5 import (
    QtCore,
    QtWidgets
)

from ui.image_viewer_widget import ImageViewerWidget
from ui.ui_circle import UiCircle
from ui.ui_arc import UiArc

from utils.iris_data_definitions import *
from utils.math_utils import euclidean_distance_points

#-------------------------------------------------------------------------------------------

#detection modes of IrisViewerWidget
AUTOMATIC = 0
MANUAL = 1

#points that guarantee certain stages of detection
NOTHING_DETECTED = 0

PUPIL_CENTER_SET = 1
PUPIL_BORDER_SET = 2
PUPIL_DETECTED = 2

IRIS_CENTER_SET = 3
IRIS_BORDER_SET = 4
IRIS_DETECTED = 4

UPPER_EYELID_DETECTED = 7
LOWER_EYELID_DETECTED = 10
SEGMENTATION_DONE = 10

#point index and state of widget: ex. if control_points_set_count == PUPIL_CENTER
#it means that the widget is waiting for the pupil center control point
PUPIL_CENTER = 0
PUPIL_BORDER = 1
IRIS_CENTER = 2
IRIS_BORDER = 3
UPPER_EYELID_1 = 4
UPPER_EYELID_2 = 5
UPPER_EYELID_3 = 6
LOWER_EYELID_1 = 7
LOWER_EYELID_2 = 8
LOWER_EYELID_3 = 9

NEAR_RADIUS = 4.0
POINT_SIZE = 3.0

#-------------------------------------------------------------------------------------------


class IrisViewerWidget(ImageViewerWidget, object):
    # signal emitted when pupil center is set
    pupilCenterSet = QtCore.pyqtSignal()

    # signal emitted when pupil border is set
    pupilBorderSet = QtCore.pyqtSignal()

    # signal emitted when pupil is detected
    pupilDetected = QtCore.pyqtSignal()

    # signal emitted when iris center is set
    irisCenterSet = QtCore.pyqtSignal()

    # signal emitted when iris border is set
    irisBorderSet = QtCore.pyqtSignal()

    # signal emitted when iris is detected
    irisDetected = QtCore.pyqtSignal()

    # signal emitted when upper eyelid is detected
    upperEyelidDetected = QtCore.pyqtSignal()

    # signal emitted when lower eyelid is detected
    lowerEyelidDetected = QtCore.pyqtSignal()

    # signal emitted when iris is segmented
    irisSegmented = QtCore.pyqtSignal()

    # signal emitted when pupil center changes
    pupilCenterChanged = QtCore.pyqtSignal()

    # signal emitted when pupil border changes
    pupilBorderChanged = QtCore.pyqtSignal()

    # signal emitted when iris center changes
    irisCenterChanged = QtCore.pyqtSignal()

    # signal emitted when iris border changes
    irisBorderChanged = QtCore.pyqtSignal()

    # signal emitted when the widget is "resetted" for detection
    viewerReset = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        # calling base initializers
        super(IrisViewerWidget, self).__init__(parent)

        # necessary and don't know why
        self.resize(parent.size())

        # enabling mouse tracking (for the hover event, etc.)
        self.setMouseTracking(True)

        # setting focus policy
        self.setFocusPolicy(QtCore.Qt.NoFocus)

        # setting the Iris Viewer Configuration
        self.__setIrisViewerConfiguration()

        # setting the visual elements
        self.__setVisualElements()

        # adding a reset to the context menu
        self.enhanceContextMenu()

    def isInAutomaticDetectionMode(self):
        return self.detection_mode == AUTOMATIC

    def isInManualDetectionMode(self):
        return self.detection_mode == MANUAL

    def isPupilCenterSet(self):
        return self.control_points_set_count >= PUPIL_CENTER_SET

    def isPupilBorderSet(self):
        return self.control_points_set_count >= PUPIL_BORDER_SET

    def isPupilDetected(self):
        return self.control_points_set_count >= PUPIL_DETECTED

    def isIrisCenterSet(self):
        return self.control_points_set_count >= IRIS_CENTER_SET

    def isIrisBorderSet(self):
        return self.control_points_set_count >= IRIS_BORDER_SET

    def isIrisDetected(self):
        return self.control_points_set_count >= IRIS_DETECTED

    def isUpperEyelidDetected(self):
        return self.control_points_set_count >= UPPER_EYELID_DETECTED

    def isLowerEyelidDetected(self):
        return self.control_points_set_count >= LOWER_EYELID_DETECTED

    def isIrisSegmented(self):
        return self.control_points_set_count == SEGMENTATION_DONE

    def getPupilCenter(self):
        return self.control_points[PUPIL_CENTER].getPosition()

    def setPupilCenter(self, pos):
        if not self.setControlPoint(PUPIL_CENTER, pos):
            return 0

        self.pupilCenterSet.emit()
        self.pupilCenterChanged.emit()
        return 1

    def changePupilCenter(self, pos):
        if not self.changeControlPoint(PUPIL_CENTER, pos):
            return 0

        self.pupilCenterChanged.emit()
        return 1

    def getPupilBorder(self):
        return self.control_points[PUPIL_BORDER].getPosition()

    def setPupilBorder(self, pos):
        if not self.setControlPoint(PUPIL_BORDER, pos):
            return 0

        self.pupilBorderSet.emit()
        self.pupilBorderChanged.emit()
        self.pupilDetected.emit()
        return 1

    def changePupilBorder(self, pos):
        if not self.changeControlPoint(PUPIL_BORDER, pos):
            return 0

        self.pupilBorderChanged.emit()
        return 1

    def getIrisCenter(self):
        return self.control_points[IRIS_CENTER].getPosition()

    def setIrisCenter(self, pos):
        if not self.setControlPoint(IRIS_CENTER, pos):
            return 0

        self.irisCenterSet.emit()
        self.irisCenterChanged.emit()
        return 1

    def changeIrisCenter(self, pos):
        if not self.changeControlPoint(IRIS_CENTER, pos):
            return 0

        self.irisCenterChanged.emit()
        return 1

    def getIrisBorder(self):
        return self.control_points[IRIS_BORDER].getPosition()

    def setIrisBorder(self, pos):
        if not self.setControlPoint(IRIS_BORDER, pos):
            return 0

        self.irisBorderSet.emit()
        self.irisBorderChanged.emit()
        self.irisDetected.emit()
        return 1

    def changeIrisBorder(self, pos):
        if not self.changeControlPoint(IRIS_BORDER, pos):
            return 0

        self.irisBorderChanged.emit()
        return 1

    def getUpperEyelid(self):
        if not self.isUpperEyelidDetected():
            return None

        p1 = self.control_points[UPPER_EYELID_1].getPosition()
        p2 = self.control_points[UPPER_EYELID_2].getPosition()
        p3 = self.control_points[UPPER_EYELID_3].getPosition()
        return p1, p2, p3

    def setUpperEyelid(self, p1, p2, p3):
        if not self.setControlPoint(UPPER_EYELID_1, p1):
            return 0

        if not self.setControlPoint(UPPER_EYELID_2, p2):
            #rolling back the first eyelid point
            self.control_points_set_count -= 1
            return 0

        if not self.setControlPoint(UPPER_EYELID_3, p3):
            #rolling back the first two eyelid points
            self.control_points_set_count -= 2
            return 0

        self.upperEyelidDetected.emit()
        return 1

    def changeUpperEyelid(self, p1, p2, p3):
        old_p1 = self.control_points[UPPER_EYELID_1]
        old_p2 = self.control_points[UPPER_EYELID_2]

        if not self.changeControlPoint(UPPER_EYELID_1, p1):
            return 0

        if not self.changeControlPoint(UPPER_EYELID_2, p2):
            #rolling back the first eyelid point
            self.changeControlPoint(UPPER_EYELID_1, old_p1)
            return 0

        if not self.changeControlPoint(UPPER_EYELID_3, p3):
            #rolling back the first two eyelid points
            self.changeControlPoint(UPPER_EYELID_1, old_p1)
            self.changeControlPoint(UPPER_EYELID_2, old_p2)
            return 0

        return 1

    def getLowerEyelid(self):
        if not self.isLowerEyelidDetected():
            return None

        p1 = self.control_points[LOWER_EYELID_1].getPosition()
        p2 = self.control_points[LOWER_EYELID_2].getPosition()
        p3 = self.control_points[LOWER_EYELID_3].getPosition()
        return p1, p2, p3

    def setLowerEyelid(self, p1, p2, p3):
        if not self.setControlPoint(LOWER_EYELID_1, p1):
            return 0

        if not self.setControlPoint(LOWER_EYELID_2, p2):
            #rolling back the first eyelid point
            self.control_points_set_count -= 1
            return 0

        if not self.setControlPoint(LOWER_EYELID_3, p3):
            #rolling back the first two eyelid points
            self.control_points_set_count -= 2
            return 0

        self.lowerEyelidDetected.emit()
        self.irisSegmented.emit()
        return 1

    def changeLowerEyelid(self, p1, p2, p3):
        old_p1 = self.control_points[LOWER_EYELID_1]
        old_p2 = self.control_points[LOWER_EYELID_2]

        if not self.changeControlPoint(LOWER_EYELID_1, p1):
            return 0

        if not self.changeControlPoint(LOWER_EYELID_2, p2):
            #rolling back the first eyelid point
            self.changeControlPoint(LOWER_EYELID_1, old_p1)
            return 0

        if not self.changeControlPoint(LOWER_EYELID_3, p3):
            #rolling back the first two eyelid points
            self.changeControlPoint(LOWER_EYELID_1, old_p1)
            self.changeControlPoint(LOWER_EYELID_2, old_p2)
            return 0

        return 1

    #------------------------------Methods-------------------------------

    def enhanceContextMenu(self):
        # add a Reset action
        reset_action = QtWidgets.QAction("&Reset", self)
        reset_action.triggered.connect(self.reset)
        self.context_menu.addAction(reset_action)

        load_segmentation_action = QtWidgets.QAction("&Load Segmentation", self)
        load_segmentation_action.triggered.connect(self.load_segmentation_data)
        self.context_menu.addAction(load_segmentation_action)

        save_segmentation_action = QtWidgets.QAction("&Save Segmentation", self)
        save_segmentation_action.triggered.connect(self.save_segmentation_data)
        self.context_menu.addAction(save_segmentation_action)

    def __setIrisViewerConfiguration(self):
        #by default the Iris Viewer is in AUTOMATIC detection mode
        self.detection_mode = AUTOMATIC

        #by default there is no point selected
        self.selected_control_point_index = -1

        #by default there are 10 control points
        self.control_points_count = 10

        #by default there are no control points set
        self.control_points_set_count = 0

        #by default the first control point
        self.hover_control_point_index = 0

    def __setVisualElements(self):
        #creating pupil circle
        self.pupil_circle = UiCircle(self)
        self.pupil_circle.hide()

        #creting iris circle
        self.iris_circle = UiCircle(self)
        self.iris_circle.setInactiveColor(QtCore.Qt.blue)
        self.iris_circle.hide()

        #creating upper eyelid arc
        self.upper_eyelid_arc = UiArc(self)
        self.upper_eyelid_arc.hide()

        #creting lower eyelid arc
        self.lower_eyelid_arc = UiArc(self)
        self.lower_eyelid_arc.hide()

        #creating the control points
        self.control_points = list()
        for i in range(self.control_points_count):
            current_point = UiCircle(self)
            current_point.hide()
            self.control_points.append(current_point)

        #setting the color of iris control points blue
        self.control_points[IRIS_CENTER].setInactiveColor(QtCore.Qt.blue)
        self.control_points[IRIS_BORDER].setInactiveColor(QtCore.Qt.blue)

    def reset(self):
        #hiding control points
        for p in self.control_points:
            p.hide()
            p.setPosition(QtCore.QPointF(0, 0))

        #hiding visual elements
        self.pupil_circle.hide()
        self.iris_circle.hide()
        self.upper_eyelid_arc.hide()
        self.lower_eyelid_arc.hide()

        #resetting configuration
        self.selected_control_point_index = -1
        self.control_points_set_count = 0
        self.detection_mode = MANUAL

        #emitting the viewer reset signal
        self.viewerReset.emit()

    def nearestControlPointIndex(self, pos, radius):
        min_distance = radius
        best_index = -1

        for i in range(self.control_points_set_count):
            d = int(euclidean_distance_points(pos, self.control_points[i].getPosition()))

            if d < min_distance:
                min_distance = d
                best_index = i

        return best_index

    def emitDetectionSignals(self):
        if self.control_points_set_count == PUPIL_CENTER_SET:
            self.pupilCenterSet.emit()
            self.pupilCenterChanged.emit()

        elif self.control_points_set_count == PUPIL_BORDER_SET:
            self.pupilBorderSet.emit()
            self.pupilBorderChanged.emit()
            self.pupilDetected.emit()

        elif self.control_points_set_count == IRIS_CENTER_SET:
            self.irisCenterSet.emit()
            self.irisCenterChanged.emit()

        elif self.control_points_set_count == IRIS_BORDER_SET:
            self.irisBorderSet.emit()
            self.irisBorderChanged.emit()
            self.irisDetected.emit()

        elif self.control_points_set_count == UPPER_EYELID_DETECTED:
            self.upperEyelidDetected.emit()

        elif self.control_points_set_count == LOWER_EYELID_DETECTED:
            self.lowerEyelidDetected.emit()
            self.irisSegmented.emit()

    def emitChangesSignals(self):
        #generating signals
        if self.hover_control_point_index == PUPIL_CENTER:
            self.pupilCenterChanged.emit()

        elif self.hover_control_point_index == PUPIL_BORDER:
            self.pupilBorderChanged.emit()

        elif self.hover_control_point_index == IRIS_CENTER:
            self.irisCenterChanged.emit()

        elif self.hover_control_point_index == IRIS_BORDER:
            self.irisBorderChanged.emit()

    def isValidPoint(self, p):
        x = p.x()
        y = p.y()

        if self.qimage is None:
            return 0

        if x < 0 or x > self.qimage.width() - 1 or y < 0 or y > self.qimage.height() - 1:
            return 0

        return 1

    def setControlPoint(self, p_index, pos):
        #if the widget is not expecting the point you're trying to set it returns false
        if self.control_points_set_count != p_index:
            return 0

        #"setting" the point
        self.control_points_set_count += 1

        #if the control point could not be changed, then return false
        if not self.changeControlPoint(p_index, pos):
            #rolling back
            self.control_points_set_count -= 1
            return 0

        return 1

    def changeControlPoint(self, p_index, new_pos):
        #p_set points must have been set in order to change p_index control point
        p_set = p_index + 1

        #if control point at p_index hasn't been set yet, the return false
        if self.control_points_set_count < p_set:
            return 0

        #point coordinates must be valid
        if not self.isValidPoint(new_pos):
            return 0

        #changing the control point position
        self.control_points[p_index].change(new_pos, POINT_SIZE)
        #showing the control point
        self.control_points[p_index].show()

        return 1

    #-------------------------Overrided methods--------------------------

    def loadImage(self):
        # calling the loadImage of parent
        result = super(IrisViewerWidget, self).loadImage()

        # resetting all control points and ui elements
        self.reset()

        # returning the result that the ImageViewerWidget returned
        return result

    def mousePressEvent(self, QMouseEvent):
        #calling the mousePressEvent of parent
        super(IrisViewerWidget, self).mousePressEvent(QMouseEvent)

        #responding to the left click only
        if QMouseEvent.button() != QtCore.Qt.LeftButton:
            return

        # not responding if qimage is not valid
        if self.qimage is None:
            return

        pos = QMouseEvent.pos()
        self.selected_control_point_index = self.nearestControlPointIndex(pos, NEAR_RADIUS)

        #if a control point was clicked
        if self.selected_control_point_index >= 0:
            self.control_points[self.selected_control_point_index].setActive(True)
            self.control_points[self.selected_control_point_index].change(pos, POINT_SIZE)
        else:
            #if detection isn't finished yet
            if not self.isIrisSegmented():
                #adding the control point
                self.control_points[self.control_points_set_count].change(pos, POINT_SIZE)
                self.control_points[self.control_points_set_count].show()
                self.control_points_set_count += 1

                self.emitDetectionSignals()

    def mouseMoveEvent(self, QMouseEvent):
        #calling the mouseMoveEvent of parent
        super(IrisViewerWidget, self).mouseMoveEvent(QMouseEvent)

        pos = QMouseEvent.pos()
        nearestControlPointIndex = self.nearestControlPointIndex(pos, NEAR_RADIUS)

        #we are near a control point
        if nearestControlPointIndex >= 0:
            self.hover_control_point_index = nearestControlPointIndex
            self.control_points[self.hover_control_point_index].setActive(True)
        else:
            self.control_points[self.hover_control_point_index].setActive(False)

        #if left click is pressed
        if QMouseEvent.buttons() == QtCore.Qt.LeftButton:
            #if there's a control point selected
            if self.selected_control_point_index >= 0:
                self.control_points[self.selected_control_point_index].change(pos, POINT_SIZE)

            self.emitChangesSignals()

    def mouseReleaseEvent(self, QMouseEvent):
        #calling the mouseReleaseEvent of parent
        super(IrisViewerWidget, self).mouseReleaseEvent(QMouseEvent)

        #image must be valid
        if self.qimage is None:
            return

        #setting the control point hovered as inactive
        self.control_points[self.hover_control_point_index].setActive(False)

        #responding to the left click only
        if QMouseEvent.button() != QtCore.Qt.LeftButton:
            return

        #if there's a control point selected, we unselect such point
        if self.selected_control_point_index >= 0:
            #changing the control point to inactive
            self.control_points[self.selected_control_point_index].setActive(False)
            self.selected_control_point_index = -1

    def paintEvent(self, QPaintEvent):
        #calling the paintEvent of parent
        super(IrisViewerWidget, self).paintEvent(QPaintEvent)

        #image must be valid
        if self.qimage is None:
            return

        #drawing pupil circle
        if self.isPupilDetected():
            r_pupil = euclidean_distance_points(
                self.control_points[PUPIL_CENTER].getPosition(),
                self.control_points[PUPIL_BORDER].getPosition()
            )

            self.pupil_circle.change(self.control_points[PUPIL_CENTER].getPosition(), r_pupil)
            self.pupil_circle.show()

        #drawing iris circle
        if self.isIrisDetected():
            r_iris = euclidean_distance_points(
                self.control_points[IRIS_CENTER].getPosition(),
                self.control_points[IRIS_BORDER].getPosition()
            )

            self.iris_circle.change(self.control_points[IRIS_CENTER].getPosition(), r_iris)
            self.iris_circle.show()

        #drawing upper eyelid
        if self.isUpperEyelidDetected():
            self.upper_eyelid_arc.change(
                self.control_points[UPPER_EYELID_1].getPosition(),
                self.control_points[UPPER_EYELID_2].getPosition(),
                self.control_points[UPPER_EYELID_3].getPosition(),
                self.qimage.width()
            )

            self.upper_eyelid_arc.show()

        #drawing lower eyelid
        if self.isLowerEyelidDetected():
            self.lower_eyelid_arc.change(
                self.control_points[LOWER_EYELID_1].getPosition(),
                self.control_points[LOWER_EYELID_2].getPosition(),
                self.control_points[LOWER_EYELID_3].getPosition(),
                self.qimage.width()
            )
            self.lower_eyelid_arc.show()

        # draw control points (always)
        for i in range(self.control_points_set_count):
            self.control_points[i].show()

    def update(self, data):
        # reset the viewer
        self.reset()

        # drawing control points in the iris image viewer
        pupil_data = data[PUPIL_DATA]
        iris_data = data[IRIS_DATA]
        eyelids_data = data[EYELIDS_DATA]

        # setting pupil control points
        xp, yp = pupil_data[CENTER]
        rp = pupil_data[RADIUS]
        p_center = QtCore.QPointF(xp, yp)
        p_border = QtCore.QPointF(xp + rp, yp)
        self.setPupilCenter(p_center)
        self.setPupilBorder(p_border)

        # setting iris control points
        xi, yi = iris_data[CENTER]
        ri = iris_data[RADIUS]
        i_center = QtCore.QPointF(xi, yi)
        i_border = QtCore.QPointF(xi + ri, yi)
        self.setIrisCenter(i_center)
        self.setIrisBorder(i_border)

        # setting eyelids
        upper_eyelid_data, lower_eyelid_data = eyelids_data
        p1_data, p2_data, p3_data = upper_eyelid_data
        p1 = QtCore.QPointF(p1_data[0], p1_data[1])
        p2 = QtCore.QPointF(p2_data[0], p2_data[1])
        p3 = QtCore.QPointF(p3_data[0], p3_data[1])

        p4_data, p5_data, p6_data = lower_eyelid_data
        p4 = QtCore.QPointF(p4_data[0], p4_data[1])
        p5 = QtCore.QPointF(p5_data[0], p5_data[1])
        p6 = QtCore.QPointF(p6_data[0], p6_data[1])

        self.setUpperEyelid(p1, p2, p3)
        self.setLowerEyelid(p4, p5, p6)

    def load_segmentation_data(self):
        try:
            # split in base path and file name
            base_path, file_name = os.path.split(self.image_path)

            # get image name
            image_name, _ = os.path.splitext(file_name)

            # pickled data of segmentation path
            data_path = '{}/{}.seg'.format(base_path, image_name)

            # load pickled segmentation data
            with open(data_path, 'rb') as f:
                data = pickle.load(f)

                # update iris viewer
                self.update(data)

            QtWidgets.QMessageBox.information(self, "Information", "<h3>Segmentation data loaded</h3>")

            return True
        except Exception as e:
            QtWidgets.QMessageBox.about(self, "Error", "<h3>Operation failed</h3>")

            return False

    def save_segmentation_data(self):
        try:
            # split in base path and file name
            base_path, file_name = os.path.split(self.image_path)

            # get image name
            image_name, _ = os.path.splitext(file_name)

            # build segmentation data
            pupil = self.pupil_circle
            iris = self.iris_circle
            upper_eyelid = self.upper_eyelid_arc
            lower_eyelid = self.lower_eyelid_arc
            segmentation_data = (
                # pupil info
                ((pupil.center.x(), pupil.center.y()), pupil.radius),
                # iris info
                ((iris.center.x(), iris.center.y()), iris.radius),
                # eyelids
                (
                    # upper eyelid
                    (
                        (upper_eyelid.p1.x(), upper_eyelid.p1.y()),
                        (upper_eyelid.p2.x(), upper_eyelid.p2.y()),
                        (upper_eyelid.p3.x(), upper_eyelid.p3.y()),
                    ),
                    (
                        # lower eyelid
                        (lower_eyelid.p1.x(), lower_eyelid.p1.y()),
                        (lower_eyelid.p2.x(), lower_eyelid.p2.y()),
                        (lower_eyelid.p3.x(), lower_eyelid.p3.y()),
                    ),
                )
            )

            # save pickled data of segmentation
            data_path = '{}/{}.seg'.format(base_path, image_name)
            with open(data_path, 'wb') as f:
                pickle.dump(segmentation_data, f)

            QtWidgets.QMessageBox.information(self, "Information", "<h3>Segmentation data stored</h3>")

            return True
        except Exception as e:
            QtWidgets.QMessageBox.about(self, "Error", "<h3>Operation failed</h3>")

            return False
