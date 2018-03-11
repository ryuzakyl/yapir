from PyQt5 import QtGui, QtCore, QtWidgets


class UiCircle(QtWidgets.QWidget):
    def __init__(self, parent=None, center=QtCore.QPointF(0.0, 0.0), radius=1.0):
        #calling base initializers
        super(UiCircle, self).__init__(parent)

        #necessary and don't know why
        self.resize(parent.size())

        self.center = center
        self.radius = radius
        self.active_color = QtCore.Qt.yellow
        self.inactive_color = QtCore.Qt.green

        self.is_active = 0

        #enabling mouse tracking
        self.setMouseTracking(True)

    #-----------------------------Properties-----------------------------

    def getActiveColor(self):
        return self.active_color

    def setActiveColor(self, value):
        self.active_color = value

    def getInactiveColor(self):
        return self.inactive_color

    def setInactiveColor(self, value):
        self.inactive_color = value

    def getPosition(self):
        return self.center

    def setPosition(self, value):
        self.center = value
        self.update()

    def getSize(self):
        return self.radius

    def setSize(self, value):
        self.radius = value
        self.update()

    def isActive(self):
        return self.is_active

    def setActive(self, value):
        self.is_active = value
        self.update()

    #------------------------------Methods-------------------------------

    def change(self, center, radius):
        self.center = center
        self.radius = radius
        self.update()

    #-------------------------Overrided methods--------------------------

    def paintEvent(self, QPaintEvent):
        #calling the paintEvent of parent
        super(UiCircle, self).paintEvent(QPaintEvent)

        qpainter = QtGui.QPainter(self)

        if self.is_active:
            qpainter.setPen(self.active_color)
        else:
            qpainter.setPen(self.inactive_color)

        qpainter.drawEllipse(self.center, self.radius, self.radius)
