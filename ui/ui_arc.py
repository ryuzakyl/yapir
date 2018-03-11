from PyQt5 import QtGui, QtCore, QtWidgets

from utils.math_utils import fit_parabola_points, in_parabola_point


class UiArc(QtWidgets.QWidget, object):

    def __init__(self, parent=None):
        #calling base initializers
        super(UiArc, self).__init__(parent)

        self.is_active = 0

        #necessary and don't know why
        self.resize(parent.size())

        self.p1 = None  #QPointF
        self.p2 = None  #QPointF
        self.p3 = None  #QPointF

        self.A = 0.0
        self.B = 0.0
        self.C = 0.0

        #arc width
        self.arc_width = 0

        self.arc_pen = QtCore.Qt.green

    #-----------------------------Properties-----------------------------

    def isActive(self):
        return self.is_active

    def setActive(self, value):
        self.is_active = value

    def getA(self):
        return self.A

    def getB(self):
        return self.B

    def getC(self):
        return self.C

    #------------------------------Methods-------------------------------

    def fitParabola(self):
        A, B, C = fit_parabola_points(self.p1, self.p2, self.p3)

        if A == -1 and B == -1 and C == -1:
            return

        self.A = A
        self.B = B
        self.C = C

    def change(self, p1, p2, p3, arc_width):
        self.is_active = 1

        self.arc_width = arc_width

        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.fitParabola()  #computing A, B and C coefficients

        #updating
        self.update()

    def contains(self, p):
        return in_parabola_point(self.A, self.B, self.C, p)

    #-------------------------Overrided methods--------------------------

    def paintEvent(self, QPaintEvent):
        #calling the paintEvent of parent
        super(UiArc, self).paintEvent(QPaintEvent)

        qpainter = QtGui.QPainter(self)
        qpainter.setPen(self.arc_pen)

        #computing coefficients A, B and C
        self.fitParabola()

        #drawing the arc
        A = self.A
        B = self.B
        C = self.C
        for x in range(self.arc_width):
            y = int(A*x*x + B*x + C)
            qpainter.drawPoint(x, y)