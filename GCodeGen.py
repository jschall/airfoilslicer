#!/usr/bin/python

# airfoil slicer
# accepts coordinate CSVs from http://airfoiltools.com/plotter
# outputs gcode

from math import *

class GCodeGen:
    def __init__(self):
        self.extruderTemp = 190.
        self.filamentDiameter = 1.75
        self.extruderDiameter = 0.4
        self.layerHeight = 0.3

        self.travelSpeed=150.
        self.liftSpeed=130.
        self.retractSpeed=40.
        self.printSpeed=90.
        self.bridgeSpeed=45.

        self.X = 0.
        self.Y = 0.
        self.Z = 0.
        self.E = 0.
        self.F = 5000.

        self.gCodeLines = []

        self.writeHeader()

    def beginLayer(self,layerHeight=None):
        if layerHeight is not None:
            self.layerHeight = layerHeight
        self.resetExtruder()
        self.writeG1(Z=self.Z+self.layerHeight,F=self.liftSpeed*60.)

    def retract(self,length=2.0,lift=0.5):
        self.writeG1(E=self.E-length, F=self.retractSpeed*60.)
        self.writeG1(Z=self.Z+lift, F=self.liftSpeed*60.)

    def unRetract(self,length=2.0,lift=0.5):
        self.writeG1(Z=self.Z-lift, F=self.liftSpeed*60.)
        self.writeG1(E=self.E+length, F=self.retractSpeed*60.)

    def resetExtruder(self):
        self.gCodeLines.append('G92 E0')
        self.E = 0.

    def drawLine(self, p1, p2, lineWidth=None, bridge=False):
        if lineWidth is None:
            lineWidth = self.extruderDiameter

        x1 = p1[0]
        y1 = p1[1]
        x2 = p2[0]
        y2 = p2[1]

        if self.X != x1 or self.Y != y1:
            # line does not start at current location, travel there
            if sqrt((self.X-x1)**2+(self.Y-y1)**2) > 2.:
                self.retract()
                self.writeG1(X=x1, Y=y1, F=self.travelSpeed*60.)
                self.unRetract()
            else:
                self.writeG1(X=x1, Y=y1, F=self.travelSpeed*60.)

        extrusion = self.getExtrusionForMoveLength(x1,y1,x2,y2,lineWidth,bridge)

        speed = self.bridgeSpeed if bridge else self.printSpeed

        self.writeG1(X=x2, Y=y2, E=self.E+extrusion,F=speed*60.)

    def getExtrusionForMoveLength(self, x1,y1,x2,y2, layerWidth, bridge):
        moveLength = sqrt((x2-x1)**2+(y2-y1)**2)
        h = self.layerHeight*0.5
        r = layerWidth*0.5
        if bridge:
            trackArea = pi*(self.extruderDiameter*0.5)**2 * .95 # 5% stretch
        else:
            trackArea = pi*r**2 - 2.*(0.5 * r**2 * 2.*acos(h/r) - 0.5 * h * sqrt(r**2-h**2))
        filamentArea = pi*(self.filamentDiameter*0.5)**2
        return moveLength*trackArea/filamentArea

    def writeG1(self, X=None, Y=None, Z=None, E=None, F=None):
        args = ""

        if X is not None and X != self.X:
            self.X = X
            args += " X%f" % (self.X)

        if Y is not None and Y != self.Y:
            self.Y = Y
            args += " Y%f" % (self.Y)

        if Z is not None and Z != self.Z:
            self.Z = Z
            args += " Z%f" % (self.Z)

        if E is not None and E != self.E:
            self.E = E
            args += " E%f" % (self.E)

        if F is not None and F != self.F:
            self.F = F
            args += " F%f" % (self.F)

        if args != "":
            self.gCodeLines.append("G1"+args)

    def writeHeader(self):
        self.gCodeLines.extend([
            "; generated by GCodeGen.py",
            "M107",
            "M190 S50 ; set bed temperature",
            "G28 X0 Y0 Z0 ; home all axes",
            "G1 Z5 F5000 ; lift nozzle",
            "M109 S%u ; set the extruder temp and wait" % (int(self.extruderTemp)),
            "G28 X0 Y0 Z0 ; Home Z again in case there was filament on nozzle",
            "G29 ; probe the bed",
            "G21 ; set units to millimeters",
            "G90 ; use absolute coordinates",
            "M82 ; use absolute distances for extrusion",
            "G92 E0",
            ])

    def writeFooter(self):
        self.retract()
        self.gCodeLines.extend([
            "M104 S0 ; turn off temperature",
            "G1 X10 Y200",
            "M84     ; disable motors"
            ])

    def output(self, fname):
        self.writeFooter()
        f = open(fname,'w')
        f.truncate()
        f.write("\n".join(self.gCodeLines))
        f.close()
