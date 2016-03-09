import csv
import sys
from math import *
from GCodeGen import GCodeGen
import polygonutil
import re
import numpy as np
from scipy.optimize import minimize

class AirfoilSlicer:
    def __init__(self, fname, wing_length=200., root_chord=120., washout=radians(1.0), dihedral=radians(1.0), sweep=radians(0.0), taper_ratio=0.75, print_center=(100.,100.)):
        self.print_center=np.asarray(print_center)
        self.wing_length=wing_length
        self.root_chord=root_chord
        self.washout=washout
        self.dihedral=dihedral
        self.sweep=sweep
        self.taper_ratio=taper_ratio
        self.readFile(fname)

    def parsePoints(self, instr):
        floatRegex = r'([-+]?[0-9]*\.?[0-9]+)'
        floats = map(float,re.findall(floatRegex, instr))
        ret = []
        for i in range(0, len(floats), 2):
            ret.append(np.array([floats[i], floats[i+1]]))

        return ret


    def readFile(self, fname):
        chordRegex = r'Chord\(mm\),([-+]?[0-9]*\.?[0-9]+)'
        surfaceSectionRegex = r'Airfoil surface,\nX\(mm\),Y\(mm\)\n(([-+]?[0-9]*\.?[0-9]+),([-+]?[0-9]*\.?[0-9]+)\n)+'
        chordSectionRegex = r'Chord line,\nX\(mm\),Y\(mm\)\n(([-+]?[0-9]*\.?[0-9]+),([-+]?[0-9]*\.?[0-9]+)\n)+'
        camberSectionRegex = r'Camber line,\nX\(mm\),Y\(mm\)\n(([-+]?[0-9]*\.?[0-9]+),([-+]?[0-9]*\.?[0-9]+)\n)+'


        with open(fname) as csvfile:
            csvstring = csvfile.read()

        chord=float(re.search(chordRegex, csvstring).group(1))

        # find the surface section
        surfaceSection = re.search(surfaceSectionRegex, csvstring).group(0)

        # get the points
        self.normSurface = map(lambda x: x/chord,self.parsePoints(surfaceSection))

        # find the chord section
        chordSection = re.search(chordSectionRegex, csvstring).group(0)
        self.normChordLine = map(lambda x: x/chord,self.parsePoints(chordSection))

        # find the camber section
        camberSection = re.search(camberSectionRegex, csvstring).group(0)

        # get the points
        self.normCamberLine = map(lambda x: x/chord,self.parsePoints(camberSection))

        # get the camber line length
        self.normCamberLineLength = 0.
        for i in range(0, len(self.normCamberLine)-1):
            secLen = np.linalg.norm(self.normCamberLine[i+1]-self.normCamberLine[i])
            self.normCamberLineLength += secLen

        # center everything around the aerodynamic center
        aeroCenter = polygonutil.traversePolyLine(self.normChordLine,.25)
        self.normSurface = map(lambda x: x-aeroCenter,self.normSurface)
        self.normChordLine = map(lambda x: x-aeroCenter,self.normChordLine)
        self.normCamberLine = map(lambda x: x-aeroCenter,self.normCamberLine)
        self.halfChordPoint = polygonutil.traversePolyLine(self.normChordLine,0.5)

        self.thickestCamberDist = minimize(lambda d: -self.getNormThicknessAtCamberPoint(d), self.normCamberLineLength/2.).x[0]

    def getChordLength(self,Z):
        return self.root_chord * (1. - (1.-self.taper_ratio)*Z/self.wing_length)

    def applyTransformations(self,Z,points):
        offset = np.array([tan(self.sweep)*Z,tan(self.dihedral)*Z])
        twist = self.washout * Z/self.wing_length
        chord = self.getChordLength(Z)

        return map(lambda x: np.append(x*chord+offset,[Z]), polygonutil.rotatePoints(points,twist))

    def getNormLineAcrossCamber(self,d,lineAngle):
        camberPoint = polygonutil.traversePolyLine(self.normCamberLine,d)
        camberAngle = polygonutil.getPolyLineDirection(self.normCamberLine,d)

        pt1 = polygonutil.getClosestIntersection(camberPoint, [camberPoint, camberPoint+np.array([cos(lineAngle+camberAngle), sin(lineAngle+camberAngle)])*1000], self.normSurface)
        pt2 = polygonutil.getClosestIntersection(camberPoint, [camberPoint, camberPoint-np.array([cos(lineAngle+camberAngle), sin(lineAngle+camberAngle)])*1000], self.normSurface)
        return (pt1,pt2)

    def getSupportLine(self,Z,d,lineAngle):
        gap = 0.3
        camberPoint = self.applyTransformations(Z,[polygonutil.traversePolyLine(self.normCamberLine,d)])[0][:2]
        camberAngle = polygonutil.getPolyLineDirection(self.normCamberLine,d)
        upVector = self.applyTransformations(Z,[np.array([cos(lineAngle+camberAngle), sin(lineAngle+camberAngle)])])[0][:2]

        intersectPoly = polygonutil.shrinkPolygon(self.getSurface(Z),gap)

        pt1 = np.append(polygonutil.getClosestIntersection(camberPoint, [camberPoint, camberPoint+upVector], intersectPoly),[Z])
        pt2 = np.append(polygonutil.getClosestIntersection(camberPoint, [camberPoint, camberPoint-upVector], intersectPoly),[Z])

        return [pt1,pt2]


    def getNormThicknessAtCamberPoint(self,d):
        pt1,pt2 = self.getNormLineAcrossCamber(d,pi/2)

        return np.linalg.norm(pt2-pt1)

    def printTruss(self,Z):
        printTruss = False
        for trussLoc in np.linspace(.25,self.wing_length-.25, self.wing_length/12.5):
            if abs(trussLoc-Z) <= 0.25:
                printTruss = True
        return printTruss

    def getTrussVerticals(self,Z):
        numTrussVerticals = 10
        return map(lambda x: self.getSupportLine(Z,x,pi/2),np.linspace(0.+1.*self.normCamberLineLength/16.,self.normCamberLineLength*(1.-1./8.),numTrussVerticals))

    def getSpar(self,Z):
        verticals = self.getTrussVerticals(Z)
        verticalLengths = map(lambda x: np.linalg.norm(x[0]-x[1]), verticals)
        maxLength = max(verticalLengths)
        return verticals[verticalLengths.index(maxLength)]

    def getTrussCrosses(self,Z):
        verticals = self.getTrussVerticals(Z)
        ret = []

        for i in range(len(verticals)-1):
            if i%2 == 0:
                ret.append([verticals[i][0], verticals[i+1][1]])
            else:
                ret.append([verticals[i][1], verticals[i+1][0]])

        for i in reversed(range(len(verticals)-1)):
            if i%2 == 0:
                ret.append([verticals[i+1][0], verticals[i][1]])
            else:
                ret.append([verticals[i+1][1], verticals[i][0]])

        return ret

    def getSurface(self, Z):
        return self.applyTransformations(Z,self.normSurface)

    def getCamber(self,Z):
        return self.applyTransformations(Z,self.normCamberLine)

    def getChord(self,Z):
        return self.applyTransformations(Z,self.normChordLine)

    def translatePointToPrintCenter(self, pt):
        return np.asarray(pt)-self.halfChordPoint+self.print_center

    def genGCode(self, fname):
        gcg = GCodeGen()
        firstLayer = True
        reverse=False
        while gcg.Z+0.1 < self.wing_length:
            gcg.beginLayer(0.3 if firstLayer else 0.1)
            if firstLayer:
                gcg.drawLine((10,10),(10,190))
                gcg.drawLine((10,190),(12,190))
                gcg.drawLine((12,190),(12,10))
            firstLayer = False
            print "%f/%f" % (gcg.Z, self.wing_length)
            lines = polygonutil.getLineSegments(self.getSurface(gcg.Z))
            if reverse:
                lines = map(lambda x: list(reversed(x)),reversed(lines))
            for seg in lines:
                gcg.drawLine(self.translatePointToPrintCenter(seg[0][0:2]), self.translatePointToPrintCenter(seg[1][0:2]))

            if self.printTruss(gcg.Z):
                verticals = self.getTrussVerticals(gcg.Z)
                if not reverse:
                    verticals = map(lambda x: list(reversed(x)),reversed(verticals))

                for seg in verticals:
                    gcg.drawLine(self.translatePointToPrintCenter(seg[0][0:2]), self.translatePointToPrintCenter(seg[1][0:2]), bridge=True)

                crosses = self.getTrussCrosses(gcg.Z)
                if reverse:
                    crosses = map(lambda x: list(reversed(x)),reversed(crosses))

                for seg in crosses:
                    gcg.drawLine(self.translatePointToPrintCenter(seg[0][0:2]), self.translatePointToPrintCenter(seg[1][0:2]), bridge=True)
            else:
                seg = self.getSpar(gcg.Z)
                if reverse:
                    seg = list(reversed(seg))
                gcg.drawLine(self.translatePointToPrintCenter(seg[0][0:2]), self.translatePointToPrintCenter(seg[1][0:2]))

            reverse = not reverse

        gcg.output(fname)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gcode', dest='gcodefile')
parser.add_argument('--visualize', action='store_true')
parser.add_argument('csvfile')

args = parser.parse_args()

slicer = AirfoilSlicer(args.csvfile)

if args.gcodefile is not None:
    slicer.genGCode(args.gcodefile)

if args.visualize:
    from visual import *

    for Z in linspace(0.,slicer.wing_length,1000):
        curve(pos=slicer.getSurface(Z))
        #curve(pos=slicer.getSpar(Z), color=color.yellow)
        #curve(pos=slicer.getCamber(Z), color=color.yellow)
        #curve(pos=slicer.getChord(Z), color=color.green)
        if slicer.printTruss(Z):
            for p in slicer.getTrussVerticals(Z):
                curve(pos=p, color=color.red)
            for p in slicer.getTrussCrosses(Z):
                curve(pos=p, color=color.yellow)
        else:
            curve(pos=slicer.getSpar(Z), color=color.green)
