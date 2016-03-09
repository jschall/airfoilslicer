import csv
import sys
from math import *
from GCodeGen import GCodeGen
import polygonutil
import re
import numpy as np
from scipy.optimize import minimize

def getSurfaceCoords(fname):
    surfaceSectionRegex = r'Airfoil surface,\nX\(mm\),Y\(mm\)\n(([-+]?[0-9]*\.?[0-9]+),([-+]?[0-9]*\.?[0-9]+)\n)+'
    camberSectionRegex = r'Camber line,\nX\(mm\),Y\(mm\)\n(([-+]?[0-9]*\.?[0-9]+),([-+]?[0-9]*\.?[0-9]+)\n)+'
    floatRegex = r'([-+]?[0-9]*\.?[0-9]+)'

    with open(fname) as csvfile:
        csvstring = csvfile.read()

    # find the surface section
    surfacesection = re.search(surfaceSectionRegex, csvstring).group(0)

    # find all the floats in the surface section
    floatlist = re.findall(floatRegex, surfacesection)

    ret = []
    for i in range(0, len(floatlist), 2):
        ret.append((float(floatlist[i]), float(floatlist[i+1])))
    return ret

class AirfoilSlicer:
    def __init__(self, fname, wing_length=200., root_chord=100., washout=radians(1.0), dihedral=radians(1.0), sweep=radians(0.0), taper_ratio=0.75, print_center=(100.,100.)):
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

    def applyTransformations(self,Z,points):
        offset = np.array([tan(self.sweep)*Z,tan(self.dihedral)*Z])
        twist = self.washout * Z/self.wing_length
        chord = self.root_chord * (1. - (1.-self.taper_ratio)*Z/self.wing_length)

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


    def getSupports2(self,Z):
        spar = self.getSupportLine(Z,self.thickestCamberDist,pi/2)
        supports = [spar]
        printTruss = False
        for trussLoc in np.linspace(.5,self.wing_length-.5, self.wing_length/20.):
            if abs(trussLoc-Z) < 0.5:
                printTruss = True

        if printTruss:
            # find truss locations for this layer
            pass

        return supports


    def getSupports(self,Z):
        supportLocations = np.linspace(.6,self.wing_length-.6, self.wing_length/20.)

        verticalSupport = False
        crossSupport = False

        for loc in supportLocations:
            if abs(loc-Z) < 0.6:
                verticalSupport = True
            if Z-loc > -0.3 and Z-loc<0.6:
                crossSupport = True

        supportCenters = [ 0.10044348,  0.22599783,  0.35155218,  0.47710653,  0.60266088,  0.7021523, (0.72821523+0.85376958)/2, 0.86376958, .92]
        supportDirections = []
        if verticalSupport:
            supportDirections.append(pi/2.)
        if crossSupport:
            supportDirections.append(pi/4.)
            supportDirections.append(3.*pi/4.)

        intersectShape = polygonutil.shrinkPolygon(self.applyTransformations(Z,self.normSurface),0.4)

        supports = []

        for centerDist in supportCenters:
            centerPoint = self.applyTransformations(Z,[polygonutil.traversePolyLine(self.normCamberLine,centerDist)])[0][0:2]
            camberAngle = polygonutil.getPolyLineDirection(self.normCamberLine,centerDist)
            for direction in supportDirections:
                supportUnitVec = np.array([cos(direction+camberAngle), sin(direction+camberAngle)])
                point1 = polygonutil.getClosestIntersection(centerPoint, [centerPoint, centerPoint+supportUnitVec*1000.], intersectShape)
                point2 = polygonutil.getClosestIntersection(centerPoint, [centerPoint, centerPoint-supportUnitVec*1000.], intersectShape)
                if point1 is not None and point2 is not None:
                    supports.append([np.append(point1,[Z]), np.append(point2,[Z])])

        return supports

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
            firstLayer = False
            print "%f/%f" % (gcg.Z, self.wing_length)
            lines = polygonutil.getLineSegments(self.getSurface(gcg.Z))
            for seg in lines:
                gcg.drawLine(self.translatePointToPrintCenter(seg[0][0:2]), self.translatePointToPrintCenter(seg[1][0:2]))

            supports = self.getSupports(gcg.Z)

            for support in supports:
                lines = polygonutil.getLineSegments(support)
                for seg in lines:
                    if reverse:
                        gcg.drawLine(self.translatePointToPrintCenter(seg[1][0:2]), self.translatePointToPrintCenter(seg[0][0:2]), bridge=True)
                    else:
                        gcg.drawLine(self.translatePointToPrintCenter(seg[0][0:2]), self.translatePointToPrintCenter(seg[1][0:2]), bridge=True)
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
        for supports in slicer.getSupports(Z):
            #print supports
            curve(pos=supports, color=color.red)
