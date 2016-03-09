import numpy as np
from math import *
from shapely.geometry import LineString, MultiPoint, Point, Polygon

def traversePolyLine(line, dist):
    accumLen = 0.
    for segment in getLineSegments(line):
        secLen = np.linalg.norm(segment[1]-segment[0])
        remainingLen = dist - accumLen
        if secLen > remainingLen:
            return segment[0]+(segment[1]-segment[0])*remainingLen/secLen
        accumLen += secLen

def getPolyLineDirection(line, dist):
    accumLen = 0.
    for segment in getLineSegments(line):
        segVec = segment[1]-segment[0]
        segLen = np.linalg.norm(segVec)
        remainingLen = dist - accumLen
        if segLen > remainingLen:
            return atan2(segVec[1],segVec[0])
        accumLen += segLen

def shrinkPolygon(points, shrink):
    poly = Polygon(points).buffer(-shrink)
    if poly.is_empty:
        return []
    else:
        return list(poly.exterior.coords)

def rotatePoints(points,theta):
    R = np.matrix([
        [cos(theta), -sin(theta)],
        [sin(theta), cos(theta)]
        ])
    return map(lambda x:np.asarray(R*np.matrix(x).T).flatten(),points)

def getLineSegments(points):
    lineSegments = []
    for i in range(0, len(points)-1):
        lineSegments.append((points[i],points[i+1]))
    return lineSegments

def getPolyLineIntersections(polyline1, polyline2):
    polyline1 = LineString(polyline1)
    polyline2 = LineString(polyline2)
    intersection = polyline1.intersection(polyline2)
    if isinstance(intersection,Point):
        return [np.array([intersection.x, intersection.y])]
    if isinstance(intersection,MultiPoint):
        return map(lambda x:np.array([x.x,x.y]),intersection)

    return []


def getClosestIntersection(p1, polyline1, polyline2):
    intersections = getPolyLineIntersections(polyline1, polyline2)

    closest = None
    closestDist = None
    for p2 in intersections:
        dist = np.linalg.norm(p2-p1)
        if closestDist is None or dist < closestDist:
            closest = p2
            closestDist = dist

    return closest
