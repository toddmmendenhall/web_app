import numpy as np
import matplotlib.pyplot as plt
from utilities import Constants
from matplotlib.collections import PathCollection
from shapely.geometry import LineString


class ContourIntersectionFinder:
    def __init__(self, tempGrid: np.ndarray, muBGrid: np.ndarray, eGrid: np.ndarray, nBGrid: np.ndarray, es: np.ndarray, nBs: np.ndarray) -> None:
        """_summary_

        Args:
            tempGrid (np.ndarray): _description_
            muBGrid (np.ndarray): _description_
            eGrid (np.ndarray): _description_
            nBGrid (np.ndarray): _description_
            es (np.ndarray): _description_
            nBs (np.ndarray): _description_
        """
        self.eContours = []
        self.nBContours = []
        for e, nB in zip(es, nBs):
            eContour = plt.contour(muBGrid, tempGrid, eGrid/Constants.MeVPerGeV**4/Constants.hbarTimesC**3, [e])
            nBContour = plt.contour(muBGrid, tempGrid, nBGrid/Constants.MeVPerGeV**3/Constants.hbarTimesC**3, [nB])
            self.eContours.append(eContour.collections)
            self.nBContours.append(nBContour.collections)
        self.__findIntersections()
    
    
    def __findIntersection(self, contour1: PathCollection, contour2: PathCollection) -> list:
        intersections = []
        for path1 in list(contour1[0].get_paths()):
            for path2 in list(contour2[0].get_paths()):
                lines1 = LineString(path1.vertices)
                lines2 = LineString(path2.vertices)
                intersections.append(lines1.intersection(lines2))
        return intersections


    def __findIntersections(self) -> None:
        self.intersections = []
        for eContour, nBContour in zip(self.eContours, self.nBContours):
            intersections = self.__findIntersection(eContour, nBContour)
            if not intersections:
                self.intersections.append([np.nan, np.nan])
            else:
                for intersection in intersections:
                    if intersection.geom_type == 'Point':
                        self.intersections.append([intersection.x, intersection.y])
                    else:
                        self.intersections.append([np.nan, np.nan])

        self.intersections = np.array(self.intersections).T
