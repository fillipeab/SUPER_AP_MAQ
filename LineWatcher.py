###LINE WATCHER - THE END OF OUR JORNEY
from dataclasses import dataclass, field
from typing import Any
from scipy.stats import trim_mean
from scipy.spatial import cKDTree
import numpy as np

@dataclass
class LineWatcher():
    ### receives the list of people
    NEIGHBOUR_MAX_RADIUS_DISTANCE : int   = 1   ###How distance, in smaller_axis value, can a neighbour be - This will be passed to find_nearest_neighbors
    PERCENT_CUT_TRIM_MEAN         : float = 0.1
    ###static method
    @staticmethod
    def find_nearest_neighbors_by_radius(list_of_temporary_people, radius=50):
        """People within radius of each person based on BBox centers."""
        # Calculate centers from BBoxes
        centers = np.array([
            [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
            for person in temporary_people_list
            for bbox in [person.bbox]
        ])
        
        tree = cKDTree(centers)
        neighbors = {}
        
        for i, center in enumerate(centers):
            indices = tree.query_ball_point(center, radius)
            indices.remove(i)  # Remove self
            neighbors[i] = indices
        return neighbors
    
    def find_smaller_axis_of_people(self, list_of_temporary_people, percent_cut = self.PERCENT_CUT_TRIM_MEAN):
        smaller_dimension_list = []
        for person in list_of_temporary_people:
            bbox = person.bb ###Its in x1,y1,x2,y2 format
            smaller_dimension = x_variation = abs(bbox[0] - bbox[2])
            y_variation = abs(bbox[1] - bbox[3])
            if x_variation>y_variation:
                smaller_dimension = y_variation
            ###add to list
            smaller_dimension_list.append(smaller_dimension)
        trim_mean(smaller_dimension_list, percent_cut) ###Finds the mean, cutting extremes
    
    
    def __call__(list_of_temporary_people): ###LineWatcher is called
        