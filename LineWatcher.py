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
        # Array com: [x_center, y_center, person_id]
        centers_with_ids = np.array([
            [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2, person.id]
            for person in list_of_temporary_people
            for bbox in [person.bbox]
        ])
        
        # Árvore só com coordenadas (ignora ID na distância)
        tree = cKDTree(centers_with_ids[:, :2])  # Apenas x,y
        neighbors_by_id = {}
        
        for i, row in enumerate(centers_with_ids):
            center = row[:2]  # x,y
            person_id = int(row[2])  # ID
            
            indices = tree.query_ball_point(center, radius)
            indices.remove(i)
            
            # Pega IDs dos vizinhos
            neighbors_ids = [int(centers_with_ids[idx][2]) for idx in indices]
            neighbors_by_id[person_id] = neighbors_ids
        
        return neighbors_by_id
    
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
        """
        Processing order
        0 - dict is empty - just adds everyone to dict, and their neighbours
        1 - check neighbours
        1.1 - runs the check neighbours
        1.2 - updates old people
        1.3 - NEW_PEOPLE -> time to check for line skippers - activate line checking
        2.1 - Is there more people? Yes => follows to 2.2. No -> skips
        2.2 - Is near corner? No => Line_skipper_prob + 0.2
        2.3 - 
        2.4 - Neighbours gained/average_number_of_neighbours from neighbours
        --this step is very important. It compares the number of neighbours gained sudenly with the number of number of neighbours of his neighbours. That is, checks if he is near "known" people, that already had a lot of neighbours.
        3.1 - Adds new people to dict, with neighbours
        Think of a line:
        D -> C -> B -> A| oclusion at the end
        E -> D -> C -> B|
        --now, compares E with D. D is close to B, C and E. So its gains are only small.
        --now, think about if E skips between C and B:
        D -> C -> E -> B|
        --E knows B. B, an "old"/"known" person in line, knows A, C and D. See that E is getting to know a big number of people. Maybe more than



        4.1 - Checks for Internal Agents - that is, people that WERE in line, but skipped to another place
        """