###LINE WATCHER - THE END OF OUR JORNEY
from dataclasses import dataclass, field
from typing import Any, Literal, Tuple
from scipy.stats import trim_mean
from scipy.spatial import cKDTree
import torch
import numpy as np
from ultralytics.utils.metrics import bbox_iou
from TempPerson import TempPerson



















@dataclass
class LineWatcher_neighbour(): ###Needs way more work, and maybe it's not the best
    ### receives the list of people
    PERCENT_CUT_TRIM_MEAN         : float = 0.1
    NEIGHBOUR_MAX_RADIUS_DISTANCE : int   = 1   ###How distance, in smaller_axis value, can a neighbour be - This will be passed to find_nearest_neighbors
    ERASE_FROM_DICT_TIMEOUT       : int   = 180 ###Remember, that means the number of runs to erase an entry
    people_neighbour_id_dict      : dict  = field(default_factory=dict) ###List to remember people and their neighbours
    people_timeout_dict           : dict  = field(default_factory=dict) ###Timeout to erasure
    previous_number_of_people_in_line       : int = 0
    
    #Neighbour disruption variables
    max_number_of_neighbours = 0
    min_number_of_neighbours = 255 ### to our code, that's almost like infinity. Remember people can't really compact that much
    
    ###PROBABILITY_SKIPPER###
    PS_MORE_PEOPLE_IN_LINE  : float = 0.2  ###If there's more people
    PS_NOT_NEAR_BORDER      : float = 0.2  ###If there's more people, and in the middle, the change is big
    PS_NEIGHBOUR_DISRUPTION : float = 0.5  ##must happen
    PS_CONFIRMED_SKIPPER    : float = 0.60 ###at least 2 to be considered. More people in line is a heavy indicator. But needs the other 2
    
    BOLD_INTERNAL_SKIPPER         : float = 0.25 
    ###Keep in mind, this is only possible if
    ### neighbours       - percent if they change all neighbours
    ### 1 neighbour     - 50%
    ### 2 neighbours    - 33%
    ### 3 neighbours    - 25%

    ###static method
    def calculate_neighbourhood(list_of_temporary_people, radius=50):
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

            ###indices.remove(i) - person is neighbour of themselves
            
            # Pega IDs dos vizinhos
            neighbors_ids = torch.tensor([int(centers_with_ids[idx][2]) for idx in indices])
            neighbors_by_id[person_id] = neighbors_ids
        
        return neighbors_by_id
    
    def calculate_smaller_axis_average(self,list_of_temporary_people, percent_cut = 0):
        if percent_cut == 0:
            percent_cut = self.PERCENT_CUT_TRIM_MEAN
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
    
    def is_near_border(self, bbox, frame_shape, margin_percent=0.05):
        """
        Check if bbox [x1,y1,x2,y2] is near frame border.
        bbox: [x1, y1, x2, y2]
        frame_shape: (height, width) from frame.shape[:2]
        margin_percent: border margin as fraction (0.0 to 0.5)
        bool: True if near any border
        """
        h, w = frame_shape[:2]
        margin = int(min(h, w) * margin_percent)
        
        x1, y1, x2, y2 = bbox
        return (x1 <= margin or x2 >= w - margin or 
                y1 <= margin or y2 >= h - margin)

    def neighbourhood_disruption(self,person_id):
        ### Lets balance like 50% to their number, 50% to their neighbours influence
        if self.max_number_of_neighbours == self.min_number_of_neighbours:
            return 0
        ### person proper disruption
        neighbour_number = len(self.people_neighbour_id_dict[person_id])
        neighbour_self_disruption = (self.max_number_of_neighbours-neighbour_number)/(self.max_number_of_neighbours-self.min_number_of_neighbours)

        ### Comparisson with other people


    def __call__(self,list_of_temporary_people : list[TempPerson], frame_shape : Tuple[int,...] = (720, 1280, 3)): ###LineWatcher is called
        return_dict : dict[int, Literal["skipper", "in line", "wait"]]  = {}
        ### 0 - calculate neighbourhood ###
        average_min_distance = self.calculate_smaller_axis_average(list_of_temporary_people,0.2)
        new_neighbourhood_dict = self.calculate_neighbourhood(list_of_temporary_people, average_min_distance*self.NEIGHBOUR_MAX_RADIUS_DISTANCE)
        ### 1 - check list - get new people ###
        new_people = []
        for temp_person in list_of_temporary_people:
            t_id=temp_person.id
            if t_id not in self.people_timeout_dict:
                new_people.append(temp_person)
            else: ###old people in dict
                ###compare to find internal line skipers - only the bold ones
                old_neighbour = self.people_neighbour_id_dict[t_id]
                bbox_iou(old_neighbour[t_id],new_neighbourhood_dict[t_id])
                if bbox_iou < self.BOLD_INTERNAL_SKIPPER:
                    ###FOUND A BOLD ONE
                    return_dict[t_id] = "skipper"
                else:
                    pass
            self.people_neighbour_id_dict[t_id] = new_neighbourhood_dict[t_id] ### Adds/updates neighbourhood
            self.people_timeout_dict[t_id] = self.ERASE_FROM_DICT_TIMEOUT ### Anyone has the timeout redifined

            ###update min/max - for neighbourhood disruption
            if self.max_number_of_neighbours < len(new_neighbourhood_dict):
                self.max_number_of_neighbours = len(new_neighbourhood_dict)
            if self.min_number_of_neighbours > len(new_neighbourhood_dict):
                self.min_number_of_neighbours = len(new_neighbourhood_dict)
            
        ###End of 1###
        ### new people localized ###
        number_people_in_line = len(list_of_temporary_people)
        if new_people:
            base_skipping_probability = 0
            ###check if the number of people in line changed
            if number_people_in_line > self.previous_number_of_people_in_line:
                base_skipping_probability+=self.PS_MORE_PEOPLE_IN_LINE
            for temp_person in list_of_temporary_people:
                skipping_probability += base_skipping_probability
                if not self.is_near_border(temp_person.bb,frame_shape): ###se nao for, + chance de ser um fura fila
                    skipping_probability += self.PS_NOT_NEAR_BORDER
                ### Calculating neighbourhood disruption - but how? well, we just need to check the number of entries of this person\
                ### max_entries ---- person_entries ---- minimum entries.
                ### It's reasonable that a new person would be close to the minimum. The farthest from it, the more probable to be skiping.
                ### Also, the same goes for their new neighbours. If their neighbours are close to the top, basically.



        ###Finally, updates variables
        self.previous_number_of_people_in_line=number_people_in_line
