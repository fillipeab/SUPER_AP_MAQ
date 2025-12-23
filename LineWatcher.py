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
class LineWatcher(): ###Needs way more work, and maybe it's not the best
    ### receives the list of people
    PERCENT_CUT_TRIM_MEAN         : float = 0.1
    NEIGHBOUR_MAX_RADIUS_DISTANCE : float = 4     ###How distance, in smaller_axis value, can a neighbour be - This will be passed to find_nearest_neighbors
    ERASE_FROM_DICT_TIMEOUT       : int   = 30    ###Remember, that means the number of runs to erase an entry
    people_neighbour_id_dict      : dict  = field(default_factory=dict) ###List to remember people and their neighbours
    people_timeout_dict           : dict  = field(default_factory=dict) ###Timeout to erasure
    previous_number_of_people_in_line       : int = 0
    
    #Neighbour disruption variables
    max_number_of_neighbours             : int   = 0
    min_number_of_neighbours             : int   = 255 ### to our code, that's almost like infinity. Remember people can't really compact that much
    neighbour_self_disruption_weight     : float = 1
    neighbourhood_disruption_weight      : float = 1

    
    ###PROBABILITY_SKIPPER###
    PS_MORE_PEOPLE_IN_LINE  : float = 0.2  ###If there's more people
    PS_NOT_NEAR_BORDER      : float = 0.2  ###If there's more people, and in the middle, the change is big
    PS_NEIGHBOUR_DISRUPTION : float = 0.45  ##must happen
    PS_CONFIRMED_SKIPPER    : float = 0.60 ###at least 2 to be considered. More people in line is a heavy indicator. But needs the other 2
    
    BOLD_INTERNAL_SKIPPER         : float = 0.25 
    ###Keep in mind, this is only possible if
    ### neighbours       - percent if they change all neighbours
    ### 1 neighbour     - 50%
    ### 2 neighbours    - 33%
    ### 3 neighbours    - 25%

    ###static method
    def calculate_neighbourhood(self, list_of_temporary_people, radius : float = 50):
    
        centers_with_ids = np.array([[(p.bb[0]+p.bb[2])/2, (p.bb[1]+p.bb[3])/2, p.id] 
                        for p in list_of_temporary_people])
        
        if len(list_of_temporary_people) == 0:
            return {}

        tree = cKDTree(centers_with_ids[:, :2])
        neighbors_by_id = {}
        
        for i, row in enumerate(centers_with_ids):
            center = row[:2]  # x,y
            person_id = int(row[2])  # ID
            
            indices = tree.query_ball_point(center, radius)

            ###indices.remove(i) - person is neighbour of themselves
            
            neighbors_ids = torch.tensor([int(centers_with_ids[idx][2]) for idx in indices])
            neighbors_by_id[person_id] = neighbors_ids
        return neighbors_by_id
    
    def calculate_smaller_axis_average(self,list_of_temporary_people, percent_cut : float = 0):
        if percent_cut == 0:
            percent_cut = self.PERCENT_CUT_TRIM_MEAN
        smaller_dimension_list = [min(abs(x2-x1), abs(y2-y1)) for p in list_of_temporary_people for x1, y1, x2, y2 in [p.bb]]
        average = trim_mean(smaller_dimension_list, percent_cut) ###Finds the mean, cutting extremes
        return average
    
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

    def neighbourhood_disruption(
        self,
        person_id : int, 
        neighbour_self_disruption_weight     : float = 1,
        neighbourhood_disruption_weight      : float = 1
        ) -> float:
        ### Lets balance like 50% to their number, 50% to their neighbours influence

        if self.max_number_of_neighbours == self.min_number_of_neighbours: ###there's no way to tell who is the disruptor without making asumptions
            return 0
        
        neighbour_self_disruption : float = 0
        neighbourhood_disruption : float = 0

        ### SELF DISRUPTION ###
        neighbour_number = len(self.people_neighbour_id_dict[person_id])
        neighbour_self_disruption = (self.max_number_of_neighbours-neighbour_number)/(self.max_number_of_neighbours-self.min_number_of_neighbours) ### 0 means its the max disruption. The bigger the number, the lest likely to be a disruptor
        neighbour_self_disruption = 1 - neighbour_self_disruption ###NOW it represents the disrupting quality
        ### SELF DISRUPTION - END ###

        ### NEIGHBOURHOOD DISRUPTION ###
        associated_neighbours : float = 0
        an_counter            : int   = 0
        all_neighbours        : float = 0
        all_counter           : int   = 0
        ### everyone ###
        for person_id in self.people_neighbour_id_dict: ###for the people in his id
            all_neighbours += len(self.people_neighbour_id_dict[person_id])
            all_counter+=1
        all_neighbours_mean = all_neighbours/all_counter ###gets average
        ### everyone - END ###
        
        ### associated ###
        ### gets for the people that are associated with the chosen person
        personal_dict = self.people_neighbour_id_dict[person_id]
        for neighbour_id in personal_dict:
            associated_neighbours+=len(self.people_neighbour_id_dict[int(neighbour_id)])
            an_counter+=1
        associated_neighbours_mean = associated_neighbours/an_counter ###gets average
        neighbourhood_disruption = (associated_neighbours_mean - all_neighbours_mean)/all_neighbours_mean ### the bigger this is, the more he got to know people with a lot of contacts. Which means he's skipping
        ### associated - end ###
        
        ### NEIGHBOURHOOD DISRUPTION - END ###

        total_disruption = (neighbour_self_disruption*neighbour_self_disruption_weight+neighbourhood_disruption*neighbourhood_disruption_weight)/(neighbour_self_disruption_weight*neighbourhood_disruption_weight)
        return total_disruption

    def similarity_neighbourhood(self, t1, t2):
        combined = torch.cat([t1, t2])
        unique, counts = combined.unique(return_counts=True)
        same = (counts > 1).sum().item()
        total_unique = len(unique)
        return same / total_unique if total_unique > 0 else 1.0

    def __call__(self,list_of_temporary_people : list[TempPerson], frame_shape : Tuple[int,...] = (720, 1280, 3)): ###LineWatcher is called
        return_dict : dict[int, Literal["skipper", "in line", "wait"]]  = {}
        ### 0 - calculate neighbourhood ###
        average_min_distance = self.calculate_smaller_axis_average(list_of_temporary_people , 0.2)
        new_neighbourhood_dict = self.calculate_neighbourhood(list_of_temporary_people, float(average_min_distance * self.NEIGHBOUR_MAX_RADIUS_DISTANCE))
        ### 1 - check list - get new people ###
        new_people = []
        for temp_person in list_of_temporary_people:
            if temp_person:
                t_id=int(temp_person.id)
                if t_id not in self.people_timeout_dict:
                    new_people.append(temp_person)
                else: ###old people in dict
                    ###compare to find internal line skipers - only the bold ones
                    old_neighbour = self.people_neighbour_id_dict[t_id]
                    new_neighbourhood = new_neighbourhood_dict[t_id]
                    IoU_neighbourhood = self.similarity_neighbourhood(old_neighbour,new_neighbourhood)
                    if IoU_neighbourhood < self.BOLD_INTERNAL_SKIPPER:
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
            skipping_probability = 0

            ### checking new people
            for temp_person in new_people:
                skipping_probability += base_skipping_probability
                ### cheks for near the border
                if not self.is_near_border(temp_person.bb,frame_shape): ###se nao for, + chance de ser um fura fila
                    skipping_probability += self.PS_NOT_NEAR_BORDER
                ### end of near the border
                ### NEIGHBOUR DISRUPTION   ###
                skipping_probability += (
                self.neighbourhood_disruption(int(temp_person.id), neighbour_self_disruption_weight = self.neighbour_self_disruption_weight, neighbourhood_disruption_weight = self.neighbour_self_disruption_weight)
                *self.PS_NEIGHBOUR_DISRUPTION
                )
                ### NEIGHBOUR DISRUPTION - END ###

                ### FINALLY, CHECKS FOR SKIPPERS ###
                if skipping_probability>self.PS_CONFIRMED_SKIPPER:
                    return_dict[int(temp_person.id)]="skipper"
                else:
                    return_dict[int(temp_person.id)]="in line" ### WITH THIS, EVERYONE IS EITHER CLASSIFIED AS "IN LINE" OR "SKIPPER". They can still be spotted skipping. The process of return someone to in_line was not implemented. The reason is simple. It would require some degree of guessing, and could make the model go in conflict with itself. Future implementations, using machine learning, could get it without much trouble, especially in a classification problem such as this.
        ###Finally, updates variables
        self.previous_number_of_people_in_line=number_people_in_line
        
        ###cleaning the dicts###
        self.people_timeout_dict = {k: v-1 for k, v in self.people_timeout_dict.items() if v > 1}
        for key in set(self.people_neighbour_id_dict) - set(self.people_timeout_dict):
            self.people_neighbour_id_dict.pop(key, None)

        return return_dict














### Calculating neighbourhood disruption - but how? well, we just need to check the number of entries of this person\
### max_entries ---- person_entries ---- minimum entries.
### It's reasonable that a new person would be close to the minimum. The farthest from it, the more probable to be skiping.
### Also, the same goes for their new neighbours. If their neighbours are close to the top, basically.