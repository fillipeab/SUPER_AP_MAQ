from dataclasses import dataclass, field
from typing import Any, Tuple
from PermanenceWatcher import PermanenceWatcher
from MovementWatcher import MovementWatcher
from LineWatcher import LineWatcher


@dataclass
class SecondProcessManager(): ###Allow for best integration of all steps
    ### expected flux of information
    ### permanence_watcher -> movement_watcher -> watch_line -> skipper_buster
    permanence_watcher      : PermanenceWatcher = field(default_factory=PermanenceWatcher)
    movement_watcher        : MovementWatcher   = field(default_factory=MovementWatcher)
    line_watcher            : LineWatcher       = field(default_factory=LineWatcher)   
    SKIP_PERMANENCE         : int   = 0
    SKIP_MOVEMENT           : int   = 0
    SKIP_LINE               : int   = 0
    counter                 : int   = 0

    def __call__(self, list_of_temporary_person : list, frame_shape : Tuple[int,...]):
        return_from_permanence_watcher = []
        return_from_movement_watcher = []
        return_from_line_watcher = []
        
        if self.counter % (self.SKIP_PERMANENCE+1) == 0:
            return_from_permanence_watcher = self.permanence_watcher(list_of_temporary_person)
            if self.counter % ((self.SKIP_PERMANENCE*self.SKIP_MOVEMENT)+1) == 0:
                return_from_movement_watcher = self.movement_watcher(return_from_permanence_watcher)
                if self.counter % ((self.SKIP_PERMANENCE*self.SKIP_MOVEMENT*self.SKIP_LINE)+1) == 0:
                    return_from_line_watcher = self.line_watcher(return_from_movement_watcher,frame_shape)
                    self.counter=0
                
        self.counter+=1
        ### por hora ###
        ###print(return_from_permanence_watcher,return_from_movement_watcher,return_from_line_watcher)
        return return_from_permanence_watcher, return_from_movement_watcher, return_from_line_watcher