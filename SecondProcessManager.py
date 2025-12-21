from dataclasses import dataclass, field
from typing import Any
from PermanenceWatcher import PermanenceWatcher
from MovementWatcher import MovementWatcher



@dataclass
class SecondProcessManager(): ###Allow for best integration of all steps
    ### expected flux of information
    ### permanence_watcher -> movement_watcher -> watch_line -> skipper_buster
    permanence_watcher      : PermanenceWatcher = field(default_factory=PermanenceWatcher)
    movement_watcher        : MovementWatcher   = field(default_factory=MovementWatcher)
    SKIP_PERMANENCE         : int   = 0
    SKIP_MOVEMENT           : int   = 0
    SKIP_LINE               : int   = 0
    counter                 : int   = 0

    def __call__(self, list_of_temporary_person : list):
        return_from_permanence_watcher = []
        return_from_movement_watcher = []
        return_from_line = []
        if self.counter % (self.SKIP_PERMANENCE+1) == 0:
            return_from_permanence_watcher = self.permanence_watcher(list_of_temporary_person)
            if self.counter % ((self.SKIP_PERMANENCE*self.SKIP_MOVEMENT)+1) == 0:
                return_from_movement_watcher = self.movement_watcher(return_from_permanence_watcher)
                self.counter=0
                """
                if self.counter % (self.SKIP_PERMANENCE*self.SKIP_MOVEMENT*self.SKIP_LINE+1) == 0:
                    return_from_line = self.watch_line(return_from_movement_watcher)
            """        
                
        self.counter+=1
        ### por hora ###
        return return_from_permanence_watcher, return_from_movement_watcher, return_from_line