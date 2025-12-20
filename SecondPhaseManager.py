### Fase 2
from dataclasses import dataclass, field
from typing import Any
from WatchPermanence import WatchPermanence
from queue import Queue
import threading
import time
from typing import Any, ClassVar

@dataclass
class SecondPhaseManager(): ### Way more linear than phase 1
    SLEEP_TIME = 0.000001
    queues_from_first_phase : list = field(default_factory=list) ###Receives from VideoFeedManager
    output_queues          : list = field(default_factory=list) ###Internal
    
    second_process_managers : list = field(default_factory=list) ###Internal

    def __post_init__(self):
        for i in range(len(self.queues_from_first_phase)):
            second_process_manager = SecondProcessManager()
            self.second_process_managers.append(second_process_manager)
        
    def run_second_process(self,pos):
        ### IN - OUT ###
        local_queue = self.queues_from_first_phase[pos]
        local_out_queue = self.output_queues[pos]
        
        local_second_process_manager = self.second_process_managers[pos]
        ### running process
        while True:
            time.sleep(self.SLEEP_TIME)
            if not local_queue.empty():
                list_of_temporary_person = local_queue.get_nowait()
                result = local_second_process_manager(list_of_temporary_person)
                local_out_queue.put(result)
    
    
    
    def start(self):
    # Iniciar threads
    for i in range(self.second_process_managers):
        thread = threading.Thread(
            target=self.run_second_process,
            args=(i,) ###args are the source, and the queue
        )
        thread.daemon = True ###Doesn't stop the program from ending
        thread.start() ###Create the thread
    
    def __call__(self):
        self.start()
        return output_queues
        

@dataclass
class SecondProcessManager(): ###Allow for best integration of all steps
    watch_permanence : WatchPermanence = field(default_factory=WatchPermanence)

    def __call__(self, list_of_temporary_person : list):
        return_watch_permanence = self.watch_permanence(list_of_temporary_person)
        ### por hora ###
        return return_watch_permanence
        
            
            