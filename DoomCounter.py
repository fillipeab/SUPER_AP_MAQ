from dataclasses import dataclass, field
import numpy as np
import time
import gc
from queue import Queue


@dataclass
class DoomCounter():
   queues_to_check           : list[Queue] = field(default_factory=list)
   SLEEP_TIME                : float = 0.000001
   CYCLES_TO_ACTION          : int   = 50
   waiting_multiplier_normal : int = 2000
   
   def __post_init__(self):
       self.queues_to_check = [Queue(),]
   def __call__(self):
        doom_counter = 0
        doom_flag    = 0
        waiting_multiplier = 0 + self.waiting_multiplier_normal
        try:
            while True:
                time.sleep(self.SLEEP_TIME*waiting_multiplier)
                ### breaking mechanism - stops the program once all queues are empty. Keep in mind that second_phase barely uses queues ###
                if doom_counter % self.CYCLES_TO_ACTION == 0: ###printing takes a lot of time. Do it only for important values
                        ###ACTIONS TO COMPLETE###
                        ###print (doom_counter) ### see if the process is getting to the end
                        gc.collect()

                if doom_counter == 1000:
                    try: ###checking for empty queues
                        if all(q.empty() for q in self.queues_to_check):
                            doom_flag+=1
                            if doom_flag == 1:
                                print("\nDoom is aproaching\n")
                            elif doom_flag == 2:
                                print("\nDoom is INEVITABLE!\n")
                            elif doom_flag == 3:
                                print("\nDOOOOOOOOOOM!!!\n")
                            waiting_multiplier = 1
                        elif all(q.empty() for q in self.queues_to_check[:-1]): ###which menas that there's only output to process
                            waiting_multiplier = 1
                        else:
                            doom_flag=0
                            waiting_multiplier = 0 + self.waiting_multiplier_normal
                    except Exception as e:
                        print("Erro: ",e)
                        
                    doom_counter = 0 ### reset doom counter
                doom_counter+=1
                
                if doom_flag>=3 :
                    break
                ### breaking mechanism - END ###
                        
        except Exception as e:
            print("ERRO IN DOOM:" , e)
        except KeyboardInterrupt:
            print("DOOM interrupted")
