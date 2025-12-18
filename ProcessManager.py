import threading
from queue import Queue
from typing import Any
from dataclasses import dataclass, field


### The objective os this class is to implement a system that does all the processing of the ID part, and the REID part. We will leave the writing part to the MemoryManager(do not confuse with the MemorySystem).