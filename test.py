import time
import numpy as np
import sharedstate as shs


class State:

    def __init__(self):

        self.test_float = 42.0
        self.test_bool = False
        self.test_int = 1
        self.test_string = "hello"
        self.test_np = np.ones((3, 100))

        self.test_obj = SubState()
        self.test_obj_second = SubState()


class SubState:

    def __init__(self):

        self.test_substate_float = 3.0
        self.test_subsub = SubSubState()


class SubSubState:

    def __init__(self):

        self.test_subsubstate_float = 4.0


store = shs.create("test", State())
root = store.get()

while True:
    time.sleep(1)
    print(root)
