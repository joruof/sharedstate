import time
import imviz as viz
import numpy as np
import sharedstate as shs

store = shs.open("test")
root = store.get()
root.read_lock_field("test_bool")

print(root)
print(root.__slots__)
print(hasattr(root, "__autogui__"))

while viz.wait():
    store.check_for_upgrade()
    root = store.get()

    if viz.begin_window("test"):
        viz.autogui(root, name="root")
    viz.end_window()

print(root.test_float)
print(root.test_np)
print(root.test_string)

root.test_np = np.zeros((5, 10))
print(root.test_np)
