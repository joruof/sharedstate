import numpy as np
import sharedstate as shs

store = shs.open("test")
root = store.get()
root.read_lock_field("test_bool")

print(root.test_float)
print(root.test_np)
print(root.test_string)

root.test_np = np.zeros((5, 10))
print(root.test_np)
