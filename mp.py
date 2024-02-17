# multiprocessing does not work in Jupyter notebook.
import multiprocessing as mp

import numpy as np


def square(x):
    return np.square(x)


x = np.arange(64)

cpu_count = mp.cpu_count()
print("CPU count:", cpu_count)

pool = mp.Pool(cpu_count)
squared = pool.map(square, [x[16 * i : 16 * i + 16] for i in range(4)])
print(squared)
