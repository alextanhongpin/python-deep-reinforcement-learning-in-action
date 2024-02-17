import multiprocessing as mp
import numpy as np

queue = mp.Queue()
processes = []


def square(i, x, queue):
    print(f"In process {i}")
    queue.put(np.square(x))


x = np.arange(64)
for i in range(mp.cpu_count()):
    start_index = 16 * i
    proc = mp.Process(target=square, args=(i, x[start_index : start_index + 16], queue))
    proc.start()
    processes.append(proc)


for proc in processes:
    proc.join()

for proc in processes:
    proc.terminate()

results = [queue.get() for _ in range(mp.cpu_count())]
print(results)
