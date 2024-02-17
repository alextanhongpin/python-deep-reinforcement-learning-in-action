```python
import sys

!{sys.executable} --version
```

    Python 3.11.7


https://github.com/DeepReinforcementLearning/DeepReinforcementLearningInAction

## Sampling 


```python
from collections import deque
replay = deque(maxlen=10)
for i in range(20):
    replay.append(i)
replay
```




    deque([10, 11, 12, 13, 14, 15, 16, 17, 18, 19], maxlen=10)




```python
import random
random.sample(replay, 5)
```




    [11, 13, 18, 14, 16]



## Index


```python
replay = deque(maxlen=10)
for i in range(20):
    replay.append((i, i+1, i+2))
replay
```




    deque([(10, 11, 12),
           (11, 12, 13),
           (12, 13, 14),
           (13, 14, 15),
           (14, 15, 16),
           (15, 16, 17),
           (16, 17, 18),
           (17, 18, 19),
           (18, 19, 20),
           (19, 20, 21)],
          maxlen=10)




```python
sample = random.sample(replay, 5)
sample
```




    [(15, 16, 17), (16, 17, 18), (14, 15, 16), (13, 14, 15), (11, 12, 13)]




```python
import numpy as np
np.array(sample)[:, [0]]
```




    array([[15],
           [16],
           [14],
           [13],
           [11]])



## Torch


```python
import torch
a = torch.Tensor([[1,2,3]])
b = torch.Tensor([[4,5,6]])
torch.cat((a, b), 0)
```




    tensor([[1., 2., 3.],
            [4., 5., 6.]])




```python
torch.cat((a, b), 1)
```




    tensor([[1., 2., 3., 4., 5., 6.]])




```python
torch.flip(torch.arange(10), dims=(0,))
```




    tensor([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])




```python
sample = torch.rand(3,4)
sample
```




    tensor([[0.4472, 0.1623, 0.4897, 0.2531],
            [0.5884, 0.3918, 0.2846, 0.7775],
            [0.2466, 0.1625, 0.7652, 0.2257]])




```python
torch.flip(sample, dims=(0,))
```




    tensor([[0.2466, 0.1625, 0.7652, 0.2257],
            [0.5884, 0.3918, 0.2846, 0.7775],
            [0.4472, 0.1623, 0.4897, 0.2531]])




```python
torch.flip(sample, dims=(0,1))
```




    tensor([[0.2257, 0.7652, 0.1625, 0.2466],
            [0.7775, 0.2846, 0.3918, 0.5884],
            [0.2531, 0.4897, 0.1623, 0.4472]])




```python
torch.flip(sample, dims=(1,))
```




    tensor([[0.2531, 0.4897, 0.1623, 0.4472],
            [0.7775, 0.2846, 0.3918, 0.5884],
            [0.2257, 0.7652, 0.1625, 0.2466]])




```python

```
