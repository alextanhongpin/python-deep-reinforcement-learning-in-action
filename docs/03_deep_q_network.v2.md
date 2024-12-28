# Deep Q-Network

## GridWorld


```python
from gridworld.gridworld import GridWorld

env = GridWorld(4)
env.reset(seed=42)
env.render()
```




    [['W', '.', '.', '.'],
     ['.', 'P', '.', '.'],
     ['.', '+', '.', '.'],
     ['-', '.', '.', '.']]




```python
# 0: up, 1: down, 2: left, 3: right
ob, reward, terminated, truncated, info = env.step(0)
print("reward: {} | terminated: {}".format(reward, terminated))
env.render()
```

    reward: -1 | terminated: False





    [['W', 'P', '.', '.'],
     ['.', '.', '.', '.'],
     ['.', '+', '.', '.'],
     ['-', '.', '.', '.']]




```python
ob, reward, terminated, truncated, info = env.step(1)
print("reward: {} | terminated: {}".format(reward, terminated))
env.render()
```

    reward: -1 | terminated: False





    [['W', '.', '.', '.'],
     ['.', 'P', '.', '.'],
     ['.', '+', '.', '.'],
     ['-', '.', '.', '.']]




```python
ob, reward, terminated, truncated, info = env.step(1)
print("reward: {} | terminated: {}".format(reward, terminated))
env.render()
```

    reward: 10 | terminated: True





    [['W', '.', '.', '.'],
     ['.', '.', '.', '.'],
     ['.', 'P', '.', '.'],
     ['-', '.', '.', '.']]




```python
# Each row represents the index of the pieces on the tile in the order:
# player, goal, pit, wall
env.observation
```




    [[[0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]],
     [[0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]],
     [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]],
     [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]




```python
import numpy as np

assert np.array(env.observation).shape == (4, 4, 4)
```

## Neural Network Q Function


```python
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from gridworld.gridworld import GridWorld
from tqdm.notebook import tqdm, trange
```


```python
model = torch.nn.Sequential(
    # Input is our gridworld state, a 1x64 grid representation.
    torch.nn.Linear(64, 150),
    torch.nn.ReLU(),
    torch.nn.Linear(150, 100),
    torch.nn.ReLU(),
    # Output is the expected rewards for each arm: 0: up, 1: down, 2: left, 3: right
    torch.nn.Linear(100, 4),
)
model
```




    Sequential(
      (0): Linear(in_features=64, out_features=150, bias=True)
      (1): ReLU()
      (2): Linear(in_features=150, out_features=100, bias=True)
      (3): ReLU()
      (4): Linear(in_features=100, out_features=4, bias=True)
    )




```python
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```


```python
def dqn(model, optimizer, epoch=1000, epsilon=1.0, gamma=0.9, seed=None):
    noise = lambda: torch.rand(1, 64) / 64
    losses = []
    for i in trange(epoch):
        env = GridWorld(size=4)
        env.reset(seed=seed)  # Static board.

        # Initial state, with noise.
        observation = env.observation
        while True:
            state = torch.Tensor(observation).view(-1, 64) + noise()
            q_val = model(state)
            if np.random.random() < epsilon:  # Exploration
                action = torch.randint(4, (1,)).item()
            else:
                action = torch.argmax(q_val, dim=1).item()
            observation, reward, terminated, _truncated, _info = env.step(action)
            if not terminated:
                state = torch.Tensor(observation).view(-1, 64) + noise()
                with torch.no_grad():
                    next_q_val = model(state)
                max_q = torch.max(next_q_val)
                reward = reward + gamma * max_q

            y_pred = torch.Tensor([reward])
            y_true = q_val[:, action]
            loss = loss_fn(y_pred, y_true)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            if terminated:
                break
        if epsilon > 0.1:
            epsilon -= 1 / epoch
    return losses
```


```python
losses = dqn(model, optimizer, epoch=1000, seed=33)  # Static board
plt.plot(np.arange(len(losses)), losses);
```


      0%|          | 0/1000 [00:00<?, ?it/s]



    
![png](03_deep_q_network.v2_files/03_deep_q_network.v2_13_1.png)
    



```python
def test_dqn(model, epoch=32):
    env = GridWorld(size=4)
    env.reset(seed=33)  # Static board.

    # Initial state, with noise.
    observation = env.observation
    for i in trange(epoch):
        print("Step: {}".format(i))
        print("\n".join(list(map(lambda row: "".join(row), env.render()))))
        print()
        state = torch.Tensor(observation).view(-1, 64)
        with torch.no_grad():
            q_val = model(state)
        action = torch.argmax(q_val, dim=1).item()
        observation, reward, terminated, _truncated, _info = env.step(action)
        if terminated:
            print("DONE")
            break
```


```python
test_dqn(model)
```


      0%|          | 0/32 [00:00<?, ?it/s]


    Step: 0
    ....
    .+-.
    ...P
    ...W
    
    Step: 1
    ....
    .+-.
    ..P.
    ...W
    
    Step: 2
    ....
    .+-.
    .P..
    ...W
    
    DONE



```python
losses = dqn(model, optimizer, epoch=1000, seed=None)  # Random board
plt.plot(np.arange(len(losses)), losses);
```


      0%|          | 0/1000 [00:00<?, ?it/s]



    
![png](03_deep_q_network.v2_files/03_deep_q_network.v2_16_1.png)
    


## Experience Replay


```python
from collections import deque


def dqn_experience_replay(model, optimizer, epochs=1000, gamma=0.9, epsilon=1.0):
    replay = deque(maxlen=1000)
    batch_size = 200
    max_moves = 50
    losses = []
    noise = lambda: torch.rand(1, 64) / 64

    for _ in trange(epochs):
        env = GridWorld(4)
        env.reset(seed=None)  # Random state

        observation = env.observation
        for _ in range(max_moves):
            state = torch.Tensor(observation).view(-1, 64) + noise()
            q_val = model(state)
            if np.random.random() < epsilon:
                action = torch.randint(4, (1,)).item()
            else:
                action = torch.argmax(q_val, dim=1).item()
            observation, reward, terminated, _truncated, _info = env.step(action)
            replay.append(
                (
                    state,
                    action,
                    reward,
                    torch.Tensor(observation).view(-1, 64),
                    terminated,
                )
            )

            if len(replay) > batch_size:
                # Experience replay.
                samples = random.sample(replay, batch_size)
                (
                    state1_batch,
                    action_batch,
                    reward_batch,
                    state2_batch,
                    terminated_batch,
                ) = list(zip(*samples))
                state1_batch = torch.cat(state1_batch)
                action_batch = torch.Tensor(action_batch)
                reward_batch = torch.Tensor(reward_batch)
                state2_batch = torch.cat(state2_batch)
                terminated_batch = torch.Tensor(terminated_batch)
                q_val = model(state1_batch)
                with torch.no_grad():
                    q_val_next = model(state2_batch)
                y_pred = (
                    reward_batch
                    + gamma * (1 - terminated_batch) * torch.max(q_val_next, dim=1)[0]
                )
                y_true = q_val.gather(
                    dim=1, index=action_batch.unsqueeze(dim=1).long()
                ).squeeze()
                loss = loss_fn(y_pred, y_true)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            if terminated:
                break
        if epsilon > 0.1:
            epsilon -= 1 / epochs
    return losses
```


```python
# The first item is the values, the second is the index
torch.max(torch.Tensor([[1, 2, 3], [3, 5, 6]]), dim=1)
```




    torch.return_types.max(
    values=tensor([3., 6.]),
    indices=tensor([2, 2]))




```python
torch.max(torch.Tensor([[1, 2, 3], [3, 5, 6]]), dim=0)
```




    torch.return_types.max(
    values=tensor([3., 5., 6.]),
    indices=tensor([1, 1, 1]))




```python
torch.max(torch.Tensor([[1, 2, 3], [3, 5, 6]]))
```




    tensor(6.)




```python
model = torch.nn.Sequential(
    # Input is our gridworld state, a 1x64 grid representation.
    torch.nn.Linear(64, 150),
    torch.nn.ReLU(),
    torch.nn.Linear(150, 100),
    torch.nn.ReLU(),
    # Output is the expected rewards for each arm: 0: up, 1: down, 2: left, 3: right
    torch.nn.Linear(100, 4),
)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```


```python
losses = dqn_experience_replay(model, optimizer, epochs=5000)
plt.plot(np.arange(len(losses)), losses)
```


      0%|          | 0/5000 [00:00<?, ?it/s]





    [<matplotlib.lines.Line2D at 0x15244b020>]




    
![png](03_deep_q_network.v2_files/03_deep_q_network.v2_23_2.png)
    



```python
def test_dqn_experience_replay(model, epoch=1000, max_moves=50):
    wins = 0
    loss = 0
    limbo = 0
    for _ in trange(epoch):
        env = GridWorld(4)
        env.reset(seed=None)  # Random state

        observation = env.observation
        for _ in range(max_moves):
            state = torch.Tensor(observation).view(-1, 64)
            q_val = model(state)
            action = torch.argmax(q_val, dim=1).item()
            observation, reward, terminated, _truncated, _info = env.step(action)
            if terminated:
                break
        if reward == 10:
            wins += 1
        if reward == -10:
            loss += 1
        if reward == -1:
            print()
            print(np.array(env.render()))
            limbo += 1
    return wins, loss, limbo
```


```python
win, loss, limbo = test_dqn_experience_replay(model, max_moves=50)
win, loss, limbo
```


      0%|          | 0/1000 [00:00<?, ?it/s]


    
    [['.' '.' '.' '.']
     ['.' '+' '.' '.']
     ['.' '-' '.' '.']
     ['.' 'P' 'W' '.']]
    
    [['W' '.' '.' '.']
     ['+' '.' '.' '.']
     ['.' '.' '-' '.']
     ['P' '.' '.' '.']]





    (998, 0, 2)


