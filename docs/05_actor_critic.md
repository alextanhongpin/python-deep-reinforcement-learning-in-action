# Actor-critic


```python
import gym
import torch
import torch.multiprocessing as mp
```


```python
from worker.worker import ActorCritic, worker

if __name__ == "__main__":
    MasterNode = ActorCritic()

    # The share_memory() will allow the parameters of the model to be shared
    # across processes rather than being copied.
    MasterNode.share_memory()

    processes = []
    params = {"epochs": 1000, "n_workers": 2}
    counter = mp.Value("i", 0)
    for i in range(params["n_workers"]):
        p = mp.Process(target=worker, args=(i, MasterNode, counter, params))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    for p in processes:
        p.terminate()

    print(counter.value, processes[1].exitcode)
```

    /Users/alextanhongpin/Documents/python/python-deep-reinforcement-learning-in-action/.venv/lib/python3.11/site-packages/gym/utils/passive_env_checker.py:233: DeprecationWarning: `np.bool8` is a deprecated alias for `np.bool_`.  (Deprecated NumPy 1.24)
      if not isinstance(terminated, (bool, np.bool8)):
    /Users/alextanhongpin/Documents/python/python-deep-reinforcement-learning-in-action/.venv/lib/python3.11/site-packages/gym/utils/passive_env_checker.py:233: DeprecationWarning: `np.bool8` is a deprecated alias for `np.bool_`.  (Deprecated NumPy 1.24)
      if not isinstance(terminated, (bool, np.bool8)):


    2000 0



```python
import numpy as np


def evaluate(model):
    env = gym.make("CartPole-v1")

    curr_state, _ = env.reset()
    transitions = []  # (state, action, rewards)
    MAX_DUR = 500

    for t in range(MAX_DUR):
        policy, value = model(torch.from_numpy(curr_state).float())
        logits = policy.view(-1)

        action_dist = torch.distributions.Categorical(logits=logits)
        action = action_dist.sample()

        prev_state = curr_state
        curr_state, _, done, truncated, info = env.step(action.detach().numpy())
        transitions.append((prev_state, action, t + 1))
        if done:
            break
    return len(transitions)
```


```python
evaluate(MasterNode)
```




    58




```python
from worker.worker_n_step import ActorCritic, worker

if __name__ == "__main__":
    MasterNode = ActorCritic()

    # The share_memory() will allow the parameters of the model to be shared
    # across processes rather than being copied.
    MasterNode.share_memory()

    processes = []
    params = {"epochs": 5000, "n_workers": 4}
    counter = mp.Value("i", 0)
    for i in range(params["n_workers"]):
        p = mp.Process(target=worker, args=(i, MasterNode, counter, params))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    for p in processes:
        p.terminate()

    print(counter.value, processes[1].exitcode)
```

    /Users/alextanhongpin/Documents/python/python-deep-reinforcement-learning-in-action/.venv/lib/python3.11/site-packages/gym/utils/passive_env_checker.py:233: DeprecationWarning: `np.bool8` is a deprecated alias for `np.bool_`.  (Deprecated NumPy 1.24)
      if not isinstance(terminated, (bool, np.bool8)):
    /Users/alextanhongpin/Documents/python/python-deep-reinforcement-learning-in-action/.venv/lib/python3.11/site-packages/gym/utils/passive_env_checker.py:233: DeprecationWarning: `np.bool8` is a deprecated alias for `np.bool_`.  (Deprecated NumPy 1.24)
      if not isinstance(terminated, (bool, np.bool8)):
    /Users/alextanhongpin/Documents/python/python-deep-reinforcement-learning-in-action/.venv/lib/python3.11/site-packages/gym/utils/passive_env_checker.py:233: DeprecationWarning: `np.bool8` is a deprecated alias for `np.bool_`.  (Deprecated NumPy 1.24)
      if not isinstance(terminated, (bool, np.bool8)):
    /Users/alextanhongpin/Documents/python/python-deep-reinforcement-learning-in-action/.venv/lib/python3.11/site-packages/gym/utils/passive_env_checker.py:233: DeprecationWarning: `np.bool8` is a deprecated alias for `np.bool_`.  (Deprecated NumPy 1.24)
      if not isinstance(terminated, (bool, np.bool8)):


    19877 0



```python
values = []
for i in range(100):
    value = evaluate(MasterNode)
    values.append(value)
np.mean(values)
```




    408.7


