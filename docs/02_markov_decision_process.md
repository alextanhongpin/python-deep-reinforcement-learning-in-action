# Markov Decision Process

## Epsilon-greedy strategy for action selection


```python
import random

import matplotlib.pyplot as plt
import numpy as np
```

Defining the reward function


```python
class MovingAverage:
    def __init__(self):
        self.count = 0
        self.value = 0

    def add(self, value):
        num = self.count * self.value + value
        den = self.count + 1
        self.count = self.count + 1
        self.value = num / den
        return self.value
```


```python
ma = MovingAverage()
for _ in range(10):
    print(ma.add(random.randint(0, 5) + 5))
```

    8.0
    6.5
    6.333333333333333
    7.25
    6.8
    7.0
    7.142857142857143
    7.375
    7.111111111111111
    7.3



```python
class Bandit:
    def __init__(self, n_arms=10, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.n_arms = n_arms
        self.probs = np.random.random(n_arms)
        self.pulls = np.zeros(self.n_arms)
        self.rewards = np.zeros(self.n_arms)

    def reward(self, prob, n=10):
        reward = 0
        for _ in range(n):
            reward += random.random() < prob
        return reward

    def policy(self):
        return random.randrange(0, self.n_arms)

    def pull(self):
        arm = self.policy()
        reward = self.reward(self.probs[arm])
        pulls = self.pulls[arm]
        mean_reward = self.rewards[arm]
        mean_reward = (pulls * mean_reward + reward) / (pulls + 1)
        self.rewards[arm] = mean_reward
        self.pulls[arm] = pulls + 1

        return mean_reward


class GreedyBandit(Bandit):
    def policy(self, e_greedy=0.2):
        if random.random() < e_greedy:
            arm = random.randrange(0, self.n_arms)
        else:
            arm = np.argmax(self.rewards)
        return arm
```


```python
bandit = Bandit(10, seed=42)
trials = 1000
rewards = []
ma = MovingAverage()
for i in range(trials):
    reward = bandit.pull()
    rewards.append(ma.add(reward))

fig, ax = plt.subplots()
ax.scatter(range(trials), rewards)
ax.set_title("Bandit selection")
ax.set_xlabel("Plays")
ax.set_ylabel("Average rewards");
```


    
![png](02_markov_decision_process_files/02_markov_decision_process_7_0.png)
    



```python
bandit = GreedyBandit(10, seed=42)
trials = 1000
rewards = []
ma = MovingAverage()
for i in range(trials):
    reward = bandit.pull()
    rewards.append(ma.add(reward))

fig, ax = plt.subplots()
ax.scatter(range(trials), rewards)
ax.set_title("GreedyBandit selection")
ax.set_xlabel("Plays")
ax.set_ylabel("Average rewards");
```


    
![png](02_markov_decision_process_files/02_markov_decision_process_8_0.png)
    


## The softmax function


```python
def softmax(values, tau=1.12):
    num = np.exp(values / tau)
    den = np.sum(num)
    return num / den
```


```python
from collections import Counter

x = np.arange(10)
p = softmax(x)
n = 1000

choices = []
for _ in range(n):
    choices.append(np.random.choice(x, p=p))
Counter(choices).most_common(5)
```




    [(9, 576), (8, 240), (7, 115), (6, 38), (5, 23)]




```python
softmax(np.zeros(10))
```




    array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])




```python
np.round(softmax(np.arange(10)), 3)
```




    array([0.   , 0.   , 0.001, 0.003, 0.007, 0.017, 0.041, 0.099, 0.242,
           0.591])




```python
np.sum(softmax(np.arange(10)))
```




    0.9999999999999999




```python
class SoftmaxBandit(Bandit):
    def policy(self, tau=1.12):
        arms = np.arange(self.n_arms)
        p = softmax(self.rewards, tau=tau)
        arm = np.random.choice(arms, p=p)
        return arm
```


```python
bandit = SoftmaxBandit(10, seed=42)
trials = 1000
rewards = []
ma = MovingAverage()
for i in range(trials):
    reward = bandit.pull()
    rewards.append(ma.add(reward))

fig, ax = plt.subplots()
ax.scatter(range(trials), rewards)
ax.set_title("SoftmaxBandit selection")
ax.set_xlabel("Plays")
ax.set_ylabel("Average rewards");
```


    
![png](02_markov_decision_process_files/02_markov_decision_process_16_0.png)
    

