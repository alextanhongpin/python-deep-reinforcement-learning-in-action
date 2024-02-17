import gym
import torch
from torch import nn, optim
from torch.nn import functional as F


class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.l1 = nn.Linear(4, 25)
        self.l2 = nn.Linear(25, 50)
        self.actor_lin1 = nn.Linear(50, 2)
        self.l3 = nn.Linear(50, 25)
        self.critic_lin1 = nn.Linear(25, 1)

    def forward(self, x):
        x = F.normalize(x, dim=0)
        y = F.relu(self.l1(x))
        y = F.relu(self.l2(y))

        # The actor head returns the log probabilities over the 2 actions.
        actor = F.log_softmax(self.actor_lin1(y), dim=0)

        c = F.relu(self.l3(y.detach()))
        # The critic returns a single number bounded by -1 and 1.
        critic = torch.tanh(self.critic_lin1(c))

        return actor, critic


def run_episode(worker_env, worker_model):
    state = torch.from_numpy(worker_env.env.state).float()
    values, logprobs, rewards = [], [], []
    done = False
    j = 0
    while not done:
        j += 1
        policy, value = worker_model(state)
        values.append(value)
        logits = policy.view(-1)

        action_dist = torch.distributions.Categorical(logits=logits)
        action = action_dist.sample()

        logprob_ = policy.view(-1)[action]
        logprobs.append(logprob_)
        state_, _, done, _, info = worker_env.step(action.detach().numpy())
        state = torch.from_numpy(state_).float()
        if done:
            reward = -10
            worker_env.reset()
        else:
            reward = 1.0
        rewards.append(reward)
    return values, logprobs, rewards


def update_params(worker_opt, values, logprobs, rewards, clc=0.1, gamma=0.95):
    # We reverse the order of the rewards, logprobs and values_ array and call .view(-1)
    # to make sure they are flat.
    rewards = torch.Tensor(rewards).flip(dims=(0,)).view(-1)
    logprobs = torch.stack(logprobs).flip(dims=(0,)).view(-1)
    values = torch.stack(values).flip(dims=(0,)).view(-1)

    returns = []
    ret_ = torch.Tensor([0])
    for r in range(rewards.shape[0]):
        ret_ = rewards[r] + gamma * ret_
        returns.append(ret_)
    returns = torch.stack(returns).view(-1)
    returns = F.normalize(returns, dim=0)

    # We need to detach the values tensor from the graph to prevent
    # backpropagating through the critic head.
    actor_loss = -1 * logprobs * (returns - values.detach())

    # The critic attempts to learn to predict the return.
    critic_loss = torch.pow(values - returns, 2)

    # We sum the actor and critic losses to get an overall loss.
    # We scale down the critic loss by the clc factor.
    loss = actor_loss.sum() + clc * critic_loss.sum()
    loss.backward()
    worker_opt.step()
    return actor_loss, critic_loss, len(rewards)


def worker(t, worker_model, counter, params):
    worker_env = gym.make("CartPole-v1")
    worker_env.reset()
    worker_opt = optim.Adam(lr=1e-4, params=worker_model.parameters())
    worker_opt.zero_grad()
    for i in range(params["epochs"]):
        worker_opt.zero_grad()
        values, logprobs, rewards = run_episode(worker_env, worker_model)
        actor_loss, critic_loss, eplen = update_params(
            worker_opt, values, logprobs, rewards
        )
        counter.value = counter.value + 1
