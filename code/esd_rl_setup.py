# -*- coding: utf-8 -*-
"""esd-rl-setup.ipynb


"""

#pilot - single neuron // did not work well as effect was not readily isolated

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# -------- Environment --------
class DriftEnv:
    def __init__(self, T=100):
        self.T = T

    def reset(self):
        self.t = 0
        self.s = 0.0
        self.drift_start = np.random.randint(30, 60)
        return np.array([self.s], dtype=np.float32)

    def step(self, action):
        noise = np.random.normal(0, 0.03)
        if self.t > self.drift_start:
            self.s += 0.01 + noise
        else:
            self.s += noise

        self.s = np.clip(self.s, 0, 1)
        self.t += 1

        correct = self.t > (self.drift_start + 20)

        reward = 0
        if action == 1 and correct:
            reward = 1
        elif action == 1 and not correct:
            reward = -1

        done = self.t >= self.T
        return np.array([self.s], dtype=np.float32), reward, done


# -------- ESD Dynamics --------
def update_esd(r, z_prev, v_prev):
    alpha_up = 0.15
    alpha_down = 0.4
    beta = 0.6

    if r < z_prev:
        alpha = alpha_down
    else:
        alpha = alpha_up

    z_hat = alpha * r + (1 - alpha) * z_prev
    v = beta * (z_hat - z_prev) + (1 - beta) * v_prev
    z = np.clip(z_hat + v, 0, 1)
    return z, v


# -------- Tiny Policy --------
class Policy:
    def __init__(self):
        self.w = np.random.randn(1)
        self.b = 0.0

    def forward(self, x):
        logits = x * self.w + self.b
        return 1 / (1 + np.exp(-logits))

    def update(self, grads, lr=0.01):
        dw, db = grads
        self.w += lr * dw
        self.b += lr * db


# -------- Training Loop --------
def train(esd_guided=False, episodes=500):
    env = DriftEnv()
    policy = Policy()

    for ep in range(episodes):
        x = env.reset()
        done = False

        xs, zs, rewards = [], [], []
        z, v = 0.0, 0.0

        while not done:
            p = policy.forward(x)[0]
            action = np.random.rand() < p

            xs.append(x[0])
            zs.append(z)

            x, r, done = env.step(int(action))
            rewards.append(r)

            z, v = update_esd(x[0], z, v)

        G = np.cumsum(rewards[::-1])[::-1]

        dw, db = 0.0, 0.0
        for t in range(len(xs)):
            x_t = xs[t]
            p = policy.forward(np.array([x_t]))[0]

            grad = (G[t]) * (1 - p)

            if esd_guided:
                grad += (zs[t] - p)

            dw += grad * x_t
            db += grad

        policy.update((dw, db))

    return policy


# -------- Train --------
policy_plain = train(False)
policy_esd = train(True)

# -------- Visualize rollout --------
env = DriftEnv()
x = env.reset()

ps_plain, ps_esd, signal = [], [], []
z, v = 0.0, 0.0

for t in range(env.T):
    ps_plain.append(policy_plain.forward(x)[0])
    ps_esd.append(policy_esd.forward(x)[0])
    signal.append(x[0])

    x, _, _ = env.step(0)
    z, v = update_esd(x[0], z, v)

plt.figure(figsize=(10,5))
plt.plot(signal, label="Signal")
plt.plot(ps_plain, label="Policy without ESD")
plt.plot(ps_esd, label="Policy with ESD")
plt.legend()
plt.show()



#drifting signal - esd reinforce vs non esd intent to act mapping

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

device = "cpu"

class DriftEnv:
    def __init__(self, T=100):
        self.T = T

    def reset(self):
        self.t = 0
        self.s = 0.0
        self.drift_start = np.random.randint(30, 60)
        return np.array([self.s], dtype=np.float32)

    def step(self, action):
        noise = np.random.normal(0, 0.03)
        if self.t > self.drift_start:
            self.s += 0.01 + noise
        else:
            self.s += noise

        self.s = np.clip(self.s, 0, 1)
        self.t += 1

        correct = self.t > (self.drift_start + 20)

        reward = 1 if (action==1 and correct) else -1 if (action==1) else 0
        done = self.t >= self.T
        return np.array([self.s], dtype=np.float32), reward, done


def update_esd(r, z_prev, v_prev):
    alpha_up, alpha_down, beta = 0.15, 0.4, 0.6
    alpha = alpha_down if r < z_prev else alpha_up
    z_hat = alpha*r + (1-alpha)*z_prev
    v = beta*(z_hat-z_prev) + (1-beta)*v_prev
    return np.clip(z_hat+v,0,1), v

class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1,32),
            nn.Tanh(),
            nn.Linear(32,1),
            nn.Sigmoid()
        )

    def forward(self,x):
        return self.net(x)

def train(esd=False, episodes=800):
    env = DriftEnv()
    policy = Policy()
    opt = optim.Adam(policy.parameters(), lr=1e-3)

    for ep in range(episodes):
        x = env.reset()
        done=False

        logps, rewards, ps, zs = [], [], [], []
        z,v = 0,0

        while not done:
            xt = torch.tensor(x)
            p = policy(xt)
            dist = torch.distributions.Bernoulli(p)
            a = dist.sample()

            logps.append(dist.log_prob(a))
            ps.append(p)
            zs.append(z)

            x,r,done = env.step(int(a.item()))
            rewards.append(r)
            z,v = update_esd(x[0],z,v)

        # returns
        G, returns = 0, []
        for r in reversed(rewards):
            G = r + 0.99*G
            returns.insert(0,G)
        returns = torch.tensor(returns)

        logps = torch.stack(logps)
        ps = torch.stack(ps).squeeze()
        zs = torch.tensor(zs, dtype=torch.float32)

        loss = -(logps*returns).mean()

        if esd:
            esd_loss = ((ps - zs)**2).mean()
            loss += 2.0 * esd_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

    return policy

p_plain = train(False)
p_esd = train(True)

env = DriftEnv()
x = env.reset()
signal, a_plain, a_esd = [], [], []

for _ in range(env.T):
    xt = torch.tensor(x)
    signal.append(x[0])
    a_plain.append(p_plain(xt).item())
    a_esd.append(p_esd(xt).item())
    x,_,_ = env.step(0)

plt.figure(figsize=(10,5))
plt.plot(signal, label="signal")
plt.plot(a_plain, label="plain")
plt.plot(a_esd, label="esd")
plt.legend()
plt.show()



#hovering signal - esd reinforce vs non esd intent to act mapping

class HoverEnv:
    def __init__(self, T=100):
        self.T = T

    def reset(self):
        self.t = 0
        self.s = 0.6 + np.random.normal(0, 0.02)  # start near threshold
        self.real_drift_start = np.random.randint(40, 70)
        return np.array([self.s], dtype=np.float32)

    def step(self, action):
        # jitter around threshold
        noise = np.random.normal(0, 0.03)

        if self.t > self.real_drift_start:
            # real sustained crossing
            self.s += 0.015 + noise
        else:
            self.s += noise

        self.s = np.clip(self.s, 0, 1)
        self.t += 1

        # correct only if sustained crossing has occurred
        correct = self.t > (self.real_drift_start + 15)

        reward = 1 if (action==1 and correct) else -1 if (action==1) else 0
        done = self.t >= self.T
        return np.array([self.s], dtype=np.float32), reward, done

def train(esd=False, episodes=800):
    env = HoverEnv()
    policy = Policy()
    opt = optim.Adam(policy.parameters(), lr=1e-3)

    for ep in range(episodes):
        x = env.reset()
        done=False

        logps, rewards, ps, zs = [], [], [], []
        z,v = 0,0

        while not done:
            xt = torch.tensor(x)
            p = policy(xt)
            dist = torch.distributions.Bernoulli(p)
            a = dist.sample()

            logps.append(dist.log_prob(a))
            ps.append(p)
            zs.append(z)

            x,r,done = env.step(int(a.item()))
            rewards.append(r)
            z,v = update_esd(x[0],z,v)

        # returns
        G, returns = 0, []
        for r in reversed(rewards):
            G = r + 0.99*G
            returns.insert(0,G)
        returns = torch.tensor(returns)

        logps = torch.stack(logps)
        ps = torch.stack(ps).squeeze()
        zs = torch.tensor(zs, dtype=torch.float32)

        loss = -(logps*returns).mean()

        if esd:
            esd_loss = ((ps - zs)**2).mean()
            loss += 2.0 * esd_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

    return policy

p_plain = train(False)
p_esd = train(True)

env = DriftEnv()
x = env.reset()
signal, a_plain, a_esd = [], [], []

for _ in range(env.T):
    xt = torch.tensor(x)
    signal.append(x[0])
    a_plain.append(p_plain(xt).item())
    a_esd.append(p_esd(xt).item())
    x,_,_ = env.step(0)

plt.figure(figsize=(10,5))
plt.plot(signal, label="signal")
plt.plot(a_plain, label="plain")
plt.plot(a_esd, label="esd")
plt.legend()
plt.show()



#act in a window - esd reinforce vs non esd intent to act mapping

class WindowEnv:
    def __init__(self, T=100):
        self.T = T

    def reset(self):
        self.t = 0
        self.s = 0.0
        return np.array([self.s], dtype=np.float32)

    def step(self, action):
        # slow steady rise
        noise = np.random.normal(0, 0.02)
        self.s += 0.012 + noise
        self.s = np.clip(self.s, 0, 1)
        self.t += 1

        # correct action window
        window_start = 55
        window_end = 70

        correct = window_start <= self.t <= window_end

        if action == 1 and correct:
            reward = 1
        elif action == 1 and not correct:
            reward = -1
        else:
            reward = 0

        done = self.t >= self.T
        return np.array([self.s], dtype=np.float32), reward, done

def train(esd=False, episodes=800):
    env = WindowEnv()
    policy = Policy()
    opt = optim.Adam(policy.parameters(), lr=1e-3)

    for ep in range(episodes):
        x = env.reset()
        done=False

        logps, rewards, ps, zs = [], [], [], []
        z,v = 0,0

        while not done:
            xt = torch.tensor(x)
            p = policy(xt)
            dist = torch.distributions.Bernoulli(p)
            a = dist.sample()

            logps.append(dist.log_prob(a))
            ps.append(p)
            zs.append(z)

            x,r,done = env.step(int(a.item()))
            rewards.append(r)
            z,v = update_esd(x[0],z,v)

        # returns
        G, returns = 0, []
        for r in reversed(rewards):
            G = r + 0.99*G
            returns.insert(0,G)
        returns = torch.tensor(returns)

        logps = torch.stack(logps)
        ps = torch.stack(ps).squeeze()
        zs = torch.tensor(zs, dtype=torch.float32)

        loss = -(logps*returns).mean()

        if esd:
            esd_loss = ((ps - zs)**2).mean()
            loss += 2.0 * esd_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

    return policy

p_plain = train(False)
p_esd = train(True)

env = DriftEnv()
x = env.reset()
signal, a_plain, a_esd = [], [], []

for _ in range(env.T):
    xt = torch.tensor(x)
    signal.append(x[0])
    a_plain.append(p_plain(xt).item())
    a_esd.append(p_esd(xt).item())
    x,_,_ = env.step(0)

plt.figure(figsize=(10,5))
plt.plot(signal, label="signal")
plt.plot(a_plain, label="plain")
plt.plot(a_esd, label="esd")
plt.legend()
plt.show()

#

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

device = "cpu"
RUNS = 40
EPISODES = 800
T = 100

# ---------------- ESD ----------------
def update_esd(r, z_prev, v_prev):
    alpha_up, alpha_down, beta = 0.15, 0.4, 0.6
    alpha = alpha_down if r < z_prev else alpha_up
    z_hat = alpha*r + (1-alpha)*z_prev
    v = beta*(z_hat-z_prev) + (1-beta)*v_prev
    return np.clip(z_hat+v,0,1), v

# ---------------- Policy ----------------
class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1,32),
            nn.Tanh(),
            nn.Linear(32,1),
            nn.Sigmoid()
        )

    def forward(self,x):
        return self.net(x)

# ---------------- Training ----------------
def train(env_class, esd=False):
    policy = Policy()
    opt = optim.Adam(policy.parameters(), lr=1e-3)

    for ep in range(EPISODES):
        env = env_class()
        x = env.reset()
        done=False

        logps, rewards, ps, zs = [], [], [], []
        z,v = 0,0

        while not done:
            xt = torch.tensor(x)
            p = policy(xt)
            dist = torch.distributions.Bernoulli(p)
            a = dist.sample()

            logps.append(dist.log_prob(a))
            ps.append(p)
            zs.append(z)

            x,r,done = env.step(int(a.item()))
            rewards.append(r)
            z,v = update_esd(x[0],z,v)

        G, returns = 0, []
        for r in reversed(rewards):
            G = r + 0.99*G
            returns.insert(0,G)
        returns = torch.tensor(returns)

        logps = torch.stack(logps)
        ps = torch.stack(ps).squeeze()
        zs = torch.tensor(zs, dtype=torch.float32)

        loss = -(logps*returns).mean()
        if esd:
            loss += 2.0*((ps-zs)**2).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

    return policy

# ---------------- Metrics ----------------
def jerk(p):
    return np.max(np.abs(np.diff(p)))

def oscillations(p, thresh=0.5):
    flips = 0
    for i in range(1,len(p)):
        if (p[i-1] > thresh) != (p[i] > thresh):
            flips += 1
    return flips

def timing(p, thresh=0.6):
    for i,v in enumerate(p):
        if v > thresh:
            return i
    return len(p)

# ---------------- Environments ----------------
class DriftEnv:
    def __init__(self): self.T=T
    def reset(self):
        self.t=0; self.s=0.0
        self.drift_start=np.random.randint(30,60)
        return np.array([self.s],dtype=np.float32)
    def step(self,a):
        noise=np.random.normal(0,0.03)
        if self.t>self.drift_start: self.s+=0.01+noise
        else: self.s+=noise
        self.s=np.clip(self.s,0,1); self.t+=1
        correct=self.t>(self.drift_start+20)
        r=1 if (a==1 and correct) else -1 if (a==1) else 0
        return np.array([self.s],dtype=np.float32),r,self.t>=self.T

class HoverEnv:
    def __init__(self): self.T=T
    def reset(self):
        self.t=0; self.s=0.6+np.random.normal(0,0.02)
        self.real=np.random.randint(40,70)
        return np.array([self.s],dtype=np.float32)
    def step(self,a):
        noise=np.random.normal(0,0.03)
        if self.t>self.real: self.s+=0.015+noise
        else: self.s+=noise
        self.s=np.clip(self.s,0,1); self.t+=1
        correct=self.t>(self.real+15)
        r=1 if (a==1 and correct) else -1 if (a==1) else 0
        return np.array([self.s],dtype=np.float32),r,self.t>=self.T

class WindowEnv:
    def __init__(self): self.T=T
    def reset(self):
        self.t=0; self.s=0.0
        return np.array([self.s],dtype=np.float32)
    def step(self,a):
        noise=np.random.normal(0,0.02)
        self.s+=0.012+noise
        self.s=np.clip(self.s,0,1); self.t+=1
        correct=55<=self.t<=70
        r=1 if (a==1 and correct) else -1 if (a==1) else 0
        return np.array([self.s],dtype=np.float32),r,self.t>=self.T

# ---------------- Evaluation ----------------
def evaluate(env_class):
    print(f"\nRunning {env_class.__name__}")
    p_plain = train(env_class, False)
    p_esd = train(env_class, True)

    curves_plain, curves_esd = [], []
    jerks_plain, jerks_esd = [], []
    osc_plain, osc_esd = [], []
    time_plain, time_esd = [], []

    for run in range(RUNS):
        if run%10==0:
            print(f"  rollout {run}/{RUNS}")

        env = env_class()
        x = env.reset()
        done=False
        p1, p2 = [], []

        while not done:
            xt=torch.tensor(x)
            p1.append(p_plain(xt).item())
            p2.append(p_esd(xt).item())
            x,_,done = env.step(0)

        curves_plain.append(p1)
        curves_esd.append(p2)
        jerks_plain.append(jerk(p1))
        jerks_esd.append(jerk(p2))
        osc_plain.append(oscillations(p1))
        osc_esd.append(oscillations(p2))
        time_plain.append(timing(p1))
        time_esd.append(timing(p2))

    curves_plain = np.array(curves_plain)
    curves_esd = np.array(curves_esd)

    # ---- Plot mean +- std ----
    t = np.arange(T)
    plt.figure(figsize=(10,5))
    plt.plot(curves_plain.mean(0), label="plain")
    plt.fill_between(t,
                     curves_plain.mean(0)-curves_plain.std(0),
                     curves_plain.mean(0)+curves_plain.std(0),
                     alpha=0.2)
    plt.plot(curves_esd.mean(0), label="esd")
    plt.fill_between(t,
                     curves_esd.mean(0)-curves_esd.std(0),
                     curves_esd.mean(0)+curves_esd.std(0),
                     alpha=0.2)
    plt.legend()
    plt.title(env_class.__name__)
    plt.show()

    print("Jerk:", np.mean(jerks_plain), "vs", np.mean(jerks_esd))
    print("Osc:", np.mean(osc_plain), "vs", np.mean(osc_esd))
    print("Timing var:", np.std(time_plain), "vs", np.std(time_esd))


# ---------------- Run all three ----------------
evaluate(DriftEnv)
evaluate(HoverEnv)
evaluate(WindowEnv)
