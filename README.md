**CartPole with Deep Q-Networks (DQN) Report**

**Project Overview**

This report summarizes the implementation and evaluation of a Deep Q-Network (DQN) to solve the CartPole reinforcement learning problem. The primary goal was to train an agent to balance a pole on a cart by taking actions to move the cart left or right.
 The implementation incorporates various critical reinforcement learning techniques, including Q-learning, epsilon-greedy exploration, and neural network-based function approximation.

---

### **1. Problem Description**

The CartPole environment requires the agent to balance a pole attached to a cart by applying a force to move the cart left or right. The environment provides:

- **State space**: Represented as a vector of four continuous values (angular position, angular velocity, cart position, cart velocity).
- **Action space**: Two discrete actions (move left or move right).
- **Reward function**: A reward of 1 for every time step the pole remains upright. The episode terminates when the pole falls or 500 time steps are reached.

The task is considered solved when the agent achieves an average return of 500 over several episodes.

---

### **2. Methodology**

#### **2.1 Q-Learning and Deep Q-Network**

The Q-learning algorithm learns an optimal state-action value function \( Q^*(s, a) \) using the Bellman equation. A neural network approximates the Q-function, where:

- **Inputs**: State vector.
- **Outputs**: Q-values for each action.

The network is trained to minimize the squared error between predicted Q-values and targets derived from the Bellman equation.

#### **2.2 Exploration vs. Exploitation**

To ensure adequate exploration, an epsilon-greedy policy was implemented. The exploration rate (epsilon) decays linearly during training, encouraging exploration initially and exploitation of learned policies later.

#### **2.3 Replay Buffer**

A replay buffer stores experiences (state, action, reward, next state, done). During training, random mini-batches are sampled to break correlations between consecutive experiences, stabilizing the learning process.

#### **2.4 Target Network**

A separate target network provides stable Q-value targets, updated periodically to reduce oscillations in training.

#### **2.5 Optimization**

The neural network parameters are updated using gradients computed via backpropagation. Optax's Adam optimizer was used with a learning rate of 3e-4.

---

### **3. Implementation Details**

#### **3.1 Neural Network Architecture**

- **Input Layer**: Flattened state vector.
- **Hidden Layers**: Two fully connected layers with 20 units each.
- **Output Layer**: Two units (Q-values for the two actions).

#### **3.2 Training Process**

- Episodes: 1001 episodes.
- Replay Buffer Size: 10,000 transitions.
- Batch Size: 512.
- Discount Factor (γ): 0.99.
- Epsilon Decay: From 1.0 to 0.1 over 3000 steps.

---

### **4. Results**

#### **4.1 Training Performance**

The agent’s performance improved steadily, achieving the maximum reward (500) after approximately 800 episodes.

**Figure 1**: Plot of episode returns over training episodes.

![Training Performance](/visuals/DQN_Train.png)

#### **4.2 Policy Visualization**

Episodes where the agent balanced the pole for 500 time steps demonstrated the learned policy’s effectiveness.

**Video 2**: Visualization of the agent’s actions during evaluation episodes.

- [Episode 792 Evaluation Video](visuals/policy.mp4)

---

### **5. Observations**

- **Convergence**: The agent successfully learned to solve the CartPole environment, consistently achieving the maximum reward.
- **Exploration Efficiency**: The epsilon-greedy strategy ensured adequate exploration initially, enabling the agent to discover optimal actions.
- **Replay Buffer Effectiveness**: Training stability was significantly enhanced by randomizing experiences.
- **Target Network Benefits**: Reduced instability caused by bootstrapping during Q-value updates.

---

### **6. Future Directions**

- Extend the implementation to more complex environments like LunarLander or Atari games.
- Experiment with alternative architectures, such as convolutional neural networks, for image-based environments.
- Investigate advanced exploration strategies (e.g., curiosity-driven exploration).
- Compare DQN with advanced algorithms like Double DQN or Dueling DQN.

---

### **7. Conclusion**

This project demonstrated the effectiveness of Deep Q-Learning in solving a classic reinforcement learning problem. Through the integration of neural networks, exploration strategies, and replay buffers, the agent successfully mastered the CartPole environment, achieving near-optimal performance.

---

### **Appendices**

#### **A. Hyperparameters**
- Learning Rate: 3e-4
- Batch Size: 512
- Replay Buffer Size: 10,000
- Epsilon Decay Steps: 3000
- Discount Factor: 0.99

#### **B. Code Excerpts**

Sample code for critical functions like the Q-learning loss calculation and epsilon-greedy action selection can be included here if needed.

---



## Project Structure

- `src/`: Contains the Python scripts implementing the DQN.
- `visuals/`: Contains training plots and evaluation videos.
- `requirements.txt`: Lists the Python dependencies.
- `.gitignore`: Specifies files and folders to exclude from Git.

## Visuals

Plots and videos are stored in the `visuals/` directory.

## Instructions

1. Install the dependencies:

```bash
pip install -r requirements.txt
```

2. Run the main script in the `src/` directory:

```bash
python cartpole.py
```


3. Visualize results in the `visuals/` directory.

---

For further details, check the code and comments in the `src/` directory.
**End of Report**