from __future__ import annotations

import numpy as np
import torch
import torch.optim as optim

from dqn.agent import DQNAgent
from dqn.replay_buffer import ReplayBuffer
from dqn.q_network import QNetwork, get_device


def run_smoke_test() -> None:
    print("=== STARTING DQN SMOKE TEST ===")

    # 1. Test Device Selector
    device = get_device()
    print(f"Device selected: {device}")

    # 2. Test QNetwork instantiation & forward pass
    print("Testing QNetwork...")
    state_dim = 4
    action_dim = 3
    net = QNetwork(state_dim, action_dim).to(device)
    
    test_state = np.array([0.1, -0.2, 0.5, 0.4], dtype=np.float32)
    state_tensor = torch.tensor(test_state, dtype=torch.float32, device=device).unsqueeze(0)
    
    with torch.no_grad():
        q_values = net(state_tensor)
    print(f"Input state: {test_state}")
    print(f"Output Q-values: {q_values.cpu().numpy()[0]}")
    assert q_values.shape == (1, action_dim), f"Expected shape (1, {action_dim}), got {q_values.shape}"
    print("QNetwork test passed!\n")

    # 3. Test Replay Buffer insertion and sampling
    print("Testing ReplayBuffer...")
    buffer = ReplayBuffer(capacity=10)
    
    for i in range(5):
        s = np.random.randn(state_dim).astype(np.float32)
        a = np.random.randint(0, action_dim)
        r = float(np.random.randn())
        s_next = np.random.randn(state_dim).astype(np.float32)
        done = bool(np.random.choice([True, False]))
        buffer.push(s, a, r, s_next, done)
        
    print(f"Buffer size: {len(buffer)}")
    assert len(buffer) == 5, f"Expected size 5, got {len(buffer)}"
    
    states, actions, rewards, next_states, terminateds = buffer.sample(batch_size=2, device=device)
    print(f"Sampled states shape: {states.shape}")
    print(f"Sampled actions shape: {actions.shape}")
    print(f"Sampled rewards shape: {rewards.shape}")
    print(f"Sampled next_states shape: {next_states.shape}")
    print(f"Sampled terminateds shape: {terminateds.shape}")
    
    assert states.shape == (2, state_dim)
    assert actions.shape == (2, 1)
    assert rewards.shape == (2, 1)
    assert next_states.shape == (2, state_dim)
    assert terminateds.shape == (2, 1)
    print("ReplayBuffer test passed!\n")

    # 4. Test DQNAgent action selection & optimization step
    print("Testing DQNAgent...")
    agent = DQNAgent(state_dim, action_dim, lr=0.001, gamma=0.95, target_update_interval=2)
    
    # Test action selection (greedy and epsilon-greedy)
    greedy_action = agent.select_action(test_state, epsilon=0.0)
    exploratory_action = agent.select_action(test_state, epsilon=1.0)
    print(f"Greedy action: {greedy_action}")
    print(f"Exploratory action: {exploratory_action}")
    
    assert 0 <= greedy_action < action_dim
    assert 0 <= exploratory_action < action_dim

    # Populate buffer to allow optimization
    agent_buffer = ReplayBuffer(capacity=100)
    for _ in range(70):
        s = np.random.randn(state_dim).astype(np.float32)
        a = np.random.randint(0, action_dim)
        r = float(np.random.randn())
        s_next = np.random.randn(state_dim).astype(np.float32)
        done = bool(np.random.choice([True, False]))
        agent_buffer.push(s, a, r, s_next, done)

    # Perform optimization
    initial_steps = agent.optimization_steps
    loss = agent.optimize_model(agent_buffer, batch_size=32)
    print(f"Optimization step completed. Loss: {loss}")
    
    assert loss is not None, "Expected loss to be calculated"
    assert agent.optimization_steps == initial_steps + 1, "Expected optimization steps to increment"
    
    # Test target network update
    agent.update_target_network()
    # Check if weights match
    for p_online, p_target in zip(agent.online_net.parameters(), agent.target_net.parameters()):
        assert torch.allclose(p_online, p_target), "Target network parameters do not match online network after update"
    
    print("DQNAgent test passed!\n")
    print("=== ALL SMOKE TESTS PASSED SUCCESSFULLY ===")


if __name__ == "__main__":
    run_smoke_test()
