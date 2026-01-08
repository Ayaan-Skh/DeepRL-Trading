import sys
sys.path.append(".")

import torch
from src.model import DQN

def test_dqn_model():
    """
    Test oif DQN works perfectly
    """
    print("="*50)
    print("Testing DQN model")
    print("="*50)
    
    state_dim=13
    action_dim=3
    model=DQN(state_dim,action_dim)
    
    print("Model created")
    print(model)
    
    # Count Parameters
    total_params=sum(p.numel() for p in model.parameters())
    # .numel() counts no. of parameters in a tensor
    print(f"Total parameters are:{total_params}")
    
    # Test 1: Single state(batch_size=1)
    print("\n Test 1: Single state forward pass")
    state=torch.randn(1,state_dim) # Random state
    # print(state)
    q_values=model(state)
    
    print(f"Input shape:{state.shape}")    
    print(f"Output shape:{q_values.shape}")    
    print(f"Q values:{q_values}")    
    
    assert q_values.shape==(1,action_dim), "Output shape missmatched!!"
    print("✓ Pass")
    
    
    # Test 2: Batch of states(batch_size=32)
    print(f"\n Test 2: Batch forward pass")
    batches=torch.randn(32,state_dim)
    
    batch_q_values=model(batches)
    
    print(f"Input shape:{batches.shape}")
    print(f"Output shape:{batch_q_values.shape}")
    
    assert batch_q_values.shape == (32,action_dim), "Batch Output shapes mismatch"
    print("✓ Pass")
    
    # Test 3: Gradient flow (Backpropogation works)
    print("\n Test 3: Gradient computation")
    
    #Forward pass
    state=torch.randn(1,state_dim,requires_grad=True)
    print(state)
    q_values=model(state)
    
    # Compute loss
    loss=q_values.sum()
    print(loss)
    # Bckward pass
    loss.backward()
    
    #Check if gradients exist
    has_gradients=all(p.grad is not None for p in model.parameters())
    print(f"All parameters has gradients:{has_gradients}")
    
    assert has_gradients, "Some parameters has no gradients"
    print("✓ Pass")
    
    # Test 4: Action selection
    print("\nTest 4: Action selection from Q-values")
    state = torch.randn(1, state_dim)
    q_values = model(state)
    
    # Select best action
    action = q_values.argmax(dim=1).item()
    action_names = ['HOLD', 'BUY', 'SELL']
    
    print(f"  Q-values: {q_values[0].detach().numpy()}")
    print(f"  Best action: {action} ({action_names[action]})")
    
    assert 0 <= action < action_dim, "Invalid action selected!"
    print("  ✓ Passed")
    
    print("\n" + "="*50)
    print("ALL MODEL TESTS PASSED! ✓")
    print("="*50)

if __name__ == "__main__":
    test_dqn_model()