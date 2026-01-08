import sys
sys.path.append('.')

from src.train import train_agent

if __name__ == "__main__":
    print("Starting DQN training...")
    print("This will take ~30-60 minutes for 500 episodes.")
    print("Watch the training progress below!\n")
    
    # Train with default settings
    agent, metrics = train_agent(
        num_episodes=500,
        initial_balance=1000000
    )
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Check 'results/training_curves/training_progress.png' for plots")
    print(f"Best model saved at 'models/checkpoints/best_agent.pth'")