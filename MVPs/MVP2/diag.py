"""Check if agent gets stuck after many steps."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from evola_pygame import EvolaPygameP2

app = EvolaPygameP2()
app.cmd_start()
app.cmd_spawn("stuck")
agent = app.agents["stuck"]

positions = []
action_history = []
for step in range(1000):
    obs = app.world.get_obs("stuck")
    action = int(agent.act(obs))
    action_history.append(action)
    _, info = app.world.step(action, agent_id="stuck")
    agent.observe(info)
    positions.append(app.world.get_agent_pos("stuck"))

# Check last 50 steps
last_positions = positions[-50:]
last_actions = action_history[-50:]
unique_last = len(set(last_positions))
action_counts = {i: last_actions.count(i) for i in range(6)}

print(f"Total steps: 500")
print(f"Last 50 steps: {unique_last} unique positions")
print(f"Last 50 actions: {action_counts}")

if unique_last <= 3:
    print("\nSTUCK: Agent barely moves in last 50 steps!")
    if action_counts.get(5, 0) > 30:
        print("  -> WAIT collapsed (action=5 dominates)")
    elif action_counts.get(4, 0) > 30:
        print("  -> EAT collapsed (action=4 dominates)")
    
    # Check if surrounded
    pos = positions[-1]
    obs = app.world.get_obs("stuck")
    walls = (obs == 1).sum()
    print(f"  Walls in view: {walls}/25 at pos {pos}")
else:
    print("\nOK: Agent still moving")

print(f"Energy: {agent.homeo.energy.value:.2f}")
print(f"Visited: {len(agent.visited)}")

# Check for NaN in LTC hidden state
arb = agent.arbiter
if arb.h is not None:
    import torch
    has_nan = torch.isnan(arb.h).any().item()
    print(f"LTC hidden NaN: {has_nan}")
