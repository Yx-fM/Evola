"""Check MVP2 agent memory state after a run."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from evola_pygame import EvolaPygameP2

app = EvolaPygameP2()
app.cmd_start()
app.cmd_spawn("test")
app.cmd_run(300)
app._show_iq = True

agent = app.agents["test"]
h = agent.homeo
arb = agent.arbiter

print(f"\n=== Agent: {agent.agent_id} ===")
print(f"Steps: {agent.step_count}")
print(f"Energy: {h.energy.value:.2f}  Novelty: {h.novelty.value:.2f}  Safety: {h.safety.value:.2f}")
print(f"Visited: {len(agent.visited)}/{h.novelty.total_cells}")

if hasattr(arb, 'memory') and arb.memory:
    m = arb.memory
    print(f"\n=== Memory ===")
    print(f"STM: {len(m.stm)} items")
    print(f"LTM: {len(m.ltm)} items")
    print(f"Memory load: {m.get_memory_load()*100:.0f}%")
    if m.stm:
        print(f"Recent STM events (last 5):")
        for item in m.stm[-5:]:
            print(f"  Step{item.step:5d} pos={item.pos} {item.event_type:12s} imp={item.importance:.1f} decay={item.decay:.2f}")
    if m.ltm:
        print(f"LTM events (last 5):")
        for item in m.ltm[-5:]:
            print(f"  Step{item.step:5d} pos={item.pos} {item.event_type:12s} imp={item.importance:.1f} decay={item.decay:.2f}")

    # Retrieve memory at current position
    pos = app.world.get_agent_pos("test")
    retrieved = m.retrieve(pos, k=3)
    print(f"\n=== Memory at current pos {pos} ===")
    if retrieved:
        for item in retrieved:
            print(f"  Step{item.step:5d} pos={item.pos} {item.event_type:12s} imp={item.importance:.1f} score={item.importance*item.decay:.2f}")
    else:
        print("  No relevant memories")

iq = app._iq.evaluate()
print(f"\n=== IQ ===")
print(f"  Composite: {iq.composite*100:.0f}/100")
print(f"  S={iq.survival:.2f} E={iq.exploration:.2f} L={iq.learning:.2f} A={iq.adaptation:.2f} F={iq.efficiency:.2f} T={iq.stability:.2f}")
