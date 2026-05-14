import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "impl"))
from self_model import SelfModel, SelfState


def test_creation():
    sm = SelfModel()
    assert sm.embed_dim == 8

def test_encode_before_update():
    sm = SelfModel()
    v = sm.encode()
    assert len(v) == 8

def test_update_and_encode():
    sm = SelfModel()
    sm.update(0.5, 0.7, 0.1, food_eaten=False, damage_taken=False, new_tile=True)
    v = sm.encode()
    assert len(v) == 8
    assert v[0] > 0.4  # energy propagated

def test_dual_speed():
    sm = SelfModel(slow_alpha=0.01, fast_alpha=0.5)
    sm.update(0.8, 1.0, 0.0, food_eaten=False, damage_taken=False, new_tile=False)
    sm.update(0.3, 0.5, 0.0, food_eaten=True, damage_taken=False, new_tile=True)
    v = sm.encode()
    assert v[0] > v[3]  # slow energy > fast eat rate

def test_get_state():
    sm = SelfModel()
    sm.update(0.6, 0.8, 0.1, food_eaten=True, damage_taken=False, new_tile=True)
    st = sm.get_state()
    assert isinstance(st, SelfState)
    assert 0 < st.energy < 1

def test_event_freshness():
    sm = SelfModel()
    sm.update(0.8, 1.0, 0.0, food_eaten=False, damage_taken=False, new_tile=False)
    st = sm.get_state()
    assert st.steps_since_event > 0
    sm.update(0.8, 1.0, 0.0, food_eaten=True, damage_taken=False, new_tile=False)
    st = sm.get_state()
    assert st.steps_since_event == 0
