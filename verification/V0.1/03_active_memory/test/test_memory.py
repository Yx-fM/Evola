import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "impl"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "11_memory_io", "impl"))
from memory_store import ActiveMemory, MemoryItem
from memory_io import MemoryIO


class TestActiveMemory:
    def test_write_and_retrieve(self):
        m = ActiveMemory()
        item = MemoryItem(id=0, step=1, pos=(3, 5), event_type="food_found",
                          energy=0.5, novelty=0.8, safety=0.0, importance=0.8)
        m.write(item)
        results = m.retrieve((3, 5), k=3)
        assert len(results) == 1
        assert results[0].event_type == "food_found"

    def test_decay_removes_old_items(self):
        m = ActiveMemory(stm_decay=0.5)
        item = MemoryItem(id=0, step=1, pos=(0, 0), event_type="food_visible",
                          energy=0.5, novelty=0.8, safety=0.0, importance=0.3)
        m.write(item)
        for _ in range(5):
            m.decay_step(0.0)
        assert len(m.stm) == 0  # fully decayed

    def test_ltm_promotion(self):
        m = ActiveMemory(ltm_importance_threshold=0.5)
        item_hi = MemoryItem(id=0, step=1, pos=(0, 0), event_type="food_found",
                              energy=0.5, novelty=0.8, safety=0.0, importance=0.9)
        item_lo = MemoryItem(id=1, step=2, pos=(1, 1), event_type="food_visible",
                              energy=0.5, novelty=0.8, safety=0.0, importance=0.3)
        m.write(item_hi); m.write(item_lo)
        assert len(m.ltm) == 1  # only important item promoted

    def test_memory_load(self):
        m = ActiveMemory(stm_capacity=10)
        for i in range(5):
            m.write(MemoryItem(id=0, step=i, pos=(i, 0), event_type="new_area",
                               energy=0.5, novelty=0.8, safety=0.0, importance=0.5))
        assert 0.4 < m.get_memory_load() < 0.6

    def test_position_match_bonus(self):
        m = ActiveMemory()
        m.write(MemoryItem(id=0, step=1, pos=(3, 5), event_type="food_found",
                           energy=0.5, novelty=0.8, safety=0.0, importance=1.0))
        m.write(MemoryItem(id=1, step=2, pos=(10, 10), event_type="danger",
                           energy=0.5, novelty=0.8, safety=0.0, importance=0.9))
        results = m.retrieve((3, 5), k=2)
        assert results[0].pos == (3, 5)  # position-matched first


class TestMemoryIO:
    def test_encode_food(self):
        m = ActiveMemory()
        io = MemoryIO(m)
        io.encode(step=10, pos=(3, 5), energy=0.5, novelty=0.8, safety=0.0,
                   food_eaten=True, damage_taken=False, new_tile=False, food_visible=False)
        assert len(m.stm) == 1
        assert m.stm[0].event_type == "food_found"
        assert m.stm[0].importance > 0.9

    def test_retrieve_context(self):
        m = ActiveMemory()
        io = MemoryIO(m)
        io.encode(step=1, pos=(3, 5), energy=0.5, novelty=0.8, safety=0.0,
                   food_eaten=True, damage_taken=False, new_tile=False, food_visible=False)
        ctx = io.retrieve_context((3, 5))
        assert len(ctx) == 4
        assert ctx[0] > 0  # food hints

    def test_no_events_return_zero_context(self):
        m = ActiveMemory()
        io = MemoryIO(m)
        ctx = io.retrieve_context((3, 5))
        assert ctx == [0.0, 0.0, 0.0, 0.0]
