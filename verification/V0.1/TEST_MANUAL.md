# V0.1 测试手册

> 13 模块全部完成，93 tests pass。MVP0/1/2 均可运行。

## 模块状态

| 模块 | 状态 | 测试 | MVP |
|------|------|------|-----|
| 08_kunyu | ✅ | 15 | 0/1/2 |
| 02_homeostasis | ✅ | 19 | 0/1/2 |
| 09_junjian | ✅ | smoke | 0/1/2 |
| 10_sensorimotor | ✅ | 22 | 0/1/2 |
| 01_liquid_dynamics | ✅ | 13 | 1/2 |
| 04_self_model | ✅ | 6 | 1/2 |
| 06_hot_warm_cold | ✅ | — | 1/2 |
| 12_mode_scheduler | ✅ | 5 | 1/2 |
| 03_active_memory | ✅ | 8 | 2 |
| 05_world_model | ✅ | 5 | 2 |
| 11_memory_io | ✅ | — | 2 |
| 07_evolution | 📋 | — | — |
| 00_prototype | 📋 | — | — |

## 快速测试

```bash
cd Q:\All_Items\DreamProjects\Evola

python -m pytest verification/V0.1/08_kunyu/test/ -v
python -m pytest verification/V0.1/02_homeostasis/test/ -v
python -m pytest verification/V0.1/10_sensorimotor/test/ -v
python -m pytest verification/V0.1/01_liquid_dynamics/test/ -v
python -m pytest verification/V0.1/04_self_model/test/ -v
python -m pytest verification/V0.1/12_mode_scheduler/test/ -v
python -m pytest verification/V0.1/03_active_memory/test/ -v
python -m pytest verification/V0.1/05_world_model/test/ -v
```

## MVP 入口

| MVP | 命令 |
|-----|------|
| MVP 0 (Rule) | `python MVPs/MVP0/evola_pygame.py` |
| MVP 1 (LTC) | `python MVPs/MVP1/evola_pygame.py` |
| MVP 2 (Full) | `python MVPs/MVP2/evola_pygame.py` |
