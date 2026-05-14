# MVP 0 — 首个完整原型

首个完整可运行的 Evola 原型。P0 四模块集成。

## 双入口

| 入口 | 命令 | 风格 |
|------|------|------|
| **Pygame** (推荐) | `python MVPs\MVP0\evola_pygame.py` | 图形窗口，鼠标点击，完整系统 |
| **CLI** | `python MVPs\MVP0\evola_cli.py` | 终端仪表盘，键盘命令，批量诊断 |

两个入口共享同一套核心模块和配置。

## Pygame 窗口

```bash
python MVPs\MVP0\evola_pygame.py
```

启动后自动创建世界和智能体。顶部工具栏：Start / Spawn / Run 100 / Pause / Extract / Debug / Stop / Quit。

界面：左侧世界地图（彩色网格），右侧智能体面板（能量/好奇/安全进度条 + 事件日志）。

操作：
- 按钮栏：鼠标左键点击
- Space：暂停/继续
- D：调试模式
- Q：退出

## CLI 观察站

```bash
python MVPs\MVP0\evola_cli.py
```

三栏布局：坤舆地图、钧鉴日志、未晞面板。键盘命令驱动。

## 一次性模式（旧）

```bash
python MVPs\MVP0\mvp_0_demo.py --render none --steps 2000
```

## 配置

编辑 `mvp_0_config.py`，所有入口共用。

## 观察

详见 [OBSERVATION.md](OBSERVATION.md)。
