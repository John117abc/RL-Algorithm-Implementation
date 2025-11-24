from typing import Optional
import numpy as np
import gymnasium as gym


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}  # 声明支持的模式
    def __init__(self, size: int = 5,render_mode: str = None):
        # 显示
        self.render_mode = render_mode
        # 网格空间的大小 (默认 5*5)
        self.size = size

        # 初始化位置
        # -1, -1未初始化的状态
        self._agent_location = np.array([-1, -1], dtype=np.int32)
        self._target_location = np.array([-1, -1], dtype=np.int32)

        # 定义智能体可以观察的内容
        # {"agent": array([1, 0]), "target": array([0, 3])}，其中数组表示 x、y坐标
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),   # [x, y] coordinates
                "target": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),  # [x, y] coordinates
            }
        )

        # 定义动作空间
        self.action_space = gym.spaces.Discrete(4)

        # 将动作编号映射到网格上的实际移动
        self._action_to_direction = {
            0: np.array([1, 0]),   # 右移 (positive x)
            1: np.array([0, 1]),   # 上移 (positive y)
            2: np.array([-1, 0]),  # 左移 (negative x)
            3: np.array([0, -1]),  # 下移 (negative y)
        }

        # 可选：为渲染准备窗口（延迟初始化）
        self.window = None
        self.clock = None

    def _get_obs(self):
        """将内部状态转换为观测格式。

        Returns:
            dict: 观察智能体和目标位置
        """
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        """计算用于调试的辅助信息。

        Returns:
            dict: 智能体与目标之间的距离信息（曼哈顿距离）
        """
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """开始一个新的回合.

        Args:
            seed: 用于生成可复现事件的随机种子
            options: A其它配置

        Returns:
            tuple: （观察，信息）初始状态
        """
        # 如果不需要随机初始位置，获取options的enable_random_pos
        if options['enable_random_pos']:
            # 注意: 必须先调用此函数来初始化随机数生成器。
            super().reset(seed=seed)

            # 将智能体随机放置在网格上的任意位置
            self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)
            # 随机放置目标，确保其位置与智能体位置不同。
            self._target_location = self._agent_location
            while np.array_equal(self._target_location, self._agent_location):
                self._target_location = self.np_random.integers(
                    0, self.size, size=2, dtype=int
                )
        else:
            self._agent_location = np.array([0,0])
            self._target_location = np.array([self.size -1,self.size -1])

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        """在环境中执行一个时间步。

        Args:
            action: 要采取的行动（0-3 表示方向）

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # 将离散动作（0-3）映射到运动方向
        direction = self._action_to_direction[action]

        # 更新智能体位置，确保其保持在网格边界内。
        # np.clip 防止智能体走到边缘之外
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        # 检查智能体是否到达目标
        terminated = np.array_equal(self._agent_location, self._target_location)

        # 在这个简单的环境中，我们不使用截断。（如果需要，可以在此处添加步数限制）
        truncated = False

        # 简单的奖励机制：达到目标加 10 分，否则不加 -1 分
        # 另一种方法：可以对每一步操作给予少量负奖励，以鼓励提高效率。
        reward = 10 if terminated else -1

        # 也可以对距离塑造奖励
        # distance = np.linalg.norm(self._agent_location - self._target_location)
        # reward = 1 if terminated else -0.1 * distance

        observation = self._get_obs()
        info = self._get_info()

        # 如果设置了 "human" 渲染模式，则更新显示。
        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            self._render_frame()
            # 如果使用 pygame，human 模式会自动显示窗口
        else:
            # render_mode is None or unsupported
            return None

    def _render_frame(self):
        try:
            import pygame
        except ImportError as e:
            raise ImportError("pygame is required...") from e

        # 动态计算 cell_size，确保窗口不超过 max_window_size
        max_window_size = 800
        cell_size = max(1, min(50, max_window_size // self.size))  # 至少 1px，最多 50px
        window_size = self.size * cell_size

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((window_size, window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((window_size, window_size))
        canvas.fill((255, 255, 255))

        # 绘制网格（可选：仅当 cell_size >= 3 时画线，避免太密）
        if cell_size >= 3:
            for i in range(self.size + 1):
                pygame.draw.line(canvas, (200, 200, 200), (0, i * cell_size), (window_size, i * cell_size))
                pygame.draw.line(canvas, (200, 200, 200), (i * cell_size, 0), (i * cell_size, window_size))

        # 绘制目标 T（绿色方块）
        tx, ty = self._target_location
        pygame.draw.rect(
            canvas,
            (0, 255, 0),
            pygame.Rect(ty * cell_size, tx * cell_size, cell_size, cell_size),
        )

        # 绘制智能体 A（蓝色圆圈）
        ax, ay = self._agent_location
        center = (ay * cell_size + cell_size // 2, ax * cell_size + cell_size // 2)
        radius = max(1, cell_size // 3)
        pygame.draw.circle(canvas, (0, 0, 255), center, radius)

        if self.render_mode == "human":
            self.window.blit(canvas, (0, 0))
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        if self.window is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None

#
# # 注册环境，以便我们可以使用 gym.make() 创建它。
# gym.register(
#     id="gymnasium_env/my_GridWorld-v0",
#     entry_point=GridWorldEnv,
#     max_episode_steps=10001,  # 防止无限循环
# )