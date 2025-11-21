# Step1：导入所有必要的库
import time
from typing import Optional, Union, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from IPython.display import display, clear_output

# Step2：辅助函数
def arr_in_list(array, _list):
    """
    检查一个numpy数组是否在另一个列表中
    """
    for element in _list:
        if np.array_equal(element, array):
            return True
    return False


# Step3：可视化工具
class Render:
    """
    网格世界可视化工具类
    """
    def __init__(self, target: Union[list, tuple, np.ndarray],
                 forbidden: Union[list, tuple, np.ndarray], size: int = 5):
        self.agent = None
        self.target = np.array(target)
        self.forbidden = [np.array(fob) for fob in forbidden]
        self.size = size
        self.fig = None
        self.ax = None
        self.trajectory = []
        self.reset_canvas()
    # 创建画布
    def create_canvas(self, figsize: Tuple[int, int] = None) -> None:
        if self.fig is not None:
            plt.close(self.fig)
        if figsize is None:
            figsize = (10, 10)
        self.fig, self.ax = plt.subplots(figsize=figsize, dpi=self.size * 20)
        if self.agent is not None:
            self.agent.remove()
        self.agent = patches.Arrow(-10, -10, 0.4, 0, color='red', width=0.5)
    # 初始化网格
    def init_grid(self) -> None:
        if self.ax is None:
            raise ValueError("请先调用create_canvas()创建画布")

        self.ax.xaxis.set_ticks_position('top')
        self.ax.invert_yaxis()
        self.ax.xaxis.set_ticks(range(0, self.size + 1))
        self.ax.yaxis.set_ticks(range(0, self.size + 1))
        self.ax.grid(True, linestyle="-", color="gray", linewidth="1", axis='both')
        self.ax.tick_params(bottom=False, left=False, right=False, top=False,
                           labelbottom=False, labelleft=False, labeltop=False)

        for y in range(self.size):
            self.write_word(pos=(-0.6, y), word=str(y + 1), size_discount=0.8)
            self.write_word(pos=(y, -0.6), word=str(y + 1), size_discount=0.8)

        self.ax.add_patch(self.agent)
    # 重置画布
    def reset_canvas(self, clear_trajectory: bool = True, figsize: Tuple[int, int] = None) -> None:
        self.create_canvas(figsize)
        self.init_grid()
        for pos in self.forbidden:
            self.fill_block(pos=pos)
        self.fill_block(pos=self.target, color='darkturquoise')
        if clear_trajectory:
            self.trajectory = []

    def fill_block(self, pos: Union[list, tuple, np.ndarray], color: str = '#EDB120',
                   width: float = 1.0, height: float = 1.0) -> patches.Rectangle:
        if self.ax is None:
            raise ValueError("请先调用create_canvas()创建画布")
        return self.ax.add_patch(
            patches.Rectangle((pos[0], pos[1]), width=width, height=height,
                            facecolor=color, fill=True, alpha=0.90)
        )
    # 画随机线
    def draw_random_line(self, pos1: Union[list, tuple, np.ndarray], pos2: Union[list, tuple, np.ndarray]) -> None:
        if self.ax is None:
            raise ValueError("请先调用create_canvas()创建画布")
        offset1 = np.random.uniform(low=-0.05, high=0.05, size=1)
        offset2 = np.random.uniform(low=-0.05, high=0.05, size=1)
        x = [pos1[0] + 0.5, pos2[0] + 0.5]
        y = [pos1[1] + 0.5, pos2[1] + 0.5]
        if pos1[0] == pos2[0]:
            x = [x[0] + offset1, x[1] + offset2]
        else:
            y = [y[0] + offset1, y[1] + offset2]
        self.ax.plot(x, y, color='g', scalex=False, scaley=False)
    # 画执行动作
    def draw_action(self, pos: Union[list, tuple, np.ndarray], toward: Union[list, tuple, np.ndarray],
                    color: str = 'green', radius: float = 0.10) -> None:
        if self.ax is None:
            raise ValueError("请先调用create_canvas()创建画布")
        if not np.array_equal(np.array(toward), np.array([0, 0])):
            self.ax.add_patch(
                patches.Arrow(pos[0] + 0.5, pos[1] + 0.5, dx=toward[0], dy=toward[1],
                            color=color, width=0.05 + 0.05 * np.linalg.norm(np.array(toward) / 0.5),
                            linewidth=0.5)
            )
        else:
            self.draw_circle(pos=tuple(pos), color='white', radius=radius, fill=False)
    # 画圈
    def draw_circle(self, pos: Union[list, tuple, np.ndarray], radius: float,
                    color: str = 'green', fill: bool = True) -> patches.Circle:
        if self.ax is None:
            raise ValueError("请先调用create_canvas()创建画布")
        return self.ax.add_patch(
            patches.Circle((pos[0] + 0.5, pos[1] + 0.5), radius=radius,
                         facecolor=color, edgecolor='green', linewidth=2, fill=fill)
        )

    def write_word(self, pos: Union[list, np.ndarray, tuple], word: str, color: str = 'black',
                   y_offset: float = 0, size_discount: float = 1.0) -> None:
        if self.ax is None:
            raise ValueError("请先调用create_canvas()创建画布")
        font_size = size_discount * (30 - 2 * self.size)
        self.ax.text(pos[0] + 0.5, pos[1] + 0.5 + y_offset, word,
                    size=font_size, ha='center', va='center', color=color)
    # 更新智能体
    def upgrade_agent(self, pos: Union[list, np.ndarray, tuple], action: Union[list, np.ndarray, tuple],
                      next_pos: Union[list, np.ndarray, tuple]) -> None:
        self.trajectory.append([tuple(pos), action, tuple(next_pos)])
    # 展示框架
    def show_frame(self, t: float = 0.2, close_after: bool = False) -> None:
        if self.fig is None:
            raise ValueError("请先调用create_canvas()创建画布")
        # 在Jupyter中显示
        clear_output(wait=True)
        display(self.fig)
        if close_after:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
    # 可视化状态值
    def visualize_state_values(self, state_values: np.ndarray, y_offset: float = 0.2) -> None:
        if self.ax is None:
            raise ValueError("请先调用create_canvas()创建画布")
        for state in range(self.size * self.size):
            x = state // self.size
            y = state % self.size
            value_text = f"{state_values[state]:.1f}"
            self.write_word(pos=(x, y), word=value_text, color='black',
                           y_offset=y_offset, size_discount=0.7)
    # 可视化策略
    def visualize_policy(self, policy: np.ndarray, action_to_direction: dict) -> None:
        if self.ax is None:
            raise ValueError("请先调用create_canvas()创建画布")
        for state in range(self.size * self.size):
            x = state // self.size
            y = state % self.size
            for action in range(len(action_to_direction)):
                prob = policy[state, action]
                if prob > 0.0:
                    direction = action_to_direction[action] * 0.4 * prob
                    self.draw_action(pos=[x, y], toward=direction, color='green', radius=0.03 + 0.07 * prob)




# Step4：网格世界环境
class GridWorldEnv:
    """
    强化学习网格世界环境
    """
    def __init__(self, size: int, start: Union[list, tuple, np.ndarray],
                 target: Union[list, tuple, np.ndarray], forbidden: Union[list, tuple, np.ndarray],
                 render_mode: Optional[str] = None, reward_list: Optional[List[float]] = None,
                 max_steps: int = 100000):

        if size <= 0 or not isinstance(size, int):
            raise ValueError("网格大小必须为正整数")

        def validate_position(pos, name):
            if (pos[0] < 0 or pos[0] >= size or pos[1] < 0 or pos[1] >= size):
                raise ValueError(f"{name}位置必须在网格范围内(0-{size-1})")
            return np.array(pos, dtype=int)

        self.time_steps = 0
        self.size = size
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.agent_location = validate_position(start, "起始点")
        self.target_location = validate_position(target, "目标点")

        self.forbidden_location = []
        for fob in forbidden:
            fob_pos = validate_position(fob, "障碍物")
            if np.array_equal(fob_pos, self.agent_location) or np.array_equal(fob_pos, self.target_location):
                raise ValueError("障碍物不能位于起点或目标点上")
            self.forbidden_location.append(fob_pos)

        self.render_ = Render(target=target, forbidden=forbidden, size=size)

        # 移除gym依赖，手动定义动作空间和观测空间
        self.action_space_size = 5  # 5个动作：停留、上、右、下、左

        # 动作到方向向量的映射
        self.action_to_direction = {
            0: np.array([0, 0]),    # 停留
            1: np.array([-1, 0]),   # 上（row 减小）
            2: np.array([0, 1]),    # 右（col 增大）
            3: np.array([1, 0]),    # 下（row 增大）
            4: np.array([0, -1]),   # 左（col 减小）
        }

        self.reward_list = reward_list if reward_list is not None else [0, 1, -10, -1]

        self.Rsa = None
        self.Psa = None
        self.psa_rsa_init()

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if options is not None and "start" in options:
            start_pos = self.state2pos(options['start'])
            start_pos = np.array(start_pos, dtype=int)
            if (start_pos[0] < 0 or start_pos[0] >= self.size or
                start_pos[1] < 0 or start_pos[1] >= self.size):
                raise ValueError(f"新起点必须在网格范围内(0-{self.size-1})")
            self.agent_location = start_pos
        else:
            self.agent_location = np.array([0, 0])

        self.time_steps = 0
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action: int):
        if action < 0 or action >= self.action_space_size:
            raise ValueError(f"动作必须在0-{self.action_space_size-1}范围内")

        current_state = self.pos2state(self.agent_location)
        reward_index = self.Rsa[current_state, action].tolist().index(1)
        reward = self.reward_list[reward_index]

        direction = self.action_to_direction[action]
        new_pos = self.agent_location + direction
        self.render_.upgrade_agent(self.agent_location, direction, new_pos)
        self.agent_location = np.clip(new_pos, 0, self.size - 1)

        self.time_steps += 1
        terminated = np.array_equal(self.agent_location, self.target_location)
        truncated = self.time_steps >= self.max_steps

        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, terminated, truncated, info

    def render(self) -> None:
        if self.render_mode == "video":
            self.render_.save_video('image/' + str(time.time()))
        self.render_.show_frame(100)

    def _get_obs(self):
        return {"agent": self.agent_location, "target": self.target_location, "barrier": self.forbidden_location}

    def _get_info(self):
        return {"time_steps": self.time_steps}

    def state2pos(self, state: int) -> np.ndarray:
        return np.array((state // self.size, state % self.size))

    def pos2state(self, pos: np.ndarray) -> int:
        return pos[0] * self.size + pos[1]

    def psa_rsa_init(self):
        state_size = self.size ** 2
        self.Psa = np.zeros(shape=(state_size, self.action_space_size, state_size), dtype=float)
        self.Rsa = np.zeros(shape=(state_size, self.action_space_size, len(self.reward_list)), dtype=float)

        for state_index in range(state_size):
            for action_index in range(self.action_space_size):
                pos = self.state2pos(state_index)
                next_pos = pos + self.action_to_direction[action_index]

                if next_pos[0] < 0 or next_pos[1] < 0 or next_pos[0] > self.size - 1 or next_pos[1] > self.size - 1:
                    self.Psa[state_index, action_index, state_index] = 1
                    self.Rsa[state_index, action_index, 3] = 1
                else:
                    next_state_index = self.pos2state(next_pos)
                    self.Psa[state_index, action_index, next_state_index] = 1

                    if np.array_equal(next_pos, self.target_location):
                        self.Rsa[state_index, action_index, 1] = 1
                    elif arr_in_list(next_pos, self.forbidden_location):
                        self.Rsa[state_index, action_index, 2] = 1
                    else:
                        self.Rsa[state_index, action_index, 0] = 1

    def get_state_space_info(self):
        total_states = self.size ** 2
        obstacle_states = [self.pos2state(obs) for obs in self.forbidden_location]
        start_state = self.pos2state(self.agent_location)
        target_state = self.pos2state(self.target_location)

        return {
            "total_states": total_states,
            "obstacle_states": obstacle_states,
            "start_state": start_state,
            "target_state": target_state,
            "valid_states": total_states - len(obstacle_states)
        }
