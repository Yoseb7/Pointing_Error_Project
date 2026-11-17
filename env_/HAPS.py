import gymnasium as gym
from gymnasium import spaces
import numpy as np


TIME_STEP = 0.1
MAX_ACCELERATION = 0.2
MAX_SPEED = 1.0
MAX_WIND_SPEED = 0.1
MAX_POSITION = 5.0
WIND_CHANGE_PER_STEP = 0.005
MAX_EPISODE_STEPS = 500
POSITION_TOLERANCE = 0.05
GROUND_SPEED_TOLERANCE = 0.025
TARGET_REWARD = 5.0
TARGET_DECAY = 2.0
EDGE_PENALTY = 5.0
EDGE_EXPONENT = 6.0
EARLY_TERMINATION_PENALTY = 250.0
COMPLETION_BONUS = 250.0
RENDER_WIDTH = 980
RENDER_HEIGHT = 600
PLOT_SIZE = 520


class HAPSEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=np.array(
                [-MAX_POSITION, -MAX_POSITION,
                 -(MAX_SPEED + MAX_WIND_SPEED), -(MAX_SPEED + MAX_WIND_SPEED)],
                dtype=np.float32,
            ),
            high=np.array(
                [MAX_POSITION, MAX_POSITION,
                 MAX_SPEED + MAX_WIND_SPEED, MAX_SPEED + MAX_WIND_SPEED],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )
        self.HAPS_x_pos = 0.0
        self.HAPS_y_pos = 0.0
        self.HAPS_x_vel = 0.0
        self.HAPS_y_vel = 0.0
        self.HAPS_x_acc = 0.0
        self.HAPS_y_acc = 0.0
        self.wind_x_vel = 0.0
        self.wind_y_vel = 0.0
        self._steps = 0
        self.total_reward = 0.0
        self._out_of_bounds = False
        self._last_action = np.zeros(2, dtype=np.float32)
        self._position_history = []
        self.screen = None
        self.clock = None
        self.font = None
        self._render_closed = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.HAPS_x_pos = float(self.np_random.uniform(-1.0, 1.0))
        self.HAPS_y_pos = float(self.np_random.uniform(-1.0, 1.0))
        self.HAPS_x_vel = 0.0
        self.HAPS_y_vel = 0.0
        self.HAPS_x_acc = 0.0
        self.HAPS_y_acc = 0.0
        self.wind_x_vel = self.update_wind_velocity()
        self.wind_y_vel = self.update_wind_velocity()
        self._steps = 0
        self.total_reward = 0.0
        self._out_of_bounds = False
        self._last_action = np.zeros(2, dtype=np.float32)
        self._position_history = [(self.HAPS_x_pos, self.HAPS_y_pos)]
        return self.get_obs(), self._get_info()

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != self.action_space.shape:
            raise ValueError(
                f"Expected action shape {self.action_space.shape}, got {action.shape}"
            )
        if not np.all(np.isfinite(action)):
            raise ValueError("Action values must be finite")

        self._last_action = np.clip(action, -1.0, 1.0)
        self.HAPS_x_acc = float(self._last_action[0] * MAX_ACCELERATION)
        self.HAPS_y_acc = float(self._last_action[1] * MAX_ACCELERATION)

        self.HAPS_x_vel = float(np.clip(
            self.HAPS_x_vel + self.HAPS_x_acc * TIME_STEP,
            -MAX_SPEED,
            MAX_SPEED,
        ))
        self.HAPS_y_vel = float(np.clip(
            self.HAPS_y_vel + self.HAPS_y_acc * TIME_STEP,
            -MAX_SPEED,
            MAX_SPEED,
        ))
        self._advance_wind()

        next_x = self.HAPS_x_pos + (
            self.HAPS_x_vel + self.wind_x_vel
        ) * TIME_STEP
        next_y = self.HAPS_y_pos + (
            self.HAPS_y_vel + self.wind_y_vel
        ) * TIME_STEP
        self._out_of_bounds = (
            abs(next_x) > MAX_POSITION or abs(next_y) > MAX_POSITION
        )
        self.HAPS_x_pos = float(np.clip(next_x, -MAX_POSITION, MAX_POSITION))
        self.HAPS_y_pos = float(np.clip(next_y, -MAX_POSITION, MAX_POSITION))
        self._position_history.append((self.HAPS_x_pos, self.HAPS_y_pos))
        self._steps += 1

        terminated = self.is_terminated()
        truncated = self.is_truncated()
        reward = self.HAPS_reward()
        self.total_reward += reward
        return self.get_obs(), reward, terminated, truncated, self._get_info()

    def get_obs(self):
        ground_vx, ground_vy = self._ground_velocity()
        return np.array(
            [
                self.HAPS_x_pos,
                self.HAPS_y_pos,
                ground_vx,
                ground_vy,
            ],
            dtype=np.float32,
        )

    def update_wind_velocity(self):
        return float(self.np_random.uniform(-MAX_WIND_SPEED, MAX_WIND_SPEED))

    def _advance_wind(self):
        wind_change = self.np_random.uniform(
            -WIND_CHANGE_PER_STEP, WIND_CHANGE_PER_STEP, size=2
        )
        self.wind_x_vel = float(np.clip(
            self.wind_x_vel + wind_change[0],
            -MAX_WIND_SPEED,
            MAX_WIND_SPEED,
        ))
        self.wind_y_vel = float(np.clip(
            self.wind_y_vel + wind_change[1],
            -MAX_WIND_SPEED,
            MAX_WIND_SPEED,
        ))

    def _ground_velocity(self):
        return (
            self.HAPS_x_vel + self.wind_x_vel,
            self.HAPS_y_vel + self.wind_y_vel,
        )

    def HAPS_reward(self):
        distance = np.hypot(self.HAPS_x_pos, self.HAPS_y_pos)
        edge_fraction = min(1.0, max(abs(self.HAPS_x_pos), abs(self.HAPS_y_pos)) / MAX_POSITION)
        target_reward = TARGET_REWARD * np.exp(-TARGET_DECAY * distance)
        edge_penalty = -EDGE_PENALTY * np.expm1(EDGE_EXPONENT * edge_fraction) / np.expm1(EDGE_EXPONENT)
        reward = target_reward + edge_penalty
        if self._out_of_bounds:
            remaining_fraction = max(0.0, 1.0 - self._steps / MAX_EPISODE_STEPS)
            reward -= EARLY_TERMINATION_PENALTY * (1.0 + remaining_fraction)
        elif self._steps >= MAX_EPISODE_STEPS:
            reward += COMPLETION_BONUS
        return float(reward)

    def is_terminated(self):
        return self._out_of_bounds

    def _is_on_target(self):
        ground_vx, ground_vy = self._ground_velocity()
        return (
            abs(self.HAPS_x_pos) <= POSITION_TOLERANCE
            and abs(self.HAPS_y_pos) <= POSITION_TOLERANCE
            and abs(ground_vx) <= GROUND_SPEED_TOLERANCE
            and abs(ground_vy) <= GROUND_SPEED_TOLERANCE
        )

    def is_truncated(self):
        return self._steps >= MAX_EPISODE_STEPS

    def _get_info(self):
        ground_vx, ground_vy = self._ground_velocity()
        return {
            "position": (self.HAPS_x_pos, self.HAPS_y_pos),
            "position_error": float(np.hypot(self.HAPS_x_pos, self.HAPS_y_pos)),
            "velocity": (self.HAPS_x_vel, self.HAPS_y_vel),
            "wind_velocity": (self.wind_x_vel, self.wind_y_vel),
            "ground_velocity": (ground_vx, ground_vy),
            "acceleration": (self.HAPS_x_acc, self.HAPS_y_acc),
            "success": self._is_on_target(),
            "total_reward": self.total_reward,
        }

    def render(self):
        if self._render_closed:
            return

        import pygame

        if not pygame.get_init():
            pygame.init()
        if self.screen is None:
            self.screen = pygame.display.set_mode((RENDER_WIDTH, RENDER_HEIGHT))
            pygame.display.set_caption("HAPS Wind Compensation")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 24)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        self.screen.fill((245, 247, 250))
        plot_left = 20
        plot_top = 40
        center_x = plot_left + PLOT_SIZE // 2
        center_y = plot_top + PLOT_SIZE // 2
        pixels_per_unit = PLOT_SIZE / (2 * MAX_POSITION)

        for coordinate in range(-int(MAX_POSITION), int(MAX_POSITION) + 1):
            x = round(center_x + coordinate * pixels_per_unit)
            y = round(center_y - coordinate * pixels_per_unit)
            color = (220, 224, 230) if coordinate else (125, 135, 150)
            pygame.draw.line(
                self.screen, color, (x, plot_top), (x, plot_top + PLOT_SIZE)
            )
            pygame.draw.line(
                self.screen, color, (plot_left, y), (plot_left + PLOT_SIZE, y)
            )

        def to_screen(position):
            x_pos, y_pos = position
            return (
                round(center_x + x_pos * pixels_per_unit),
                round(center_y - y_pos * pixels_per_unit),
            )

        if len(self._position_history) > 1:
            pygame.draw.lines(
                self.screen,
                (90, 145, 210),
                False,
                [to_screen(position) for position in self._position_history],
                2,
            )

        target = to_screen((0.0, 0.0))
        haps = to_screen((self.HAPS_x_pos, self.HAPS_y_pos))
        pygame.draw.circle(self.screen, (45, 160, 90), target, 8)
        pygame.draw.circle(self.screen, (210, 65, 65), haps, 9)

        def draw_vector(vector, scale, color):
            direction = pygame.math.Vector2(vector[0], -vector[1]) * scale
            if direction.length_squared() < 1e-8:
                return
            end = pygame.math.Vector2(haps) + direction
            pygame.draw.line(self.screen, color, haps, end, 3)
            head_length = min(10.0, direction.length() * 0.4)
            for angle in (-30, 30):
                tip = end - direction.normalize().rotate(angle) * head_length
                pygame.draw.line(self.screen, color, end, tip, 3)

        draw_vector((self.wind_x_vel, self.wind_y_vel), 45 / MAX_WIND_SPEED,
                    (225, 145, 35))
        draw_vector(self._last_action, 60, (145, 65, 200))

        ground_vx, ground_vy = self._ground_velocity()
        details = [
            "HAPS wind compensation",
            "Green: target (0, 0)",
            "Red: HAPS",
            "Blue: path",
            "Orange: wind",
            "Purple: action (acceleration)",
            "",
            f"Position: ({self.HAPS_x_pos:.2f}, {self.HAPS_y_pos:.2f})",
            f"Ground velocity: ({ground_vx:.3f}, {ground_vy:.3f})",
            f"Wind velocity: ({self.wind_x_vel:.3f}, {self.wind_y_vel:.3f})",
            f"Acceleration: ({self.HAPS_x_acc:.3f}, {self.HAPS_y_acc:.3f})",
            f"Action: ({self._last_action[0]:.2f}, {self._last_action[1]:.2f})",
            f"Step: {self._steps}/{MAX_EPISODE_STEPS}",
            f"Total reward: {self.total_reward:.1f}",
        ]
        for index, text in enumerate(details):
            color = (35, 40, 50)
            if text == "Green: target (0, 0)":
                color = (45, 145, 80)
            elif text == "Red: HAPS":
                color = (190, 55, 55)
            elif text == "Blue: path":
                color = (90, 145, 210)
            elif text == "Orange: wind":
                color = (195, 120, 25)
            elif text == "Purple: action (acceleration)":
                color = (145, 65, 200)
            label = self.font.render(text, True, color)
            self.screen.blit(label, (565, 55 + index * 34))

        pygame.display.flip()
        self.clock.tick(20)

    def close(self):
        self._render_closed = True
        if self.screen is not None:
            import pygame
            pygame.quit()
            self.screen = None
            self.clock = None
            self.font = None
