import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
import sys
import time

# =========================
# CONFIG VISUAL
# =========================
CELL_SIZE = 60
GRID_SIZE = 10
WINDOW_SIZE = CELL_SIZE * GRID_SIZE


def init_pygame():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("Submarine Battle")
    return screen


def draw_grid(screen):
    for x in range(0, WINDOW_SIZE, CELL_SIZE):
        for y in range(0, WINDOW_SIZE, CELL_SIZE):
            rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, (200, 200, 200), rect, 1)


def draw_entity(screen, pos, color):
    x, y = pos
    rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    pygame.draw.rect(screen, color, rect)


# =========================
# ENTORNO
# =========================
class SubmarineBattleEnv(gym.Env):

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, render_mode=None):
        super().__init__()
        self.grid_size = 10
        self.render_mode = render_mode

        # 0-4: mover, 5: atacar
        self.action_space = spaces.MultiDiscrete([6, 6])

        self.observation_space = spaces.Box(
            low=0,
            high=self.grid_size - 1,
            shape=(8,),
            dtype=np.int32,
        )

        self.sub1 = None
        self.sub2 = None
        self.cargo = None
        self.destroyer = None

    def _get_obs(self):
        return np.array([
            self.sub1[0], self.sub1[1],
            self.sub2[0], self.sub2[1],
            self.cargo[0], self.cargo[1],
            self.destroyer[0], self.destroyer[1]
        ], dtype=np.int32)

    def _clip_position(self, pos):
        pos[0] = int(np.clip(pos[0], 0, self.grid_size - 1))
        pos[1] = int(np.clip(pos[1], 0, self.grid_size - 1))

    def _manhattan_distance(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _move_destroyer_towards_nearest_sub(self):
        d1 = self._manhattan_distance(self.destroyer, self.sub1)
        d2 = self._manhattan_distance(self.destroyer, self.sub2)
        target = self.sub1 if d1 <= d2 else self.sub2

        if self.destroyer[0] < target[0]:
            self.destroyer[0] += 1
        elif self.destroyer[0] > target[0]:
            self.destroyer[0] -= 1
        elif self.destroyer[1] < target[1]:
            self.destroyer[1] += 1
        elif self.destroyer[1] > target[1]:
            self.destroyer[1] -= 1

        self._clip_position(self.destroyer)

    def _attack(self, attacker_pos, target_pos):
        return self._manhattan_distance(attacker_pos, target_pos) <= 1

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.sub1 = [0, 9]
        self.sub2 = [1, 9]
        self.cargo = [0, 0]
        self.destroyer = [1, 0]

        # HP
        self.sub1_hp = 3
        self.sub2_hp = 3
        self.destroyer_hp = 4

        # Control de pasos
        self.step_count = 0
        self.max_steps = 100

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), {}

    def _move_submarine(self, pos, action):
        if action == 0:
            pos[1] -= 1
        elif action == 1:
            pos[1] += 1
        elif action == 2:
            pos[0] -= 1
        elif action == 3:
            pos[0] += 1
        elif action == 4:
            pass  # quieto
        elif action == 5:
            pass  # atacar se maneja fuera
        self._clip_position(pos)

    def step(self, action):
        reward = -1
        terminated = False
        truncated = False
        self.step_count += 1

        a1, a2 = int(action[0]), int(action[1])
        self._move_submarine(self.sub1, a1)
        self._move_submarine(self.sub2, a2)

        # Carguero avanza
        self.cargo[0] = min(self.cargo[0] + 1, self.grid_size - 1)

        # Destructor avanza hacia submarinos
        self._move_destroyer_towards_nearest_sub()

        # Ataque submarinos
        if a1 == 5 and self._attack(self.sub1, self.destroyer):
            self.destroyer_hp -= 1
            reward += 10
        if a2 == 5 and self._attack(self.sub2, self.destroyer):
            self.destroyer_hp -= 1
            reward += 10

        # Ataque destructor
        if self._attack(self.destroyer, self.sub1):
            self.sub1_hp -= 1
            reward -= 5
        if self._attack(self.destroyer, self.sub2):
            self.sub2_hp -= 1
            reward -= 5

        # Condiciones de fin
        if self.destroyer_hp <= 0:
            reward += 100
            terminated = True
        elif self.sub1_hp <= 0 and self.sub2_hp <= 0:
            reward -= 100
            terminated = True
        elif self.step_count >= self.max_steps:
            terminated = True

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        if not hasattr(self, "screen"):
            self.screen = init_pygame()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        self.screen.fill((0, 0, 50))

        draw_grid(self.screen)
        draw_entity(self.screen, self.cargo, (255, 255, 0))  # amarillo
        draw_entity(self.screen, self.destroyer, (255, 0, 0))  # rojo
        draw_entity(self.screen, self.sub1, (0, 255, 0))  # verde
        draw_entity(self.screen, self.sub2, (0, 200, 255))  # azul

        # Mostrar HP
        font = pygame.font.SysFont(None, 24)
        text = font.render(
            f"S1:{self.sub1_hp}  S2:{self.sub2_hp}  D:{self.destroyer_hp}",
            True,
            (255, 255, 255)
        )
        self.screen.blit(text, (10, 10))

        pygame.display.flip()
        time.sleep(0.2)

    def close(self):
        pygame.quit()


# =========================
# REGISTRO
# =========================
gym.register(
    id="SubmarineBattle-v0",
    entry_point=SubmarineBattleEnv,
)

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    env = gym.make("SubmarineBattle-v0", render_mode="human")

    while True:
        obs, info = env.reset()
        done = False

        while not done:
            action = env.action_space.sample()  # aleatorio
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated