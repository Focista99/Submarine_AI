import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt

class SubmarineBattleEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, grid_size=10, render_mode=None):
        super().__init__()
        self.grid_size = grid_size
        self.render_mode = render_mode
        self.action_space = spaces.MultiDiscrete([5, 5])
        
        self.observation_space = spaces.Box(low=-grid_size*2, high=grid_size*2, shape=(8,), dtype=np.int32)
        
        # Variables para el renderizado
        self.fig = None
        self.ax = None
        
        self.reset()

    def _get_obs(self):
        return np.array([
            self.cargo[0] - self.sub1[0], self.cargo[1] - self.sub1[1],
            self.destroyer[0] - self.sub1[0], self.destroyer[1] - self.sub1[1],
            self.cargo[0] - self.sub2[0], self.cargo[1] - self.sub2[1],
            self.destroyer[0] - self.sub2[0], self.destroyer[1] - self.sub2[1]
        ], dtype=np.int32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sub1 = [0, self.grid_size - 1]
        self.sub2 = [1, self.grid_size - 1]
        self.cargo = [0, 0]
        self.destroyer = [1, 0]
        
        self.sub1_alive = True
        self.sub2_alive = True
        self.steps = 0
        
        if self.render_mode == "human":
            self.render()
            
        return self._get_obs(), {}

    def step(self, action):
        self.steps += 1
        reward = -0.1
        terminated = False
        
        d_cargo_prev = min(
            self._dist(self.sub1, self.cargo) if self.sub1_alive else float('inf'),
            self._dist(self.sub2, self.cargo) if self.sub2_alive else float('inf')
        )
        
        # 1. Mover submarinos
        if self.sub1_alive: self._move(self.sub1, action[0])
        if self.sub2_alive: self._move(self.sub2, action[1])
        
        # 2. Mover carguero (1 de cada 2 pasos)
        if self.steps % 2 == 0:
            self.cargo[0] = min(self.cargo[0] + 1, self.grid_size - 1)
            
        # 3. Mover destructor
        self._move_destroyer()

        # 4. Colisiones (Muertes)
        if self.sub1_alive and self.sub1 == self.destroyer:
            self.sub1_alive = False
            self.sub1 = [-10, -10]
            reward -= 50
            
        if self.sub2_alive and self.sub2 == self.destroyer:
            self.sub2_alive = False
            self.sub2 = [-10, -10]
            reward -= 50

        # Reward shaping
        d_cargo_now = min(
            self._dist(self.sub1, self.cargo) if self.sub1_alive else float('inf'),
            self._dist(self.sub2, self.cargo) if self.sub2_alive else float('inf')
        )
        if d_cargo_now < d_cargo_prev:
            reward += 0.5

        # Fin de juego
        if (self.sub1_alive and self.sub1 == self.cargo) or (self.sub2_alive and self.sub2 == self.cargo):
            reward += 100
            terminated = True
        elif not self.sub1_alive and not self.sub2_alive:
            reward -= 20
            terminated = True
        elif self.cargo[0] == self.grid_size - 1:
            reward -= 20
            terminated = True
            
        if self.render_mode == "human":
            self.render()
            
        return self._get_obs(), reward, terminated, False, {}

    def _dist(self, a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    def _move(self, pos, act):
        if act == 0: pos[1] = max(0, pos[1]-1)
        elif act == 1: pos[1] = min(self.grid_size-1, pos[1]+1)
        elif act == 2: pos[0] = max(0, pos[0]-1)
        elif act == 3: pos[0] = min(self.grid_size-1, pos[0]+1)

    def _move_destroyer(self):
        if not self.sub1_alive and not self.sub2_alive: return

        d1 = self._dist(self.destroyer, self.sub1) if self.sub1_alive else float('inf')
        d2 = self._dist(self.destroyer, self.sub2) if self.sub2_alive else float('inf')
        target = self.sub1 if d1 <= d2 else self.sub2
        
        if self.destroyer[0] < target[0]: self.destroyer[0] += 1
        elif self.destroyer[0] > target[0]: self.destroyer[0] -= 1
        elif self.destroyer[1] < target[1]: self.destroyer[1] += 1
        elif self.destroyer[1] > target[1]: self.destroyer[1] -= 1

    def render(self):
        if self.render_mode != "human":
            return
            
        if self.fig is None:
            plt.ion() # Modo interactivo para que no bloquee la ejecución
            self.fig, self.ax = plt.subplots(figsize=(6,6))

        self.ax.clear()
        self.ax.set_xlim(-0.5, self.grid_size - 0.5)
        self.ax.set_ylim(self.grid_size - 0.5, -0.5)
        self.ax.grid(True)
        
        # Dibujar entidades
        self.ax.scatter(self.cargo[0], self.cargo[1], c='green', s=200, marker='s', label='Carguero')
        self.ax.scatter(self.destroyer[0], self.destroyer[1], c='red', s=200, marker='X', label='Destructor')
        
        if self.sub1_alive:
            self.ax.scatter(self.sub1[0], self.sub1[1], c='blue', s=100, label='Sub 1')
        if self.sub2_alive:
            self.ax.scatter(self.sub2[0], self.sub2[1], c='cyan', s=100, label='Sub 2')
        
        self.ax.legend(loc='upper right')
        plt.title(f"Batalla Submarina - Paso {self.steps}")
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.2) # Ajusta este valor para que vaya más rápido o más lento

    def close(self):
        if self.fig is not None:
            plt.ioff()
            plt.close(self.fig)
            self.fig = None


class QAgent:
    def __init__(self):
        self.q_table = {} 
        self.lr = 0.1
        self.gamma = 0.95
        self.epsilon = 1.0

    def get_state_key(self, obs):
        return tuple(obs)

    def choose_action(self, obs):
        state = self.get_state_key(obs)
        if np.random.random() < self.epsilon:
            return [np.random.randint(0, 5), np.random.randint(0, 5)]
        
        if state not in self.q_table:
            return [4, 4]
        
        idx = np.argmax(self.q_table[state])
        return [idx // 5, idx % 5]

    def learn(self, obs, action, reward, next_obs):
        s, ns = self.get_state_key(obs), self.get_state_key(next_obs)
        a_idx = action[0] * 5 + action[1]
        
        if s not in self.q_table: self.q_table[s] = np.zeros(25)
        if ns not in self.q_table: self.q_table[ns] = np.zeros(25)
        
        target = reward + self.gamma * np.max(self.q_table[ns])
        self.q_table[s][a_idx] += self.lr * (target - self.q_table[s][a_idx])

# --- EJECUCIÓN ---
if __name__ == "__main__":
    # 1. ENTRENAMIENTO SIN RENDER (para que sea rápido)
    env_train = SubmarineBattleEnv(grid_size=10, render_mode=None)
    agent = QAgent()

    print("Entrenando a la IA... (2000 episodios)")
    for episode in range(2000):
        obs, _ = env_train.reset()
        terminated = False
        while not terminated:
            action = agent.choose_action(obs)
            next_obs, reward, terminated, _, _ = env_train.step(action)
            agent.learn(obs, action, reward, next_obs)
            obs = next_obs
        agent.epsilon = max(0.1, agent.epsilon * 0.999) 
    env_train.close()

    # 2. EVALUACIÓN CON RENDER VISUAL
    print("Entrenamiento completado. ¡Iniciando visualización!")
    env_visual = SubmarineBattleEnv(grid_size=10, render_mode="human")
    
    obs, _ = env_visual.reset()
    terminated = False
    
    # Quitamos la exploración para que el agente use solo lo que ha aprendido
    agent.epsilon = 0.0 
    
    while not terminated:
        action = agent.choose_action(obs)
        obs, reward, terminated, _, _ = env_visual.step(action)
        
    print("Partida finalizada. Cerrando ventana en 3 segundos...")
    plt.pause(3)
    env_visual.close()