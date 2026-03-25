import numpy as np
import gymnasium as gym
from gymnasium import spaces


class SubmarineBattleEnv(gym.Env):
    """
    Entorno básico de batalla submarina en un grid NxN.
    Observación: [x1,y1, x2,y2, xc,yc, xd,yd]
    Acciones: MultiDiscrete([5,5]) para dos submarinos.
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, render_mode=None, grid_size: int = 10):  # <-- 30 → 10
        super().__init__()
        self.grid_size = int(grid_size)
        self.render_mode = render_mode

        # Penalización fija por paso (ya no hace falta escalar porque el grid
        # de referencia y el actual coinciden en el valor por defecto).
        self.step_penalty = -1.0

        # 2 submarinos, cada uno con 5 acciones:
        # 0=arriba, 1=abajo, 2=izquierda, 3=derecha, 4=quieto
        self.action_space = spaces.MultiDiscrete([5, 5])

        # [x1,y1, x2,y2, xc,yc, xd,yd], cada coordenada entre 0 y grid_size-1
        self.observation_space = spaces.Box(
            low=0,
            high=self.grid_size - 1,
            shape=(8,),
            dtype=np.int32,
        )

        # Posiciones internas (se inicializan en reset)
        self.sub1 = None
        self.sub2 = None
        self.cargo = None
        self.destroyer = None

    def _get_obs(self):
        """Construye la observación actual en formato vector de 8 enteros."""
        return np.array(
            [
                self.sub1[0], self.sub1[1],
                self.sub2[0], self.sub2[1],
                self.cargo[0], self.cargo[1],
                self.destroyer[0], self.destroyer[1],
            ],
            dtype=np.int32,
        )

    def _clip_position(self, pos):
        """Asegura que una posición permanezca dentro del grid [0, grid_size-1]."""
        pos[0] = int(np.clip(pos[0], 0, self.grid_size - 1))
        pos[1] = int(np.clip(pos[1], 0, self.grid_size - 1))

    def _move_submarine(self, pos, action):
        """Mueve un submarino según la acción discreta indicada."""
        if action == 0:      # arriba
            pos[1] -= 1
        elif action == 1:    # abajo
            pos[1] += 1
        elif action == 2:    # izquierda
            pos[0] -= 1
        elif action == 3:    # derecha
            pos[0] += 1
        elif action == 4:    # quieto
            pass
        self._clip_position(pos)

    def _manhattan_distance(self, a, b):
        """Distancia Manhattan entre dos posiciones (x,y)."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _move_destroyer_towards_nearest_sub(self):
        """
        Mueve el destructor 1 casilla hacia el submarino más cercano.
        Heurística simple: primero alinear fila (x), luego columna (y).
        """
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

    def reset(self, *, seed=None, options=None):
        """
        Reinicia el entorno a posiciones fijas iniciales.
        - Submarinos: abajo a la izquierda
        - Carguero: arriba a la izquierda
        - Destructor: cerca del carguero
        """
        super().reset(seed=seed)

        # Posiciones iniciales coherentes con el tamaño del grid.
        self.sub1 = [0, self.grid_size - 1]
        self.sub2 = [1, self.grid_size - 1]
        self.cargo = [0, 0]
        self.destroyer = [1, 0]

        obs = self._get_obs()
        info = {}

        if self.render_mode == "human":
            self.render()

        return obs, info

    def step(self, action):
        """
        Ejecuta un paso del entorno:
        1) Mueve submarinos por acción del agente
        2) Mueve carguero a la derecha (+1 en x)
        3) Mueve destructor hacia submarino más cercano
        4) Evalúa recompensas y terminación
        """
        # Penalización base por paso para incentivar rapidez.
        reward = self.step_penalty
        terminated = False
        truncated = False

        # Acciones para cada submarino
        a1, a2 = int(action[0]), int(action[1])
        self._move_submarine(self.sub1, a1)
        self._move_submarine(self.sub2, a2)

        # Carguero: siempre avanza 1 casilla a la derecha
        self.cargo[0] = min(self.cargo[0] + 1, self.grid_size - 1)

        # Destructor: avanza hacia el submarino más cercano
        self._move_destroyer_towards_nearest_sub()

        # Reglas de terminación/recompensa (orden de prioridad simple)
        if self.sub1 == self.cargo or self.sub2 == self.cargo:
            reward += 100
            terminated = True
        elif self.sub1 == self.destroyer or self.sub2 == self.destroyer:
            reward -= 50
            terminated = True
        elif self.cargo[0] == self.grid_size - 1:
            reward -= 80
            terminated = True

        obs = self._get_obs()
        info = {}

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        """Render en texto para modo human."""
        if self.render_mode != "human":
            return

        grid = [["." for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        # Colocar entidades en el grid.
        grid[self.cargo[1]][self.cargo[0]] = "C"
        grid[self.destroyer[1]][self.destroyer[0]] = "D"
        grid[self.sub1[1]][self.sub1[0]] = "U"
        grid[self.sub2[1]][self.sub2[0]] = "U"

        print("\n" + "=" * 25)
        print("SubmarineBattle-v0")
        for row in grid:
            print(" ".join(row))
        print("=" * 25)

    def close(self):
        """Cierre del entorno (sin recursos externos en esta versión)."""
        pass


# Registro opcional para usar gym.make("SubmarineBattle-v0")
gym.register(
    id="SubmarineBattle-v0",
    entry_point=SubmarineBattleEnv,
)


if __name__ == "__main__":
    # Test mínimo: ejecutar un episodio aleatorio para validar que no hay errores.
    env = gym.make("SubmarineBattle-v0", render_mode="human")
    obs, info = env.reset()

    done = False
    total_reward = 0
    step_count = 0

    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
        done = terminated or truncated

    print(f"Episodio finalizado en {step_count} pasos.")
    print(f"Recompensa total: {total_reward}")
    env.close()
    