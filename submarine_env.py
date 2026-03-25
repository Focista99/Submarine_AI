import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
import sys
import time
import random

# =========================
# CONFIGURACIÓN VISUAL Y TÁCTICA
# =========================
GRID_SIZE = 30
CELL_SIZE = 25
WINDOW_SIZE = CELL_SIZE * GRID_SIZE
COAST_LINE_START = GRID_SIZE - 6 

def init_pygame():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("Batalla del Atlántico - IA RL (Q-Learning)")
    return screen

def load_image(filename, fallback_color):
    try:
        # Intenta cargar la imagen PNG y mantener las transparencias
        img = pygame.image.load(filename).convert_alpha()
        # Escala la imagen para que quepa en la celda (-2 para dejar un minimargen)
        img = pygame.transform.scale(img, (CELL_SIZE - 2, CELL_SIZE - 2))
        return img
    except FileNotFoundError:
        # Si no tienes la imagen, crea un cuadrado del color de reserva
        surf = pygame.Surface((CELL_SIZE - 2, CELL_SIZE - 2), pygame.SRCALPHA)
        pygame.draw.rect(surf, fallback_color, (0, 0, CELL_SIZE-2, CELL_SIZE-2), border_radius=4)
        return surf

# =========================
# ENTORNO DE SIMULACIÓN
# =========================
class SubmarineBattleEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, render_mode=None):
        super().__init__()
        self.grid_size = GRID_SIZE
        self.render_mode = render_mode
        self.action_space = spaces.MultiDiscrete([6, 6])
        self.observation_space = spaces.Box(low=-GRID_SIZE, high=GRID_SIZE, shape=(11,), dtype=np.int32)
        
        self.coast_line = self._generate_irregular_coast()
        self.screen = None
        self.reset()

    def _generate_irregular_coast(self):
        coast = []
        curr_x = COAST_LINE_START
        for _ in range(self.grid_size):
            curr_x += np.random.choice([-1, 0, 1])
            coast.append(np.clip(curr_x, self.grid_size - 8, self.grid_size - 1))
        return coast

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.coast_line = self._generate_irregular_coast()

        self.sub1 = [np.random.randint(0, 5), np.random.randint(20, 28)]
        self.sub2 = [np.random.randint(10, 15), np.random.randint(20, 28)]
        self.cargo = [0, np.random.randint(5, 20)]
        self.destroyer = [self.cargo[0] + 2, self.cargo[1]]
        
        self.sub1_hp, self.sub2_hp = 3, 3
        self.destroyer_hp = 6
        self.step_count = 0
        self.max_steps = 200 # Reducido para agilizar el RL
        self.game_over_msg = ""

        return self._get_obs(), {}

    def _get_obs(self):
        return np.array([
            self.sub1[0], self.sub1[1], self.sub1_hp,
            self.sub2[0], self.sub2[1], self.sub2_hp,
            self.cargo[0], self.cargo[1],
            self.destroyer[0], self.destroyer[1], self.destroyer_hp
        ], dtype=np.int32)

    def _dist(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _is_land(self, x, y):
        safe_y = int(np.clip(y, 0, self.grid_size - 1))
        return x >= self.coast_line[safe_y]

    def _smart_destroyer_logic(self):
        if self.destroyer_hp <= 0: return
        d1 = self._dist(self.destroyer, self.sub1) if self.sub1_hp > 0 else 999
        d2 = self._dist(self.destroyer, self.sub2) if self.sub2_hp > 0 else 999
        dist_c1 = self._dist(self.sub1, self.cargo) if self.sub1_hp > 0 else 999
        dist_c2 = self._dist(self.sub2, self.cargo) if self.sub2_hp > 0 else 999
        
        if min(dist_c1, dist_c2) > 10:
            target_x, target_y = self.cargo[0] + 2, self.cargo[1]
        else:
            target_x, target_y = (self.sub1[0], self.sub1[1]) if d1 <= d2 else (self.sub2[0], self.sub2[1])

        if self.destroyer[0] < target_x: self.destroyer[0] += 1
        elif self.destroyer[0] > target_x: self.destroyer[0] -= 1
        elif self.destroyer[1] < target_y: self.destroyer[1] += 1
        elif self.destroyer[1] > target_y: self.destroyer[1] -= 1
        
        if self._is_land(self.destroyer[0], self.destroyer[1]):
            self.destroyer[0] -= 1 

    def step(self, action):
        self.step_count += 1
        reward = -0.1 # Penalización constante para que aprendan a darse prisa
        terminated = False
        
        # 1. Movimiento Submarinos
        for i, (pos, hp) in enumerate([(self.sub1, self.sub1_hp), (self.sub2, self.sub2_hp)]):
            if hp > 0:
                old_pos = list(pos)
                act = action[i]
                if act == 0: pos[1] -= 1
                elif act == 1: pos[1] += 1
                elif act == 2: pos[0] -= 1
                elif act == 3: pos[0] += 1
                
                pos[0] = int(np.clip(pos[0], 0, self.grid_size - 1))
                pos[1] = int(np.clip(pos[1], 0, self.grid_size - 1))
                if self._is_land(pos[0], pos[1]):
                    pos[0], pos[1] = old_pos[0], old_pos[1]
                    reward -= 0.5 # Penalizar por chocar con tierra

        # 2. IA del Destructor
        self._smart_destroyer_logic()

        # 3. Movimiento Carguero
        if np.random.random() < 0.5:
            self.cargo[0] += 1
            self.cargo[1] = int(np.clip(self.cargo[1] + np.random.choice([-1, 0, 1]), 0, self.grid_size - 1))

        # 4. Combate Submarinos -> Destructor
        for i, (pos, hp) in enumerate([(self.sub1, self.sub1_hp), (self.sub2, self.sub2_hp)]):
            if hp > 0 and action[i] == 5 and self._dist(pos, self.destroyer) <= 1:
                self.destroyer_hp -= 1
                reward += 10 # RL Recompensa fuerte por dañar destructor
        
        # Combate Destructor -> Submarinos
        if self.destroyer_hp > 0:
            for pos, hp, idx in [(self.sub1, self.sub1_hp, 1), (self.sub2, self.sub2_hp, 2)]:
                if hp > 0 and self._dist(self.destroyer, pos) <= 1:
                    if idx == 1: self.sub1_hp -= 1
                    else: self.sub2_hp -= 1
                    reward -= 5 # RL Penalización por recibir daño

        # 5. RL Recompensa de Acercamiento (Shaping Reward)
        d_c1 = self._dist(self.sub1, self.cargo) if self.sub1_hp > 0 else 999
        d_c2 = self._dist(self.sub2, self.cargo) if self.sub2_hp > 0 else 999
        if min(d_c1, d_c2) < 5:
            reward += 1 # Premio por estar muy cerca del carguero

        # 6. Finales
        if self._is_land(self.cargo[0], self.cargo[1]):
            self.game_over_msg = "DERROTA: Inglaterra alcanzada"
            terminated = True
            reward -= 50
        elif (self.sub1_hp > 0 and self.sub1 == self.cargo) or (self.sub2_hp > 0 and self.sub2 == self.cargo):
            self.game_over_msg = "¡VICTORIA! Suministros hundidos"
            terminated = True
            reward += 150 # GORDO: Premio mayor
        elif self.destroyer_hp <= 0:
            self.game_over_msg = "¡VICTORIA! Escolta eliminada"
            terminated = True
            reward += 100
        elif self.sub1_hp <= 0 and self.sub2_hp <= 0:
            self.game_over_msg = "DERROTA: Lobos aniquilados"
            terminated = True
            reward -= 100
        elif self.step_count >= self.max_steps:
            self.game_over_msg = "EMPATE: Fin del tiempo"
            terminated = True

        if self.render_mode == "human": self.render()
        return self._get_obs(), reward, terminated, False, {}

    def render(self):
        if self.screen is None: 
            self.screen = init_pygame()
            # Cargar imágenes al iniciar la pantalla
            self.img_cargo = load_image("cargo.png.jpeg", (255, 215, 0)) # Amarillo si falla
            self.img_dest = load_image("dest.png", (255, 50, 50))   # Rojo si falla
            self.img_sub = load_image("sub.png", (0, 255, 120))     # Verde si falla
        for event in pygame.event.get():
            if event.type == pygame.QUIT: 
                pygame.quit()
                sys.exit()
        self.screen.fill((10, 30, 60))
        for y, x_b in enumerate(self.coast_line):
            rect = pygame.Rect(x_b * CELL_SIZE, y * CELL_SIZE, (self.grid_size - x_b) * CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(self.screen, (34, 139, 34), rect)
            pygame.draw.line(self.screen, (200, 220, 255), (x_b * CELL_SIZE, y * CELL_SIZE), (x_b * CELL_SIZE, (y+1) * CELL_SIZE), 2)

        def draw_sq(pos, color, txt_str):
            r = pygame.Rect(pos[0]*CELL_SIZE+2, pos[1]*CELL_SIZE+2, CELL_SIZE-4, CELL_SIZE-4)
            pygame.draw.rect(self.screen, color, r)
            f = pygame.font.SysFont("Arial", 12, bold=True)
            txt = f.render(txt_str, True, (255,255,255))
            self.screen.blit(txt, (pos[0]*CELL_SIZE+4, pos[1]*CELL_SIZE+4))

        draw_sq(self.cargo, (255, 215, 0), "C")
        if self.destroyer_hp > 0: draw_sq(self.destroyer, (255, 50, 50), f"D:{self.destroyer_hp}")
        if self.sub1_hp > 0: draw_sq(self.sub1, (0, 255, 120), f"S1:{self.sub1_hp}")
        if self.sub2_hp > 0: draw_sq(self.sub2, (0, 190, 255), f"S2:{self.sub2_hp}")

        pygame.draw.rect(self.screen, (0,0,0), (0, WINDOW_SIZE-40, WINDOW_SIZE, 40))
        font = pygame.font.SysFont("Verdana", 16)
        hud = font.render(f"PASO: {self.step_count} | RECOMPENSA IA: Oculta en juego", True, (255, 255, 255))
        self.screen.blit(hud, (15, WINDOW_SIZE - 30))

        if self.game_over_msg:
            overlay = pygame.Surface((WINDOW_SIZE, WINDOW_SIZE), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 190))
            self.screen.blit(overlay, (0,0))
            font_go = pygame.font.SysFont("Impact", 45)
            text = font_go.render(self.game_over_msg, True, (255, 255, 255))
            self.screen.blit(text, text.get_rect(center=(WINDOW_SIZE//2, WINDOW_SIZE//2)))

        pygame.display.flip()
        time.sleep(0.15)


# =========================
# AGENTE Q-LEARNING (El Cerebro)
# =========================
class QLearningBrain:
    def __init__(self, actions=6):
        self.q_table = {}
        self.alpha = 0.1      # Learning rate (Cuánto aprende en cada paso)
        self.gamma = 0.95     # Discount factor (Visión a largo plazo)
        self.epsilon = 1.0    # Empieza explorando 100% al azar
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.999
        self.actions = actions

    def _bin_dist(self, d):
        # Discretizamos distancias: -2(lejos neg), -1(cerca neg), 0(igual), 1(cerca pos), 2(lejos pos)
        if d < -5: return -2
        if d < 0: return -1
        if d == 0: return 0
        if d <= 5: return 1
        return 2

    def extract_state(self, env, sub_idx):
        """Traduce la coordenada global a un estado RELATIVO para la Q-Table"""
        pos = env.sub1 if sub_idx == 1 else env.sub2
        hp = env.sub1_hp if sub_idx == 1 else env.sub2_hp
        
        if hp <= 0:
            return "DEAD"

        dx_cargo = self._bin_dist(env.cargo[0] - pos[0])
        dy_cargo = self._bin_dist(env.cargo[1] - pos[1])
        dx_dest = self._bin_dist(env.destroyer[0] - pos[0])
        dy_dest = self._bin_dist(env.destroyer[1] - pos[1])
        
        # El estado es una tupla: (diferencia carguero X/Y, dif destructor X/Y, Mi_HP)
        return (dx_cargo, dy_cargo, dx_dest, dy_dest, hp)

    def choose_action(self, state):
        if state == "DEAD": return 4 # Quieto si está muerto
        
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.actions)

        # Epsilon-Greedy (Explorar vs Explotar)
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.randint(0, self.actions)
        else:
            return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state):
        if state == "DEAD": return
        
        if state not in self.q_table: self.q_table[state] = np.zeros(self.actions)
        if next_state not in self.q_table and next_state != "DEAD": 
            self.q_table[next_state] = np.zeros(self.actions)

        # Ecuación de Bellman
        max_future_q = 0 if next_state == "DEAD" else np.max(self.q_table[next_state])
        current_q = self.q_table[state][action]
        
        new_q = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)
        self.q_table[state][action] = new_q

# =========================
# BUCLE PRINCIPAL (ENTRENAMIENTO + JUEGO)
# =========================
if __name__ == "__main__":
    
    # 1. FASE DE ENTRENAMIENTO (Sin gráficos, muy rápido)
    print("\n" + "="*50)
    print("FASE 1: ENTRENANDO A LA IA CON Q-LEARNING")
    print("="*50)
    
    env_train = SubmarineBattleEnv(render_mode=None)
    brain = QLearningBrain()
    episodes = 3000 # Partidas de entrenamiento
    
    for ep in range(episodes):
        env_train.reset()
        done = False
        
        # Extraer estados iniciales
        s1 = brain.extract_state(env_train, 1)
        s2 = brain.extract_state(env_train, 2)
        
        while not done:
            a1 = brain.choose_action(s1)
            a2 = brain.choose_action(s2)
            
            _, reward, terminated, _, _ = env_train.step([a1, a2])
            done = terminated
            
            ns1 = brain.extract_state(env_train, 1)
            ns2 = brain.extract_state(env_train, 2)
            
            # Ambos submarinos aprenden en la misma tabla compartida
            brain.learn(s1, a1, reward, ns1)
            brain.learn(s2, a2, reward, ns2)
            
            s1, s2 = ns1, ns2
            
        # Reducir exploración (Que la IA confíe más en lo que aprende)
        if brain.epsilon > brain.epsilon_min:
            brain.epsilon *= brain.epsilon_decay
            
        if (ep + 1) % 500 == 0:
            print(f"Partida {ep+1}/{episodes} completada. Epsilon: {brain.epsilon:.2f} | Tamaño Memoria IA: {len(brain.q_table)} estados")


    # 2. FASE DE EXHIBICIÓN (Con gráficos)
    print("\n" + "="*50)
    print("FASE 2: LA IA TOMA EL CONTROL. ¡A JUGAR!")
    print("="*50)
    time.sleep(2)
    
    # Desactivamos el aprendizaje al azar, que use solo lo que ha aprendido
    brain.epsilon = 0.0 
    
    env_play = SubmarineBattleEnv(render_mode="human")
    
    try:
        while True:
            env_play.reset()
            done = False
            s1 = brain.extract_state(env_play, 1)
            s2 = brain.extract_state(env_play, 2)
            
            while not done:
                # La IA elige las mejores acciones según su Q-Table
                a1 = brain.choose_action(s1)
                a2 = brain.choose_action(s2)
                
                _, _, terminated, _, _ = env_play.step([a1, a2])
                done = terminated
                
                s1 = brain.extract_state(env_play, 1)
                s2 = brain.extract_state(env_play, 2)
                
            time.sleep(3) # Pausa para leer el mensaje final
            
    except KeyboardInterrupt:
        print("\nSimulación RL terminada.")
        pygame.quit()
        sys.exit()