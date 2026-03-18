# Submarine_AI
Practica Reinforcement Learning - Inteligencia Artificial


# AI – Batalla del Atlántico

Entorno de aprendizaje por refuerzo (Reinforcement Learning) inspirado en la Batalla del Atlántico de la Segunda Guerra Mundial, creado con Gymnasium.

## Idea principal

En lugar del clásico enfoque multiagente con agentes independientes que aprenden por separado, este proyecto utiliza control centralizado:

- Un único agente (la IA) actúa como comandante de la flota
- Observa todo el estado del tablero de forma global  
- Toma decisiones simultáneas para todos los submarinos al mismo tiempo

El objetivo es coordinar una flota de U-Boats para interceptar y hundir un carguero enemigo fuertemente escoltado.

## Entorno actual (v0)

- Gridworld discreto (por defecto 10×10, configurable)
- Entidades
  - 2 × U-Boats (submarinos) → controlados por la política de la IA
  - 1 × Carguero → objetivo principal (se mueve de forma determinista o estocástica)
  - 1 × Destructor → escolta protectora (patrulla, persigue o reacciona según reglas predefinidas)

- Los buques aliados (carguero + destructor) se mueven de manera automática siguiendo reglas fijas o patrones programados.
- La IA decide, en cada paso, la acción de **cada uno** de los submarinos (avanzar, girar, sumergirse, lanzar torpedo, permanecer quieto, etc.).

## Objetivos del proyecto

- Estudiar si el control centralizado es más eficiente que el aprendizaje multiagente independiente en escenarios de coordinación naval
- Explorar diferentes arquitecturas de red neuronal para el comandante (CNN + MLP, Transformer, etc.)
- Experimentar con recompensas densas vs. sparse, currículum learning y diferentes niveles de observabilidad parcial

