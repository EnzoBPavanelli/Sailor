import numpy as np
import matplotlib.pyplot as plt
import math

# ==============================================================================
# --- PARÂMETROS E CONSTANTES ---
# ==============================================================================

# --- Parâmetros APF ---
K_ATT = 80.0
K_REP = 50000.0
RHO_0 = 40.0

# --- Parâmetros de Navegação e Vento ---
WIND_DIRECTION_MET_DEG = 0.0
UPWIND_NO_GO_HALF_ANGLE_DEG = 45.0
DOWNWIND_NO_GO_HALF_ANGLE_DEG = 15.0

# --- Parâmetros de Histerese (Diferenciados) ---
GH_UPWIND = 300.0
GH_DOWNWIND = 250.0

# --- Parâmetros da Simulação ---
STEP_SIZE = 0.1
MAX_ITERS = 3000
TOL = 2.1

# --- Configuração Inicial ---
X_ROBOT_START = np.array([0.0, 0.0])
X_GOAL = np.array([100.0, 100.0])
X_OBSTACLES = [
    np.array([50.0, 50.0]), np.array([30.0, 60.0]), np.array([60.0, 80.0]),
    np.array([80.0, 80.0]), np.array([80,60])
]

# ==============================================================================
# --- CÁLCULOS DERIVADOS DE VENTO ---
# ==============================================================================

WIND_MATH_DEG = WIND_DIRECTION_MET_DEG
UPWIND_MATH_DEG = (WIND_MATH_DEG + 180) % 360
WIND_VECTOR_MATH = np.array([np.cos(np.radians(WIND_MATH_DEG)), np.sin(np.radians(WIND_MATH_DEG))])
WIND_PERP_LEFT = np.array([-WIND_VECTOR_MATH[1], WIND_VECTOR_MATH[0]])
WIND_PERP_RIGHT = np.array([WIND_VECTOR_MATH[1], -WIND_VECTOR_MATH[0]])

print("--- Configuração do Vento ---")
print(f"Input Vento (Interpretado como Mat.): {WIND_DIRECTION_MET_DEG}°")
print(f"Direção Downwind (Mat.): {WIND_MATH_DEG:.1f}°")
print(f"Direção Upwind (Mat.): {UPWIND_MATH_DEG:.1f}°")
print(f"Zona Upwind: {UPWIND_MATH_DEG:.1f}° +/- {UPWIND_NO_GO_HALF_ANGLE_DEG}°")
print(f"Zona Downwind: {WIND_MATH_DEG:.1f}° +/- {DOWNWIND_NO_GO_HALF_ANGLE_DEG}°")
# Imprime ambos os ganhos
print(f"Ganho Histerese (Gh Upwind): {GH_UPWIND}")
print(f"Ganho Histerese (Gh Downwind): {GH_DOWNWIND}")
print("-" * 30)

# ==============================================================================
# --- FUNÇÕES DE FORÇA E POTENCIAL ---
# ==============================================================================

def attractive_force(pos, goal, k_att):
    """Calcula a força atrativa linear em direção ao objetivo."""
    return k_att * (goal - pos)

def repulsive_force(pos, obstacles, k_rep, rho_0):
    """Calcula a força repulsiva total devido aos obstáculos próximos."""
    F_rep = np.zeros(2)
    if not obstacles: return F_rep
    for obs in obstacles:
        diff = pos - obs
        rho = np.linalg.norm(diff)
        if 0 < rho <= rho_0:
            if rho < 1e-6: continue
            gradient_magnitude = k_rep * ((1/rho) - (1/rho_0)) * (1 / rho**2)
            F_rep += gradient_magnitude * (diff / rho)
    return F_rep

# ---- FUNÇÃO MODIFICADA ----
def calculate_hysteresis_force(current_heading_deg, upwind_math_deg, Gh_upwind, Gh_downwind):
    """
    Calcula a força de histerese com ganhos diferenciados para upwind/downwind.
    """
    Fh = np.zeros(2)
    # Calcula ângulo do vento relativo ao heading ATUAL
    relative_wind_deg = shortest_angle_diff(current_heading_deg, upwind_math_deg)

    # Determina qual ganho usar baseado no ângulo relativo
    # Se |ângulo| <= 90, considera upwind/reaching, usa Gh_upwind
    # Se |ângulo| > 90, considera downwind/reaching, usa Gh_downwind
    if abs(relative_wind_deg) <= 90.0:
        current_Gh = Gh_upwind
        # print(f"Iter using Gh_upwind = {current_Gh}") # Debug opcional
    else:
        current_Gh = Gh_downwind
        # print(f"Iter using Gh_downwind = {current_Gh}") # Debug opcional


    # Aplica força lateral baseada no bordo com o ganho apropriado
    if relative_wind_deg > 1e-6 :      # Bombordo (Vento relativo da Esquerda)
        Fh = current_Gh * WIND_PERP_LEFT      # Força para Esquerda
    elif relative_wind_deg < -1e-6:    # Estibordo (Vento relativo da Direita)
        Fh = current_Gh * WIND_PERP_RIGHT     # Força para Direita

    return Fh
# ---- FIM DA MODIFICAÇÃO ----

# ==============================================================================
# --- FUNÇÕES AUXILIARES DE ÂNGULO E RESTRIÇÕES ---
# ==============================================================================
# (calculate_angle_deg, shortest_angle_diff, is_in_no_go_zone,
#  find_closest_edge_heading, apply_no_go_constraints - sem alterações)
def calculate_angle_deg(vector):
    if np.linalg.norm(vector) < 1e-6: return 0.0
    angle_rad = np.arctan2(vector[1], vector[0]) # Y, X
    angle_deg = np.degrees(angle_rad)
    return (angle_deg + 360) % 360

def shortest_angle_diff(angle1_deg, angle2_deg):
    diff = (angle2_deg - angle1_deg + 180) % 360 - 180
    return diff

def is_in_no_go_zone(heading_deg, zone_center_deg, zone_half_angle):
    diff = shortest_angle_diff(zone_center_deg, heading_deg)
    return abs(diff) <= zone_half_angle + 1e-9

def find_closest_edge_heading(heading_deg, zone_center_deg, zone_half_angle):
    diff = shortest_angle_diff(zone_center_deg, heading_deg)
    if diff >= 0: return (zone_center_deg + zone_half_angle) % 360
    else: return (zone_center_deg - zone_half_angle + 360) % 360

def apply_no_go_constraints(desired_heading_deg, desired_direction):
    final_heading_deg = desired_heading_deg
    final_direction = desired_direction
    adjusted = False
    if is_in_no_go_zone(desired_heading_deg, UPWIND_MATH_DEG, UPWIND_NO_GO_HALF_ANGLE_DEG):
        final_heading_deg = find_closest_edge_heading(desired_heading_deg, UPWIND_MATH_DEG, UPWIND_NO_GO_HALF_ANGLE_DEG)
        adjusted = True
    elif is_in_no_go_zone(desired_heading_deg, WIND_MATH_DEG, DOWNWIND_NO_GO_HALF_ANGLE_DEG):
        final_heading_deg = find_closest_edge_heading(desired_heading_deg, WIND_MATH_DEG, DOWNWIND_NO_GO_HALF_ANGLE_DEG)
        adjusted = True
    if adjusted:
        final_heading_rad = np.radians(final_heading_deg)
        final_direction = np.array([np.cos(final_heading_rad), np.sin(final_heading_rad)])
    return final_direction, final_heading_deg

# ==============================================================================
# --- FUNÇÃO PRINCIPAL DA SIMULAÇÃO ---
# ==============================================================================

def run_simulation():
    """Executa a simulação de planejamento de trajetória."""
    x_robot = X_ROBOT_START.copy()
    trajectory = [x_robot.copy()]
    headings_deg_history = []
    initial_goal_dir = X_GOAL - x_robot
    current_heading_deg = calculate_angle_deg(initial_goal_dir) if np.linalg.norm(initial_goal_dir) > 1e-6 else 0.0
    headings_deg_history.append(current_heading_deg)

    print("Iniciando simulação...")
    goal_reached = False
    for i in range(MAX_ITERS):
        F_att = attractive_force(x_robot, X_GOAL, K_ATT) # Chamada corrigida
        F_rep = repulsive_force(x_robot, X_OBSTACLES, K_REP, RHO_0)
        # Passa ambos os ganhos para a função
        Fh = calculate_hysteresis_force(current_heading_deg, UPWIND_MATH_DEG, GH_UPWIND, GH_DOWNWIND)
        F_total = F_att + F_rep + Fh
        norm_F_total = np.linalg.norm(F_total)

        if norm_F_total > 1e-6:
            desired_direction = F_total / norm_F_total
        else:
            if np.linalg.norm(x_robot - X_GOAL) <= TOL:
                print(f"\nForça zero no objetivo ou próximo. Parando na iteração {i+1}.")
                goal_reached = True
                break
            else:
                desired_direction = np.zeros(2)

        desired_heading_deg = calculate_angle_deg(desired_direction)

        if np.linalg.norm(desired_direction) > 1e-6:
            final_direction, final_heading_deg = apply_no_go_constraints(
                desired_heading_deg, desired_direction
            )
        else:
            final_direction = desired_direction
            final_heading_deg = current_heading_deg

        x_robot = x_robot + STEP_SIZE * final_direction
        trajectory.append(x_robot.copy())
        current_heading_deg = final_heading_deg
        headings_deg_history.append(current_heading_deg)

        if np.linalg.norm(x_robot - X_GOAL) <= TOL:
            print(f"\nObjetivo alcançado (por distância) na iteração {i+1}!")
            goal_reached = True
            break

    if not goal_reached:
        print(f"\nObjetivo não alcançado após {MAX_ITERS} iterações.")

    while len(headings_deg_history) < len(trajectory):
        headings_deg_history.append(headings_deg_history[-1] if headings_deg_history else 0.0)

    return np.array(trajectory), np.array(headings_deg_history)

# ==============================================================================
# --- FUNÇÃO DE PLOTAGEM ---
# ==============================================================================
def plot_results(trajectory, headings):
    plt.figure(figsize=(11, 10))
    if X_OBSTACLES:
        for i_obs, obs in enumerate(X_OBSTACLES):
            plt.plot(obs[0], obs[1], 'ro', markersize=10, label='Obstáculo' if i_obs == 0 else "")
    plt.plot(trajectory[:, 0], trajectory[:, 1], '-b', label='Trajetória')
    plt.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=10, label='Início')
    plt.plot(X_GOAL[0], X_GOAL[1], 'gx', markersize=12, label='Objetivo')
    # Ajusta posição inicial da seta do vento dinamicamente
    plot_min_x = min(trajectory[:,0].min(), X_ROBOT_START[0]) - 10
    plot_min_y = min(trajectory[:,1].min(), X_ROBOT_START[1]) - 10
    wind_arrow_start = np.array([plot_min_x + 5, plot_min_y + 5])
    arrow_scale = max(X_GOAL[0], X_GOAL[1]) / 10.0 # Escala da seta do vento
    plt.arrow(wind_arrow_start[0], wind_arrow_start[1],
              WIND_VECTOR_MATH[0] * arrow_scale, WIND_VECTOR_MATH[1] * arrow_scale,
              head_width=arrow_scale*0.3, head_length=arrow_scale*0.5, fc='cyan', ec='blue',
              label=f'Vento ({WIND_DIRECTION_MET_DEG}° Input)')
    plt.xlabel('X')
    plt.ylabel('Y')
    # Atualiza título para refletir os dois ganhos
    plt.title(f'APF c/ Histerese (Gh Up={GH_UPWIND}, Down={GH_DOWNWIND}), Vento ({WIND_DIRECTION_MET_DEG}°) '
              f'e No-Go ({UPWIND_NO_GO_HALF_ANGLE_DEG*2}° Up, {DOWNWIND_NO_GO_HALF_ANGLE_DEG*2}° Down)')
    plt.legend(loc='best')
    plt.grid(True)
    plt.axis('equal')
    plt.xlim(plot_min_x, max(trajectory[:,0].max(), X_GOAL[0]) + 10)
    plt.ylim(plot_min_y, max(trajectory[:,1].max(), X_GOAL[1]) + 10)
    plt.show()

# ==============================================================================
# --- EXECUÇÃO ---
# ==============================================================================

if __name__ == "__main__":
    final_trajectory, final_headings = run_simulation()
    plot_results(final_trajectory, final_headings)