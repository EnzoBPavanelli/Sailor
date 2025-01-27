#!/home/enzo/miniconda3/envs/esailor/bin/python
import numpy as np
import matplotlib.pyplot as plt

def repulsive_potential(x, y, obstacles, k):
    potential = 0
    for (ox, oy) in obstacles:
        dx = x - ox
        dy = y - oy
        distance = np.sqrt(dx ** 2 + dy ** 2)
        if distance >40:
            return 0
        if distance > 0:  # Evitar divisão por zero
            potential += k * (1 / distance)
        else:
            potential += k * (1 / (distance))
    return potential


def attractive_potential(x, y, goal, g):
    dx = x - goal[0]
    dy = y - goal[1]
    distance = np.sqrt(dx ** 2 + dy ** 2)
    return g * (distance)


def heading(candidate, current_pos):
    x_move = candidate[0] - current_pos[0]
    y_move = candidate[1] - current_pos[1]
    hdg = np.degrees(np.arctan2(y_move, x_move))
    return hdg


def hdg_towind(hdg, wind):
    adjusted_hdg = (hdg - wind + 360) % 360
    if adjusted_hdg > 180:
        adjusted_hdg -= 360
    return adjusted_hdg


def upwind_potential(current_pos, candidate, wind_direction, fiup, gup):
    dist = np.linalg.norm(candidate - current_pos)
    hdg = heading(candidate, current_pos)
    fi = abs(hdg_towind(hdg, wind_direction))
    if 0 <= fi <= fiup:
        pup = gup * (1 - (fi / fiup)) * dist
    else:
        pup = 0
    return pup

def downwind_potential(current_pos, candidate, wind_direction, fidown, gdown):
    dist = np.linalg.norm(candidate - current_pos)
    hdg = heading(candidate, current_pos)
    fi = abs(hdg_towind(hdg, wind_direction))
    if 180 - fidown <= fi <= 180:
        pdown = gdown * (1 - ((180 - fi) / fidown)) * dist
    else:
        pdown = 0
    return pdown


def hysteresis_potential(current_pos, candidate, wind_direction, fiup, fidown, gh):
    dist = np.linalg.norm(candidate - current_pos)
    hdg = heading(candidate, current_pos)
    fi = abs(hdg_towind(hdg, wind_direction))
    if fiup < fi < 180 - fidown:
        ph = gh * dist
    else:
        ph = 0
    return ph


def generate_waypoints_with_potential(start, goal, obstacles, wind):
    g, k, gup, gdown, gh, fiup, fidown = 15, 200, 30, 20, 15, 45, 20
    boat_dir=180
    man_cost = 1
    ang_cost = 90
    waypoints = [start]
    current_pos = np.array(start)
    goal = np.array(goal)
    potentials_data = []
    visited_positions = set()
    max_iterations = 1000
    iterations = 0
    last_pos = []
    candidate_positions=[(0.0, 5.0),(0.8675, 4.915),(1.71, 4.805),(2.5, 4.61),(3.29, 4.305),(4.08, 3.89),(4.805, 3.29),(5.475, 2.5),(6.05, 1.71),(6.505, 0.8675),(6.805, 0.0),(6.505, -0.8675),(6.05, -1.71),(5.475, -2.5),(4.805, -3.29),(4.08, -3.89),(3.29, -4.305),(2.5, -4.61),(1.71, -4.805),(0.8675, -4.915),(0.0, -5.0),(-0.8675, -4.915),(-1.71, -4.805),(-2.5, -4.61),(-3.29, -4.305),(-4.08, -3.89),(-4.805, -3.29),(-5.475, -2.5),(-6.05, -1.71),(-6.505, -0.8675),(-6.805, 0.0),(-6.505, 0.8675),(-6.05, 1.71),(-5.475, 2.5),(-4.805, 3.29),(-4.08, 3.89),(-3.29, 4.305),(-2.5, 4.61),(-1.71, 4.805),(-0.8675, 4.915),(0.0, 5.0)]
    # Loop principal para gerar waypoints
    multiplicador=2
    candidate_positions_scaled = [(x * multiplicador, y * multiplicador) for x, y in candidate_positions]

    while np.linalg.norm(current_pos - goal) > 8 and iterations < max_iterations:
        min_potential = float('inf')
        next_pos = current_pos

        # Testar todas as posições candidatas
        for dx, dy in candidate_positions_scaled:
            candidate = current_pos + np.array([dx, dy])

            # Verificar se a posição candidata é a mesma que a anterior
            if np.array_equal(candidate, last_pos):
                continue  # Ignorar a posição anterior para evitar voltar

            if tuple(candidate) in visited_positions:
                continue  # Ignorar posições já visitadas

            # Calcular o potencial total para essa posição candidata
            total_potential = (
                    attractive_potential(candidate[0], candidate[1], goal, g) +
                    repulsive_potential(candidate[0], candidate[1], obstacles, k)+
                    upwind_potential(current_pos, candidate, wind, fiup, gup)+
                    downwind_potential(current_pos, candidate, wind, fidown, gdown)+
                    hysteresis_potential(current_pos, candidate, wind, fiup, fidown, gh)
            )

            # Armazenar a posição candidata e seu valor de potencial
            potentials_data.append((candidate[0], candidate[1], total_potential))

            # Atualizar a posição com o menor potencial
            if total_potential < min_potential:
                min_potential = total_potential
                next_pos = candidate

        # Verificar se houve mudança de posição
        if np.array_equal(current_pos, next_pos):
            break

        # Adicionar o próximo waypoint e marcar como visitado
        waypoints.append(tuple(next_pos))
        visited_positions.add(tuple(next_pos))
        last_pos = current_pos
        current_pos = next_pos
        iterations += 1

    # Adicionar o ponto final (goal)
    waypoints.append(tuple(goal))
    return np.array(waypoints), potentials_data


# def plot_and_save_results(temporary_waypoints, obstacle_positions, real_positions):
#     fig, ax = plt.subplots(figsize=(10, 8))
#
#     if len(real_positions) > 0:
#         real_pos = np.array(real_positions)
#         ax.plot(real_pos[:, 0], real_pos[:, 1], color="red", label="Boat Trajectory")
#
#     # Ajustar os limites dos eixos
#     ax.set_xlim(0, 100)
#     ax.set_ylim(-100, 100)
#
#     # Adicionar grid customizado
#     ax.set_xticks(np.arange(0, 101, 20))  # Linhas verticais de 20 em 20 metros no eixo X
#     ax.set_yticks(np.arange(-100, 101, 20))  # Linhas horizontais de 20 em 20 metros no eixo Y
#     ax.grid(True, which='both', linestyle='-', linewidth=0.5, color='gray')  # Configuração do estilo do grid
#
#     if len(temporary_waypoints) > 0:
#         temp_wp = np.array(temporary_waypoints)
#         ax.plot(temp_wp[:, 0], temp_wp[:, 1], linestyle="--", color="green", label="Temporary Waypoints")
#         ax.scatter(temp_wp[:, 0], temp_wp[:, 1], color="green", marker="x", s=100, label="Temporary WP", zorder=5)
#
#     ax.scatter(0, 0, color="cyan", marker="o", s=200, label="Waypoint 0", zorder=6)
#
#     for i, obs_pos in enumerate(obstacle_positions):
#         ax.add_patch(plt.Rectangle((obs_pos[0] - 0.5, obs_pos[1] - 0.5), 1.5, 1.5, color='blue',
#                                    label="Obstacle" if i == 0 else "", zorder=4))
#
#     ax.legend(loc='lower right')
#     ax.set_xlabel("X Position")
#     ax.set_ylabel("Y Position")
#     ax.set_title("Real Path with Temporary Waypoints and Obstacles")
#
#     plt.savefig('apfplot.pdf')
#     plt.show()
def plot_and_save_results(temporary_waypoints, obstacle_positions, real_positions):

        avg_speed = 0

        # Plotar a trajetória, caminho A* e waypoints temporários
        fig, ax = plt.subplots(figsize=(10, 8))
        if len(real_positions) > 0:
            real_pos = np.array(real_positions)
            ax.plot(real_pos[:, 0], real_pos[:, 1], color="red", label="Boat Trajectory")

        # Ajustar os limites dos eixos para aumentar a área de plotagem
        ax.set_xlim(0, 120)  # Ajustar os limites do eixo X
        ax.set_ylim(-100, 100)  # Ajustar os limites do eixo Y

        # Plotar o waypoint principal
        ax.scatter(0, 0, color="cyan", marker="o", s=200, label="Waypoint 0", zorder=6)

        # Plotar os obstáculos
        for i, obs_pos in enumerate(obstacle_positions):
            ax.add_patch(plt.Rectangle((obs_pos[0] - 0.5, obs_pos[1] - 0.5), 1.5,
                                       1.5, color='blue', label="Obstacle" if i == 0 else "", zorder=4))
        ax.scatter(100,0, color="green",
                   marker="o", s=200, label=f"Waypoint {1}", zorder=6)



        ax.legend(loc='lower right')
        ax.grid(True)
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_title(
            f"Trajectory of the boat with PID agent and APF controller\nAverage speed: {avg_speed:.2f} m")
        plt.savefig('apfplot.pdf')
        plt.show()

def plot_before(temporary_waypoints, obstacle_positions):
    # Configurar o gráfico
    fig, ax = plt.subplots(figsize=(10, 8))

    # Verificar se há waypoints temporários
    if len(temporary_waypoints) > 0:  # Corrigido para evitar ambiguidade
        temp_wp = np.array(temporary_waypoints)
        ax.plot(temp_wp[:, 0], temp_wp[:, 1], linestyle="--", color="green", label="Temporary Waypoints")
        ax.scatter(temp_wp[:, 0], temp_wp[:, 1], color="green", marker="x", s=100, label="Temporary WP", zorder=5)


    # Plotar os obstáculos
    for i, obs_pos in enumerate(obstacle_positions):
        ax.add_patch(plt.Rectangle((obs_pos[0] - 0.5, obs_pos[1] - 0.5), 1.5, 1.5, color='blue',
                                   label="Obstacle" if i == 0 else "", zorder=2))

    # Configurações do gráfico
    ax.legend(loc='lower right')
    ax.grid(True, linestyle='--', linewidth=0.5, color='gray', zorder=1)
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_title("Real Path with Temporary Waypoints and Obstacles")

    # Exibir o gráfico
    plt.show()
def calcular_distancia_total(pos_barco):
        distancia_total = 0
        for i in range(1, len(pos_barco)):
            distancia_total += np.linalg.norm(pos_barco[i] - pos_barco[i - 1])
        return distancia_total

def main():
    start = (0.0, 0.0)
    goal = (100.0, 100.0)
    obstacles = [(20, 2), (20, 0), (20, -2), (20, -5), (20, -7), (20, -10), (25, 10), (30, -5)]
    wind_direction = 0
    wind = wind_direction - 180
    waypoints, potentials_data = generate_waypoints_with_potential(start, goal, obstacles, wind)
    print(waypoints)
    # plot_before(waypoints,obstacles)
    #boat_positions = np.array(pos_barco, dtype=np.float32)
    #distanciatotal = totaldist
    #print("DISTANCIA TOTAL:", distanciatotal)
    #print(f"Distância total percorrida pelo barco: {distanciatotal:.2f} unidades")
    #plot_and_save_results(waypoints, obstacles, boat_positions)



if __name__ == "__main__":
    main()
