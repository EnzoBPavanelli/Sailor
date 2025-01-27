#!/home/enzo/miniconda3/envs/esailor/bin/python
import math
#-->PYTHON UTIL
import time
import random
import numpy as np
import glob
import sys
from datetime import datetime
import sys, os, signal, subprocess
from common import BoatNow
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.cbook import get_sample_data
from matplotlib.transforms import Affine2D
import matplotlib.image as mpimg
from simple_pid import PID
from matplotlib.patches import FancyArrowPatch
#-->ROS
import rospy

#-->GYM
import gymnasium as gym

#-->GAZEBO
import esailor_gym
from std_srvs.srv import Empty
from std_msgs.msg import Float32, Int16, Float32MultiArray
from gazebo_msgs.srv import SetModelState, GetModelState, SpawnModel, DeleteModel, SetPhysicsProperties, GetPhysicsProperties
from geometry_msgs.msg import Point, Pose, Vector3
from gazebo_msgs.msg import ODEPhysics, ModelState
from sensor_msgs.msg import LaserScan
from tf.transformations import quaternion_from_euler

#-->STABLE-BASELINES3
# from gymnasium import wrappers
from stable_baselines3 import PPO, SAC, A2C
from stable_baselines3.common.callbacks import CheckpointCallback

#-->PYTORCH
import torch as th


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

def manoeuvrability_cost(current_pos, candidate, man_cost, last_pos, ang_cost, wind):
    if last_pos == []:
        hdg1 = 180
        hdg2 = heading(current_pos, candidate)
        hdg1 = (hdg1 + 180) % 360 - 180
        hdg2 = (hdg2 + 180) % 360 - 180
        hdgtowind1 = hdg_towind(hdg1, wind)
        hdgtowind2 = hdg_towind(hdg2, wind)
        if hdgtowind1 > hdgtowind2:
            diff = abs(hdgtowind1 - hdgtowind2)
        else:
            diff = abs(hdgtowind2 - hdgtowind1)
            print("diff primeiro angulo :",diff)
        return diff*2
    else:
        hdg1 = heading(last_pos, current_pos)
        hdg2 = heading(current_pos, candidate)
        hdg1 = (hdg1 + 180) % 360 - 180
        hdg2 = (hdg2 + 180) % 360 - 180
        hdgtowind1 = hdg_towind(hdg1, wind)
        hdgtowind2 = hdg_towind(hdg2, wind)
        if hdgtowind1 > hdgtowind2:
            diff = hdgtowind1 - hdgtowind2
        else:
            diff = hdgtowind2 - hdgtowind1
            temp = hdgtowind1
            hdgtowind1 = hdgtowind2
            hdgtowind2 = temp
        if diff == 90 and abs(hdgtowind1) == 45 and abs(hdgtowind2) == 45:
            return man_cost * 0.5
        elif hdgtowind1 == 135 and hdgtowind2 == -135:
            return man_cost
        elif diff > 90:
            return man_cost * 1
        return 0
def angle_cost(current_pos,candidate,last_pos,boat_dir):
    def normalize_angle(angle):
        return (angle + 180) % 360 - 180

    def is_forward_movement(current_pos, candidate, reference_dir):
        """Verifica se o movimento é para frente, em relação à direção inicial."""
        move_dir = heading(current_pos, candidate)
        move_dir = normalize_angle(move_dir)
        reference_dir = normalize_angle(reference_dir)

        # Calcula a diferença angular
        angle_diff = abs(move_dir - reference_dir)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff

        # Movimentos para frente têm diferença de até 90°
        return angle_diff <= 90
    if last_pos == []:
        if not is_forward_movement(current_pos, candidate, boat_dir):
            return float('inf')  # Penaliza retrocesso

            # Penaliza desvios angulares no primeiro movimento
        hdg1 = normalize_angle(boat_dir)
        hdg2 = normalize_angle(heading(current_pos, candidate))
        diff = abs(hdg2 - hdg1)
        if diff > 180:
            diff = 360 - diff

        return diff *7
    else:  # Movimentos subsequentes
        hdg1 = normalize_angle(heading(last_pos, current_pos))
        hdg2 = normalize_angle(heading(current_pos, candidate))

        diff = abs(hdg2 - hdg1)
        if diff > 180:
            diff = 360 - diff

        return diff /4
def cos_custo(last_pos,current_pos, candidate):
    if last_pos == []:
        hdg1=0
        hdg2 = heading(current_pos, candidate)
        hdg1 = (hdg1 + 180) % 360 - 180
        hdg2 = (hdg2 + 180) % 360 - 180
        if hdg1 > hdg2:
            ang = hdg1 - hdg2
        else:
            ang = hdg2 - hdg1
        x = ang * np.pi / 180
        custo = math.cos(x**5)
        print("custo =", custo)
        return abs(custo * 100)
    else:
        hdg1 = heading(last_pos, current_pos)
        hdg2 = heading(current_pos, candidate)
        hdg1 = (hdg1 + 180) % 360 - 180
        hdg2 = (hdg2 + 180) % 360 - 180
        if hdg1 > hdg2:
            ang = hdg1 - hdg2
        else:
            ang = hdg2 - hdg1
    x=ang *np.pi/180
    custo=math.cos(x**2)
    print("custo =",custo)
    return abs(custo*20)
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
                    attractive_potential(candidate[0], candidate[1], goal, g)
                    + repulsive_potential(candidate[0], candidate[1], obstacles, k)
                    + upwind_potential(current_pos, candidate, wind, fiup, gup)
                    + downwind_potential(current_pos, candidate, wind, fidown, gdown)
                    # + hysteresis_potential(current_pos, candidate, wind, fiup, fidown, gh)
                    + angle_cost(current_pos, candidate, last_pos, boat_dir)
                    # + manoeuvrability_cost(current_pos, candidate, man_cost, last_pos, ang_cost, wind)
                    # + cos_custo(last_pos,current_pos,candidate)

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
# def plot_and_save_results(temporary_waypoints, obstacle_positions, real_positions):
#
#         avg_speed = 0
#
#         # Plotar a trajetória, caminho A* e waypoints temporários
#         fig, ax = plt.subplots(figsize=(10, 8))
#         if len(real_positions) > 0:
#             real_pos = np.array(real_positions)
#             ax.plot(real_pos[:, 0], real_pos[:, 1], color="red", label="Boat Trajectory")
#
#         # Ajustar os limites dos eixos para aumentar a área de plotagem
#         ax.set_xlim(0, 120)  # Ajustar os limites do eixo X
#         ax.set_ylim(-100, 100)  # Ajustar os limites do eixo Y
#
#         # Plotar o waypoint principal
#         ax.scatter(0, 0, color="cyan", marker="o", s=200, label="Waypoint 0", zorder=6)
#
#         # Plotar os obstáculos
#         for i, obs_pos in enumerate(obstacle_positions):
#             ax.add_patch(plt.Rectangle((obs_pos[0] - 0.5, obs_pos[1] - 0.5), 1.5,
#                                        1.5, color='blue', label="Obstacle" if i == 0 else "", zorder=4))
#         ax.scatter(100,0, color="green",
#                    marker="o", s=200, label=f"Waypoint {1}", zorder=6)
#
#
#
#         ax.legend(loc='lower right')
#         ax.grid(True)
#         ax.set_xlabel("X Position")
#         ax.set_ylabel("Y Position")
#         ax.set_title(
#             f"Trajectory of the boat with PID agent and APF controller\nAverage speed: {avg_speed:.2f} m")
#         plt.savefig('apfplot.pdf')
#         plt.show()
#
# def plot_before(temporary_waypoints, obstacle_positions):
#     # Configurar o gráfico
#     fig, ax = plt.subplots(figsize=(10, 8))
#
#     # Verificar se há waypoints temporários
#     if len(temporary_waypoints) > 0:  # Corrigido para evitar ambiguidade
#         temp_wp = np.array(temporary_waypoints)
#         ax.plot(temp_wp[:, 0], temp_wp[:, 1], linestyle="--", color="green", label="Temporary Waypoints")
#         ax.scatter(temp_wp[:, 0], temp_wp[:, 1], color="green", marker="x", s=100, label="Temporary WP", zorder=5)
#
#
#     # Plotar os obstáculos
#     for i, obs_pos in enumerate(obstacle_positions):
#         ax.add_patch(plt.Rectangle((obs_pos[0] - 0.5, obs_pos[1] - 0.5), 1.5, 1.5, color='blue',
#                                    label="Obstacle" if i == 0 else "", zorder=2))
#
#     # Configurações do gráfico
#     ax.legend(loc='lower right')
#     ax.grid(True, linestyle='--', linewidth=0.5, color='gray', zorder=1)
#     ax.set_xlabel("X Position")
#     ax.set_ylabel("Y Position")
#     ax.set_title("Real Path with Temporary Waypoints and Obstacles")
#
#     # Exibir o gráfico
#     plt.show()
# def calcular_distancia_total(pos_barco):
#         distancia_total = 0
#         for i in range(1, len(pos_barco)):
#             distancia_total += np.linalg.norm(pos_barco[i] - pos_barco[i - 1])
#         return distancia_total

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
