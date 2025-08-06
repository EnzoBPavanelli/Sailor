#!/usr/bin/env python3
import numpy as np
import time
import math
import matplotlib.pyplot as plt
import rospy
from std_msgs.msg import Int16, Float32, Float32MultiArray
from sensor_msgs.msg import LaserScan
distanciatotal = 0
class captain():
    def __init__(self, frequence=2.0):
        # --------------------------------------------------------------------------------------
        # PARAMETROS
        self.truewind = [0, 6]
        # self.obstacles = [(50, 20), (50, 40), (50, 0)]
        #self.obstacles = [(50, 0), (50, 10), (50, -10), (70, 0), (80, 0)]
        self.windHdgGlobal = 90
        self.ultimo_angulo_navegacao = None
        # --------------------------------------------------------------------------------------
        self.cps = None
        self.apwind_speed = 0
        self.apwind_angle = 0
        self.ahead_dir = None
        self.boat_vel = np.zeros(2, dtype=np.float32)
        self.laser_scan = np.zeros(5, dtype=int)
        self.waypoints_list = None
        self.obstacles=[(-50,0)]
        # self.obstacles=[(50,0),(48,2),(75,-8)]
        #self.obstacles= [(75, -15), (75, 0), (75, 15), (75, 30),(50,-20), (75, 45), (75, 60), (75, 75), (75, 90), (90,-35),(75, 105), (75, 120), (120, 15), (120, 0), (120, -15), (120, -30), (120, -45), (120, -60), (120, -75), (120, -90), (120, -105), (120, -120)]
        self.pos_barco = []
        self.timestep = 1.0 / frequence
        self._cps_log = []
        self.K_ATT = 1
        self.K_REP = 150
        self.RHO_0 = 30.0
        # --- Parâmetros de Navegação e Vento ---
        self.WIND_DIRECTION_MET_DEG = 0
        self.UPWIND_NO_GO_HALF_ANGLE_DEG = 45.0
        self.DOWNWIND_NO_GO_HALF_ANGLE_DEG = 15.0
        # --- Parâmetros de Histerese (Diferenciados) ---
        self.GH_UPWIND = 40.0
        self.GH_DOWNWIND = 70.0
        # --- Parâmetros da Simulação ---
        self.STEP_SIZE = 0.1
        self.MAX_ITERS = 3000
        self.TOL = 2.1
        self.WIND_MATH_DEG = self.WIND_DIRECTION_MET_DEG
        self.UPWIND_MATH_DEG = (self.WIND_MATH_DEG + 180) % 360
        self.WIND_VECTOR_MATH = np.array(
            [np.cos(np.radians(self.WIND_MATH_DEG)), np.sin(np.radians(self.WIND_MATH_DEG))])
        self.WIND_PERP_LEFT = np.array([-self.WIND_VECTOR_MATH[1], self.WIND_VECTOR_MATH[0]])
        self.WIND_PERP_RIGHT = np.array([self.WIND_VECTOR_MATH[1], -self.WIND_VECTOR_MATH[0]])
        #
        # --> INITIALIZE THE ROS NODE
        success = False
        fails = 0
        while (not (success)) & (fails < 10):
            try:
                rospy.init_node(f"ecrew_captain", anonymous=False)
                success = True
            except:
                print("ROSMASTER is not running!")
                time.sleep(1)
                fails += 1
        l_scan = None
        model_namespace = "eboat"
        rospy.logdebug("Waiting for /scan to be READY...")
        while ((l_scan is None) and (not rospy.is_shutdown())):
            try:
                l_scan = rospy.wait_for_message(f"/{model_namespace}/laser/scan", LaserScan, timeout=1.0)
                rospy.logdebug(f"Current /{model_namespace}/laser/scan READY=>")
            except:
                rospy.logerr(f"Current /{model_namespace}/laser/scan not ready yet, retrying for getting laser_scan")
        # --> START CALLBACK FUNCTIONS TO LISTEN ROS TOPICS WITH SENSOR DATA
        rospy.Subscriber(f"/{model_namespace}/laser/scan", LaserScan, self._laser_scan_callback)
        rospy.Subscriber(f"/{model_namespace}/sensors/CPS", Float32MultiArray, self._CPS_callback)
        rospy.Subscriber(f"/{model_namespace}/sensors/wind_sensor", Float32MultiArray, self._wind_sensor_callback)
        rospy.Subscriber(f"/{model_namespace}/sensors/bow_vector", Float32MultiArray, self._boat_move_callback)
        self.nav_angle_pub = rospy.Publisher(f"/ecrew/captain/nav_angle", Float32, queue_size=1)
        self.propPwr_pub = rospy.Publisher(f"/eboat/control_interface/propulsion", Int16, queue_size=1)

    def _laser_scan_callback(self, data):
        laser_ranges = np.asarray(data.ranges)
        laser_ranges[laser_ranges == np.inf] = data.range_max

        # self.laser_scan[4] = np.min(laser_ranges[0:23])
        # self.laser_scan[3] = np.min(laser_ranges[24:47])
        # self.laser_scan[2] = np.min(laser_ranges[48:72])
        # self.laser_scan[1] = np.min(laser_ranges[73:96])
        # self.laser_scan[0] = np.min(laser_ranges[97:120])
        self.laser_scan_a = np.where(laser_ranges < 60)[0] * (-1)
        self.laser_scan_d = laser_ranges[self.laser_scan_a]
        self.laser_scan_a += 60
        print(self.laser_scan_a)
        print(self.laser_scan_d)

    def _wind_sensor_callback(self, msg):
        self.apwind_speed = msg.data[0]
        self.apwind_angle = msg.data[1]

    def _CPS_callback(self, msg):
        global path
        self.cps = np.array(msg.data)

    def _boat_move_callback(self, msg):
        self.ahead_dir = np.array(msg.data[:2])
        self.boat_vel = np.array(msg.data[3:])

    def loadMission(self, path2missionFile):
        self.waypoints_list = np.genfromtxt(path2missionFile, dtype=str, encoding=None, delimiter=",")

    def setMission(self, waypoints_list=np.array([[100.0, 0.0]], dtype=np.float32)):
        self.waypoints_list = waypoints_list

    def angle_between_vectors(self, u, v):
        dot_product = sum(i * j for i, j in zip(u, v))
        norm_u = math.sqrt(sum(i ** 2 for i in u))
        norm_v = math.sqrt(sum(i ** 2 for i in v))
        cos_theta = dot_product / (norm_u * norm_v)
        angle_rad = math.acos(cos_theta)
        angle_deg = math.degrees(angle_rad)
        return angle_rad, angle_deg

    def trueWindEstimation(self):
        theta = (self.apwind_angle - 180.0) * np.pi / 180.0
        apwind_vec = np.array([self.apwind_speed * np.cos(theta), self.apwind_speed * np.sin(theta)], dtype=np.float32)
        truewind = np.add(self.boat_vel[:2], apwind_vec)
        print("bvel: {}".format(self.boat_vel[:2]))
        _, truewind_angle = self.angle_between_vectors(np.array([1, 0]), truewind)
        return truewind, truewind_angle * np.sign(truewind[1])

    def attractive_force(self, pos, goal, k_att):
        """Calcula a força atrativa linear em direção ao objetivo."""
        return k_att * (goal - pos)

    def normalize(self,angle):
        """Normaliza ângulos para o intervalo [-180, 180)"""
        return ((angle + 180) % 360) - 180

    def angulosPossiveis(self,obstaclesHdg, windHdg):
        angles = list(range(-180, 180))
        angMinObs = 20
        # Remove ângulos de frente pro vento
        for i in range(45):
            angles = [a for a in angles if a != self.normalize(windHdg + i) and a != self.normalize(windHdg - i)]
        # Remove ângulos de cauda
        tailHdg = self.normalize(windHdg + 180)
        for i in range(30):
            angles = [a for a in angles if a != self.normalize(tailHdg + i) and a != self.normalize(tailHdg - i)]
        # Remove ângulos em torno dos obstáculos
        for x in obstaclesHdg:
            for i in range(angMinObs):
                angles = [a for a in angles if a != self.normalize(x + i) and a != self.normalize(x - i)]
        return angles

    def heading(self, candidate, current_pos):
        x_move = candidate[0] - current_pos[0]
        y_move = candidate[1] - current_pos[1]
        hdg = int(np.degrees(np.arctan2(y_move, x_move)))
        return self.normalize(hdg)
    def headingOBS(self, obstacles, current_pos):
        obsHeadings = []
        for x in obstacles:
            x_move = x[0] - current_pos[0]
            y_move = x[1] - current_pos[1]
            hdg = int(np.degrees(np.arctan2(y_move, x_move)))
            obsHeadings.append(self.normalize(hdg))
        return obsHeadings

    def varredura(self, angles, goalHdg, currentHdg):
        goalHdg = self.normalize(goalHdg)
        currentHdg = self.normalize(currentHdg)

        if not angles:
            return None

        if goalHdg in angles:
            self.ultimo_angulo_navegacao = goalHdg
            return goalHdg
        else:
            if self.ultimo_angulo_navegacao is not None:
                ult = self.normalize(self.ultimo_angulo_navegacao)
                candidatos = [
                    ang for ang in angles
                    if min(abs(ang - self.ultimo_angulo_navegacao), 360 - abs(ang - self.ultimo_angulo_navegacao)) <= 45
                ]
                print("Candidatos próximos da última navegação:", candidatos)
                if candidatos:
                    closest = min(candidatos, key=lambda x: abs(self.normalize(x - goalHdg)))
                    self.ultimo_angulo_navegacao = closest
                    print("Ângulo escolhido:", closest)
                    return closest
            closest_fallback = min(angles, key=lambda x: abs(self.normalize(x - goalHdg)))
            self.ultimo_angulo_navegacao = closest_fallback
            print("Ângulo fallback escolhido:", closest_fallback)
            return closest_fallback
    def distancia(self, p1, p2):
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def filtrar_obstaculos_relevantes(self, obstacles, current_pos, goal):
        dist_barco_ate_goal = self.distancia(current_pos, goal)
        obstaculos_filtrados = [
            obs for obs in obstacles
            if self.distancia(obs, current_pos) <= 40
               and self.distancia(obs, goal) < dist_barco_ate_goal
        ]

        return obstaculos_filtrados
    def normalize_angle(self,angle, currentHdg):
        return int((angle - currentHdg + 180) % 360)-180



    def engage(self):
        while np.any(self.cps) == None:
            # print(f"CPS: {self.cps}")
            # print(f"WP : {waypoint}")
            time.sleep(1)

        for k, waypoint in enumerate(self.waypoints_list):
            dist2goal = np.linalg.norm(np.subtract(waypoint, self.cps))
            steps = 0
            distancia_limite = 5 if k == len(self.waypoints_list) - 1 else 5
            while (dist2goal > distancia_limite) & (steps < 1000):
                print("Laser no engage:", self.laser_scan)
                truewind, truewind_angle = self.trueWindEstimation()
                true_line = np.subtract(waypoint, self.cps)
                dist2goal = np.linalg.norm(true_line)
                _, alpha = self.angle_between_vectors(np.array([1, 0]), self.ahead_dir)
                _, beta = self.angle_between_vectors(np.array([1, 0]), true_line)
                alpha *= np.sign(self.ahead_dir[1])
                beta *= np.sign(true_line[1])
                _, goal_angle = self.angle_between_vectors(self.ahead_dir,
                                                           true_line)  # --> angular "distance" of our current goal form the bow
                if (beta - alpha > 180) | ((alpha > beta) & (alpha - beta < 180)):
                    goal_angle *= -1
                else:
                    pass
                # self.propPwr_pub.publish(0)
                if abs(truewind_angle) > 136:  # --> truewind_angle is the true wind vector angle from the surge axis
                    self.propPwr_pub.publish(2)
                else:
                    self.propPwr_pub.publish(0)
                print("--------------------------------")
                # print("apwind_angle  : {:5.2f}".format(self.apwind_angle))
                # print("truewind_angle: {:5.2f}".format(truewind_angle))
                # print("goal_angle    : {:5.2f}".format(goal_angle))
                # print("angulo barco: {:5.2f}".format(alpha))
                # print("angulo objetivo: {:5.2f}".format(beta))
                # print("dist to goal  : {:5.2f}".format(dist2goal))
                print("Goiing ")

                theta = np.abs(truewind_angle - goal_angle)
                if theta > 180:
                    theta = 360 - theta
                else:
                    pass
                # print("theta         : {:5.2f}".format(theta))

                # if ((theta <= 135) & (theta >= 30)):
                #     nav_angle = goal_angle
                # elif theta > 135:
                #     nav_angle = np.sign(truewind_angle) * (abs(truewind_angle) - 135)
                # else:
                #     nav_angle = truewind_angle - np.sign(truewind_angle) * (30 - abs(truewind_angle))

                # print("nav_angle     : {:5.2f}".format(nav_angle))
                # self.nav_angle_pub.publish(np.float32(nav_angle))




                obstacles_filtrados = self.filtrar_obstaculos_relevantes(self.obstacles, self.cps, waypoint)
                obstaclesHdgGlobal = self.headingOBS(obstacles_filtrados, self.cps)
                print("obstacles hdg global", obstaclesHdgGlobal)
                goalHdgGlobal = self.heading(waypoint, self.cps)
                obstaclesHdg = [self.normalize_angle(h, alpha) for h in obstaclesHdgGlobal]
                print("obstacles hdg:", obstaclesHdg)
                goalHdg = self.normalize_angle(goalHdgGlobal, alpha)
                windHdg = self.normalize_angle(self.windHdgGlobal, alpha)
                angles = self.angulosPossiveis(obstaclesHdg, windHdg)
                angFinal = self.varredura(angles, goalHdg, alpha)
                if angFinal > 180:
                    angFinal -= 360
                print("ANGULO NOVO:", angFinal)
                # self.nav_angle_pub.publish(np.float32(anguloAPF))
                self.nav_angle_pub.publish(np.float32(angFinal))
                # angulo apf- angulo barco
                print("angulo publicado:", goal_angle)
                self.pos_barco.append(np.copy(self.cps))
                time.sleep(self.timestep)
                steps += 1

            print(self.pos_barco)

            def calcular_distancia_total(pos_barco):
                if len(pos_barco) < 2:
                    return 0  # Sem deslocamento se houver 0 ou 1 ponto

                distancia_total = 0

                for i in range(1, len(pos_barco)):
                    ponto_anterior = np.array(pos_barco[i - 1])
                    ponto_atual = np.array(pos_barco[i])
                    dist = np.linalg.norm(ponto_atual - ponto_anterior)  # Calcula a distância entre pontos consecutivos

                    print(
                        f"Movimento {i}: de {ponto_anterior} para {ponto_atual} -> Distância percorrida: {dist:.2f}")  # Depuração

                    distancia_total += dist  # Soma corretamente todas as movimentações

                print(f"\nDistância total percorrida pelo barco: {distancia_total:.2f} metros")  # Depuração final
                return distancia_total

            self.totaldist = calcular_distancia_total(self.pos_barco)


def repulsive_potential(x, y, obstacles, k):
    potential = 0
    for (ox, oy) in obstacles:
        dx = x - ox
        dy = y - oy
        distance = np.sqrt(dx ** 2 + dy ** 2)
        if distance > 40:
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
    def normalize_angle(angle):
        """Normaliza um ângulo para o intervalo [-180, 180]."""
        return (angle + 180) % 360 - 180

    if last_pos == []:
        hdg1 = 180  # Direção inicial padrão
    else:
        hdg1 = heading(last_pos, current_pos)  # Direção do último movimento

    hdg2 = heading(current_pos, candidate)  # Direção do próximo movimento

    # Normaliza os ângulos
    hdg1 = normalize_angle(hdg1)
    hdg2 = normalize_angle(hdg2)

    # Converte os ângulos para o sistema relativo ao vento
    hdgtowind1 = normalize_angle(hdg_towind(hdg1, wind))
    hdgtowind2 = normalize_angle(hdg_towind(hdg2, wind))

    # Calcula a menor diferença angular
    diff = abs(hdgtowind2 - hdgtowind1)
    if diff > 180:
        diff = 360 - diff  # Garante que a diferença não ultrapasse 180°

    # Detecta mudanças de bordo corretamente
    crossing_tack = (hdgtowind1 * hdgtowind2 < 0) and (abs(hdgtowind1) > 90 or abs(hdgtowind2) > 90)

    # Penalização de jibe e tack
    if crossing_tack:
        if abs(hdgtowind2) > 90:  # Jibe (mudança de bordo a favor do vento)
            return man_cost * 3.0  # Penalização fixa para jibe
        else:  # Tacking (mudança de bordo contra o vento)
            return man_cost * 5.0  # Penalização maior para tack

    return diff / 10  # Pequenos ajustes de rumo têm penalização menor


def angle_cost(current_pos, candidate, last_pos, boat_dir):
    def normalize_angle(angle):
        """Normaliza um ângulo para o intervalo [-180, 180]."""
        return (angle + 180) % 360 - 180

    def angle_difference(angle1, angle2):
        """Retorna a menor diferença angular entre dois ângulos normalizados."""
        diff = abs(angle1 - angle2)
        return min(diff, 360 - diff)

    def is_forward_movement(current_pos, candidate, reference_dir):
        """Verifica se o movimento está dentro de uma margem de 90° para frente."""
        move_dir = normalize_angle(heading(current_pos, candidate))
        reference_dir = normalize_angle(reference_dir)
        diff = angle_difference(move_dir, reference_dir)
        return diff <= 90  # Mantém movimento dentro de um ângulo natural

    # Se for a primeira iteração, verificar se está indo para frente
    if last_pos == []:
        if not is_forward_movement(current_pos, candidate, boat_dir):
            return float('inf')  # Penaliza fortemente movimentos para trás

        hdg1 = normalize_angle(boat_dir)
        hdg2 = normalize_angle(heading(current_pos, candidate))
        return angle_difference(hdg1, hdg2) / 1  # Reduz a penalização

    else:
        hdg1 = normalize_angle(heading(last_pos, current_pos))
        hdg2 = normalize_angle(heading(current_pos, candidate))
        return angle_difference(hdg1, hdg2) / 5  # Suaviza as mudanças de ângulo


def generate_waypoints_with_potential(start, goal, obstacles, wind):
    g, k, gup, gdown, gh, fiup, fidown = 15, 200, 30, 20, 15, 45, 20
    # g, k, gup, gdown, gh, fiup, fidown = 3, 100, 10, 5, 2, 60, 15
    boat_dir = 180
    man_cost = 0.3
    ang_cost = 90
    waypoints = [start]
    current_pos = np.array(start)
    goal = np.array(goal)
    potentials_data = []
    visited_positions = set()
    max_iterations = 1000
    iterations = 0
    last_pos = []
    candidate_positions = [(0.0, 5.0), (0.8675, 4.915), (1.71, 4.805), (2.5, 4.61), (3.29, 4.305), (4.08, 3.89),
                           (4.805, 3.29), (5.475, 2.5), (6.05, 1.71), (6.505, 0.8675), (6.805, 0.0), (6.505, -0.8675),
                           (6.05, -1.71), (5.475, -2.5), (4.805, -3.29), (4.08, -3.89), (3.29, -4.305), (2.5, -4.61),
                           (1.71, -4.805), (0.8675, -4.915), (0.0, -5.0), (-0.8675, -4.915), (-1.71, -4.805),
                           (-2.5, -4.61), (-3.29, -4.305), (-4.08, -3.89), (-4.805, -3.29), (-5.475, -2.5),
                           (-6.05, -1.71), (-6.505, -0.8675), (-6.805, 0.0), (-6.505, 0.8675), (-6.05, 1.71),
                           (-5.475, 2.5), (-4.805, 3.29), (-4.08, 3.89), (-3.29, 4.305), (-2.5, 4.61), (-1.71, 4.805),
                           (-0.8675, 4.915), (0.0, 5.0)]
    # Loop principal para gerar waypoints
    multiplicador = 1
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
                    + attractive_potential(candidate[0], candidate[1], goal, g)
                    + repulsive_potential(candidate[0], candidate[1], obstacles, k)
                    + upwind_potential(current_pos, candidate, wind, fiup, gup)
                    + downwind_potential(current_pos, candidate, wind, fidown, gdown)
                    + hysteresis_potential(current_pos, candidate, wind, fiup, fidown, gh)
                # + manoeuvrability_cost(current_pos, candidate, man_cost, last_pos, ang_cost, wind)
                # + angle_cost(current_pos,candidate,last_pos,boat_dir)
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

    fig, ax = plt.subplots(figsize=(10, 8))
    if len(real_positions) > 0:
        real_pos = np.array(real_positions)
        ax.plot(real_pos[:, 0], real_pos[:, 1], color="red", label="Boat Trajectory")

    ax.set_xlim(-25, 175)  # Ajustar os limites do eixo X
    ax.set_ylim(-100, 100)  # Ajustar os limites do eixo Y

    # ax.set_xlim(-75, 225)  # Ajustar os limites do eixo X
    # ax.set_ylim(-175, 125)  # Ajustar os limites do eixo Y

    ax.scatter(0, 0, color="cyan", marker="o", s=200, label="Waypoint 0", zorder=6)

    for i, obs_pos in enumerate(obstacle_positions):
        ax.add_patch(plt.Rectangle((obs_pos[0] - 0.5, obs_pos[1] - 0.5), 1.5,
                                   1.5, color='blue', label="Obstacle" if i == 0 else "", zorder=4))
    # ax.scatter(100,0, color="green",marker="o", s=200, label=f"Waypoint {1}", zorder=6)

    if len(temporary_waypoints) > 0:  # Corrigido para evitar ambiguidade
        temp_wp = np.array(temporary_waypoints)
        ax.scatter(temp_wp[:, 0], temp_wp[:, 1], color="green", marker="x", s=100, label="Temporary WP", zorder=5)

    # ax.legend(loc='lower right')
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
        ax.add_patch(plt.Rectangle((obs_pos[0] - 0.5, obs_pos[1] - 0.5), 1.5, 1.5, color='blue',label="Obstacle" if i == 0 else "", zorder=2))
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
    cap = captain()
    start = (0.0, 0.0)
    # goal = (0.0, 105.0)
    # obstacles = [(20, 2), (20, 0), (20, -2), (20, -5), (20, -7), (20, -10), (25, 10), (30, -5)]
    # obstacles = [(0,15), (15,15), (30,15), (45,15), (60,15)]
    wind_direction = 90
    wind = wind_direction - 180
    # waypoints, potentials_data = generate_waypoints_with_potential(start, goal, obstacles, wind)
    # waypoints=[(15,45), (15,60), (0,105)]

    # 1
    # goal = (150, -30)
    # obstacles = [(75, -15), (75, 0), (75, 15), (75, 30), (75, 45), (120, 15), (120, 0), (120, -15), (120, -30), (120, -45)]
    # wind_direction= 90
    # barbara:(somente com motor)
    # waypoints=[(0, 0), (30, -15), (60, -30), (90, -45), (120, -60), (150, -30)]

    # 2024:
    # waypoints=[(0, 0), (15, 0), (30, 0), (45, 0), (60, 0), (105, -15), (150, -30)]

    # 2010:
    # waypoints=[(0, 0), (30, 0), (60, 0), (90, -15), (105, -30), (135, -45), (150, -30)]
    # waypoints = [(0, 0), (15, 0), (30, 0), (45, 0), (60, 0),(60, 15), (60, 30), (60, 45), (60, 60), (75, 60),(90, 45), (105, 30), (120, 30), (135, 15), (150, 0),(150, 15), (135, 0), (150, -15), (165, -15),(150, -30), (150, -30)]

    # wind= 0
    # barbara:
    # waypoints=[(0, 0), (30, -30), (60, -60), (90, -90), (120, -60), (150, -30)]

    # 2024:
    # waypoints=[(0, 0), (30, -15), (75, -30), (105, -30), (150, -30)]

    # 2010:
    # waypoints=[(0, 0), (30, -30), (60, -60), (90, -90), (120, -60), (150, -30)]

    # 2
    goal = (100, 0)
    obstacles=[(-50,0)]
    #obstacles = [(75, -15), (75, 0), (75, 15), (75, 30), (75, 45), (75, 60), (50, -20), (90, -35), (75, 75), (75, 90), (75, 105), (75, 120), (120, 15), (120, 0), (120, -15), (120, -30), (120, -45), (120, -60), (120, -75),(120, -90), (120, -105), (120, -120)]
    # wind=0
    #obstacles = [(50, 0), (50, 10), (50, -10), (70, 0), (80, 0)]
    waypoints=[(0,0),(400,0)]
    # obstacles=[(50,50)]
    # barbara:(somente com motor)
    # waypoints =[(0, 0), (15, -30), (30, -60), (45, -90), (75, -120), (105, -150), (135, -120), (165, -90), (180, -60), (150, -30)]

    # 2024:
    # waypoints =[(0, 0), (30, -15), (75, -30), (105, -45), (150, -30)]
    #waypoints = [(0, 0), (150, 0)]

    # 2010:
    # waypoints =[(0, 0), (30, -30), (60, -60), (90, -90), (105, -120), (135, -150), (150, -135), (150, -105), (150, -75), (150, -45), (150, -30)]

    # wind=90
    # barbara:
    # waypoints =[(0, 0), (15, 0), (45, -15), (75, -30), (105, -45), (135, -30), (150, -30)]

    # 2024:
    # waypoints =[(0, 0), (15, 0), (30, 0), (45, 0), (60, 0), (105, -15), (150, -30)]

    # 2010:
    # waypoints =[(0, 0), (15, 0), (45, -15), (75, -30), (105, -45), (135, -30), (150, -30)]

    # plot_before(waypoints, obstacles)
    print("Waypoints gerados :")
    for wp in waypoints:
        print(wp)
    # plot_before(waypoints,obstacles)
    cap.setMission(np.array(waypoints, dtype=np.float32))
    cap.engage()
    boat_positions = np.array(cap.pos_barco, dtype=np.float32)
    distanciatotal = cap.totaldist
    print("DISTANCIA TOTAL:", distanciatotal)
    print(f"Distância total percorrida pelo barco: {distanciatotal:.2f} unidades")
    plot_and_save_results(waypoints, obstacles, boat_positions)

if __name__ == "__main__":
    main()