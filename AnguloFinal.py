import numpy as np
import matplotlib.pyplot as plt

def rot(V, theta):
    x = V[0] * np.cos(theta) - V[1] * np.sin(theta)
    y = V[0] * np.sin(theta) + V[1] * np.cos(theta)
    return [x, y]

def angBetween2Vecs(V1, V2):
    dot = V1[0] * V2[0] + V1[1] * V2[1]
    return np.arccos(dot / (np.sqrt(V1[0]**2 + V1[1]**2) * np.sqrt(V2[0]**2 + V2[1]**2)))

def heading(candidate, current_pos):
    x_move = candidate[0] - current_pos[0]
    y_move = candidate[1] - current_pos[1]
    hdg = int(np.degrees(np.arctan2(y_move, x_move)))
    return (hdg + 360) % 360

def headingOBS(obstacles, current_pos):
    obsHeadings = []
    for x in obstacles:
        x_move = x[0] - current_pos[0]
        y_move = x[1] - current_pos[1]
        hdg = int(np.degrees(np.arctan2(y_move, x_move)))
        obsHeadings.append((hdg + 360) % 360)
    return obsHeadings

def varredura(angles, goalHdg):
    if not angles:
        return None
    closest_angle = min(angles, key=lambda x: min(abs(x - goalHdg), 360 - abs(x - goalHdg)))
    return closest_angle

def angulosPossiveis(obstaclesHdg, windHdg):
    angles = list(range(360))
    angMinObs = 10

    for i in range(45):
        angles = [a for a in angles if a != (windHdg + i) % 360 and a != (windHdg - i) % 360]
    tailHdg = (windHdg + 180) % 360
    for i in range(15):
        angles = [a for a in angles if a != (tailHdg + i) % 360 and a != (tailHdg - i) % 360]
    for x in obstaclesHdg:
        for i in range(angMinObs):
            angles = [a for a in angles if a != (x + i) % 360 and a != (x - i) % 360]

    return angles

def normalize_angle(angle, currentHdg):
    return (angle - currentHdg + 360) % 360

def distancia(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def filtrar_obstaculos_relevantes(obstacles, current_pos, goal):
    dist_barco_ate_goal = distancia(current_pos, goal)
    return [obs for obs in obstacles
            if distancia(obs, current_pos) <= dist_barco_ate_goal]


if __name__ == "__main__":
    #--------------------------------------------------------------------------------------
    # PARAMETROS
    truewind = [0, 6]
    goal = [100,-10]
    #obstacles = [(50, 20), (50, 40), (50, 0)]
    obstacles=[(50,-50),(50,60)]
    currentPos = [0, 0]
    currentHdg = 0
    windHdgGlobal = 0
    #--------------------------------------------------------------------------------------
    obstacles_filtrados = filtrar_obstaculos_relevantes(obstacles, currentPos, goal)
    obstaclesHdgGlobal = headingOBS(obstacles_filtrados, currentPos)
    goalHdgGlobal = heading(goal, currentPos)
    obstaclesHdg = [normalize_angle(h, currentHdg) for h in obstaclesHdgGlobal]
    goalHdg = normalize_angle(goalHdgGlobal, currentHdg)
    windHdg = normalize_angle(windHdgGlobal, currentHdg)
    angles = angulosPossiveis(obstaclesHdg, windHdg)
    angFinal = varredura(angles, goalHdg)
    print(angFinal)


    fig = plt.figure(figsize=[6, 6])
    ax = fig.add_subplot(111, projection='polar')
    for h in obstaclesHdg:
        ax.plot(np.deg2rad(h), 1, marker='o', markersize=15, color='red')
    if angFinal is not None:
        ax.quiver(0, 0, np.deg2rad(angFinal), 1, angles='xy',
                  scale_units='xy', scale=1, width=0.012, color='black')
    ax.plot(np.deg2rad(goalHdg), 1, marker='o', markersize=15, color='orange')
    ax.quiver(0, 0, np.deg2rad(windHdg), 1, angles='xy',
              scale_units='xy', scale=1, width=0.012, color='blue')
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xticks(np.arange(0, 2 * np.pi, np.pi / 6))
    ax.set_xticklabels(['0', '30', '60', '90', '120', '150', '180', '210', '240', '270', '300', '330'])
    ax.set_ylim(0, 1.2)

    plt.show()
