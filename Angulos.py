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
    return hdg
def headingOBS(obstacles, current_pos):
    obsHeadings=[]
    for x in obstacles:
        x_move = x[0] - current_pos[0]
        y_move = x[1] - current_pos[1]
        hdg = int(np.degrees(np.arctan2(y_move, x_move)))
        obsHeadings.append(hdg)
    return obsHeadings
def varredura(angles, goalHdg):
    if not angles:
        return None
    closest_angle = min(angles, key=lambda x: min(abs(x - goalHdg), 360 - abs(x - goalHdg)))
    return closest_angle
def angulosPossiveis(obstaclesHdg,windHdg):
    angles=[]
    angMinObs=10
    windHdg=90 #esta vindo de 90 graus
    for i in range(360):
        angles.append(i)
    #print(angles)
    if windHdg>44 and windHdg<316:
        for i in range(45):
            if windHdg in angles:
                angles.remove(windHdg)
            if windHdg+i in angles:
                angles.remove(windHdg+i)
            if windHdg-i in angles:
                angles.remove(windHdg-i)
    else:
        for i in range(45):
            if windHdg in angles:
                angles.remove(windHdg)
            if windHdg + i <= 359:
                if windHdg + i in angles:
                    angles.remove(windHdg + i)
            else:
                if (windHdg + i) % 360 in angles:
                    angles.remove((windHdg + i) % 360)
            if windHdg - i >= 0:
                if windHdg - i in angles:
                    angles.remove(windHdg - i)
            else:
                if 360 + (windHdg - i) in angles:
                    angles.remove(360 + (windHdg - i))
    for x in obstaclesHdg:
        if x > angMinObs-1 and x < 360-angMinObs+1:
            for i in range(angMinObs):
                if x in angles:
                    angles.remove(x)
                if x+i in angles:
                    angles.remove(x+i)
                if x-i in angles:
                    angles.remove(x-i)
        elif x<angMinObs:
            for i in range(angMinObs):
                if x in angles:
                    angles.remove(x)
                if x + i in angles:
                    angles.remove(x + i)
                if x-i>=0:
                    if x - i in angles:
                        angles.remove(x - i)
                else:
                    if 359+x - i in angles:
                        angles.remove(360+x - i)
        else:
            for i in range(angMinObs):
                if x in angles:
                    angles.remove(x)
                if x - i in angles:
                    angles.remove(x - i)
                if x+i <=359:
                    if x + i in angles:
                        angles.remove(x + i)
                else:
                    if (x+i)%360 in angles:
                        angles.remove((x+i)%360)
    print(angles)
    return angles

if __name__ == "__main__":
    truewind  = [0, 6]
    goal      = [100, 50]
    obstacles = [(50,20),(50,40),(50,0)]
    currentPos = [0, 0]
    currentHdg=0
    #goal = rot(goal, np.deg2rad(-45))
    #truewind = rot(truewind, np.deg2rad(-45))
    obstaclesHdg=headingOBS(obstacles, currentPos)
    goalHdg=heading(goal, currentPos)
    #print(goalHdg)
    angles = angulosPossiveis(obstaclesHdg, truewind)
    angFinal=varredura(angles,goalHdg)
    print(angFinal)
    # obstacle1 = rot(obstacle1, np.deg2rad(80))

    nav_angle = np.rad2deg(angBetween2Vecs(goal,[1, 0]))

    alpha = np.rad2deg(angBetween2Vecs(goal, truewind))
    theta = np.rad2deg(angBetween2Vecs(truewind, [1, 0]))
    dang = np.rad2deg(angBetween2Vecs(obstacles[0], [1, 0]))


    if truewind[1] != 0:
        theta *= np.sign(truewind[1])
    else:
        pass

    fig = plt.figure(figsize=[6, 6])
    ax = fig.add_subplot(111, projection='polar')
    # Plotar todos os obstáculos em vermelho
    if obstacles:
        for obs in obstacles:
            dang = np.rad2deg(angBetween2Vecs(obs, [1, 0]))
            if obs[1] != 0:
                dang *= np.sign(obs[1])
            ax.plot(np.deg2rad(dang), 1, marker='o', markersize=15, color='red')

    q = ax.quiver(
        0,  # x-coordinates of the arrow origins
        0,  # y-coordinates of the arrow origins
        np.deg2rad(theta),  # angles of the arrows
        1,  # magnitudes (lengths) of the arrows
        angles='xy',  # Use 'xy' for arrows to point in the correct angle
        scale_units='xy',  # Set scale units to 'xy' to scale arrow length by data units
        scale=1,  # Set scale to 1 to match arrow length to magnitude
        width=0.012,
        color='tab:blue'
    )

    if goal[1] != 0:
        beta = np.sign(goal[1]) * angBetween2Vecs(goal,[1, 0])
    else:
        beta = angBetween2Vecs(goal, [1, 0])

    ax.plot(beta, 1, marker='o', markersize=15, color='orange')

    ax.plot(np.deg2rad(dang), 1, marker='o', markersize=15, color='red')

    #------------------------------------------------------------------------------
    if alpha < 45:
        if (goal[1] != 0):
            nav_angle = (45 - theta) * np.sign(goal[1])
        else:
            nav_angle = (45 - theta)
    else:
        pass

    if (dang < 30) & (int(theta) <= (45 + dang)):
        nav_angle = abs(theta) + 45
        if truewind[1] != 0:
            nav_angle *= np.sign(truewind[1])
        else:
            pass
    #print(dang, int(theta))
    #print(nav_angle, theta + 45, (dang < 30), (int(theta) < (45 + dang)))

    if angFinal is not None:
        ax.quiver(
            0,
            0,
            np.deg2rad(angFinal),
            1,
            angles='xy',
            scale_units='xy',
            scale=1,
            width=0.012,
            color='black'
        )

    # Customize plot
    ax.set_theta_zero_location("N")  # Set 0 degrees to North
    ax.set_theta_direction(-1)  # Clockwise angle direction
    ax.set_xticks(np.arange(0, 2 * np.pi, np.pi / 6))  # Show angle ticks every 30 degrees
    ax.set_xticklabels(['0', '30', '60', '90', '120', '150', '180', '210', '240', '270', '300',
                        '330'])  # Label angle ticks in degrees
    ax.set_ylim(0, 1.2)  # Set radial limits

    plt.show()