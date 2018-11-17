
import numpy as np

def GroundTruth(colors):
    final = np.zeros(3)

    for color in colors:
        final = color[3] * color[0:3] + (1 - color[3]) * final

    return final

def Meshkin(colors):

    background = None
    background_alpha = None
    foreground = np.zeros(3)
    foreground_alpha = np.zeros(3)

    for i in range(len(colors)):
        if i == 0:
            background = colors[i][0:3]
            background_alpha = colors[i][3]
        else:
            foreground += colors[i][0:3] * colors[i][3]
            foreground_alpha += colors[i][3]

    final = foreground + background * (1 - foreground_alpha)

    return final

def Bavoil(colors):
    background = None
    background_alpha = None
    foreground = np.zeros(3)
    foreground_alpha = np.zeros(3)

    for i in range(len(colors)):
        if i == 0:
            background = colors[i][0:3]
            background_alpha = colors[i][3]
        else:
            foreground += colors[i][0:3]
            foreground_alpha += colors[i][3]

    objects = len(colors) - 1

    accum_alpha = (1 - foreground_alpha / objects) ** objects

    final = foreground / foreground_alpha * (1 - accum_alpha) + background * accum_alpha

    return final
    
def Mcguire(colors):
    background = None
    background_alpha = None
    foreground = np.zeros(3)
    foreground_alpha_mult = np.ones(3)
    foreground_alpha_add = np.ones(3)

    for i in range(len(colors)):
        if i == 0:
            background = colors[i][0:3]
            background_alpha = colors[i][3]
        else:
            foreground += colors[i][0:3]
            foreground_alpha_add += colors[i][3]
            foreground_alpha_mult *= (1 - colors[i][3])

    objects = len(colors) - 1

    final = foreground / foreground_alpha_add * (1 - foreground_alpha_mult) + background * foreground_alpha_mult

    return final

def McguireDepth(colors):
    background = None
    background_alpha = None
    foreground = np.zeros(3)
    foreground_alpha_mult = np.ones(3)
    foreground_alpha_add = np.ones(3)

    for i in range(len(colors)):
        if i == 0:
            background = colors[i][0:3]
            background_alpha = colors[i][3]
        else:
            depth = 500 * (1.0 -  i / len(colors))
            z_weight = colors[i][3] * max(0.01, min(3000.0, 0.03 / (0.00001 + (depth / 200) ** 6)))

            foreground += colors[i][0:3] * z_weight
            foreground_alpha_add += colors[i][3] * z_weight
            foreground_alpha_mult *= (1 - colors[i][3])

    objects = len(colors) - 1

    final = foreground / foreground_alpha_add * (1 - foreground_alpha_mult) + background * foreground_alpha_mult

    return final
    
def MomentOIT(colors):
    final = None

    return final
