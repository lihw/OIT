
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

def computeTransmittanceAtDepthFrom4PowerMoments(b_0, b_even, b_odd, depth, bias, overestimation, bias_vector):
    float4 b = float4(b_odd.x, b_even.x, b_odd.y, b_even.y);

        // Bias input data to avoid artifacts
        b = lerp(b, bias_vector, bias);
        float3 z;
        z[0] = depth;

        // Compute a Cholesky factorization of the Hankel matrix B storing only non-
        // trivial entries or related products
        float L21D11=mad(-b[0],b[1],b[2]);
        float D11=mad(-b[0],b[0], b[1]);
        float InvD11=1.0f/D11;
        float L21=L21D11*InvD11;
        float SquaredDepthVariance=mad(-b[1],b[1], b[3]);
        float D22=mad(-L21D11,L21,SquaredDepthVariance);

        // Obtain a scaled inverse image of bz=(1,z[0],z[0]*z[0])^T
        float3 c=float3(1.0f,z[0],z[0]*z[0]);
        // Forward substitution to solve L*c1=bz
        c[1]-=b.x;
        c[2]-=b.y+L21*c[1];
        // Scaling to solve D*c2=c1
        c[1]*=InvD11;
        c[2]/=D22;
        // Backward substitution to solve L^T*c3=c2
        c[1]-=L21*c[2];
        c[0]-=dot(c.yz,b.xy);
        // Solve the quadratic equation c[0]+c[1]*z+c[2]*z^2 to obtain solutions 
        // z[1] and z[2]
        float InvC2=1.0f/c[2];
        float p=c[1]*InvC2;
        float q=c[0]*InvC2;
        float D=(p*p*0.25f)-q;
        float r=sqrt(D);
        z[1]=-p*0.5f-r;
        z[2]=-p*0.5f+r;
        // Compute the absorbance by summing the appropriate weights
        float3 polynomial;
        float3 weight_factor = float3(overestimation, (z[1] < z[0])?1.0f:0.0f, (z[2] < z[0])?1.0f:0.0f);
        float f0=weight_factor[0];
        float f1=weight_factor[1];
        float f2=weight_factor[2];
        float f01=(f1-f0)/(z[1]-z[0]);
        float f12=(f2-f1)/(z[2]-z[1]);
        float f012=(f12-f01)/(z[2]-z[0]);
        polynomial[0]=f012;
        polynomial[1]=polynomial[0];
        polynomial[0]=f01-polynomial[0]*z[1];
        polynomial[2]=polynomial[1];
        polynomial[1]=polynomial[0]-polynomial[1]*z[0];
        polynomial[0]=f0-polynomial[0]*z[0];
        float absorbance = polynomial[0] + dot(b.xy, polynomial.yz);;
        // Turn the normalized absorbance into transmittance
        return saturate(exp(-b_0 * absorbance));
    
def MomentOIT(colors):
    final = None

    # Generate moments
    #

    moments = np.zeros((3))
    moments0 = 0

    for i in range(len(colors)):
        if i == 0:
            background = colors[i][0:3]
            background_alpha = colors[i][3]
        else:
            depth = 500 * (1.0 -  i / len(colors))
            absorbance = -log(colors[i][3]);

            b_0 = absorbance
            b = np.array((depth, depth ** 2, depth ** 3, depth ** 4)) * absorbance;

            moments0 += b_0
            moments += b

    bias_vector = np.array((0, 0.375, 0.375))

    # Resolve
    for i, b in enumerate(moments):
        b_odd = b[0:2]
        b_even = b[2:4]

        b_even /= b_0
        b_odd /= b_0

        t = computeTransmittanceAtDepthFrom4PowerMoments(b_0, b_even, b_odd, depth, bias, overestimation, bias_vector):


    return final
