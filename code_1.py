import sympy as sp
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import linalg
import scipy as scpy


from sympy.physics.vector import init_vprinting
Axes3D = Axes3D

init_vprinting(use_latex='mathjax', pretty_print=False)


a, alpha, d, theta, theta1, theta2, theta3, l1, a2, a3 = sp.symbols(
    'a alpha d theta theta1 theta2 theta3 l1 a2 a3')
# Helper functions


def scos(x): return sp.cos(x).evalf()
def ssin(x): return sp.sin(x).evalf()



# Cross product function


def cross(A, B):
    return [A[1]*B[2] - A[2]*B[1], A[2]*B[0] - A[0]*B[2], A[0]*B[1] - A[1]*B[0]]

# DH Transformation


def dh_trans(q):

    # Constant D-H parameters
    l1 = 1.0
    a2 = 1.0
    a3 = 1.0


    theta1 = q[0]
    theta2 = q[1]
    theta3 = q[2]

    rot = sp.Matrix([[scos(theta), -ssin(theta)*scos(alpha), ssin(theta)*ssin(alpha)],
                     [ssin(theta), scos(theta)*scos(alpha), -
                      scos(theta)*ssin(alpha)],
                     [0, ssin(alpha), scos(alpha)]])
    trans = sp.Matrix([a*scos(theta), a*ssin(theta), d])
    last_row = sp.Matrix([[0, 0, 0, 1]])
    t = sp.Matrix.vstack(sp.Matrix.hstack(rot, trans), last_row)

    t01 = t.subs({a: 0, alpha: sp.pi/2, d: l1, theta: theta1})
    t12 = t.subs({a: a2, alpha: 0, d: 0, theta: theta2})
    t23 = t.subs({a: a3, alpha: 0, d: 0, theta: theta3})


    t06 = t01 * t12 * t23
    return t06

# DH for Jacobian

def dh_generic_jacobian():
    # Constant D-H parameters
    l1 = 1.0
    a2 = 1.0
    a3 = 1.0


    rot = sp.Matrix([[scos(theta), -ssin(theta)*scos(alpha), ssin(theta)*ssin(alpha)],
                     [ssin(theta), scos(theta)*scos(alpha), -
                      scos(theta)*ssin(alpha)],
                     [0, ssin(alpha), scos(alpha)]])
    trans = sp.Matrix([a*scos(theta), a*ssin(theta), d])
    last_row = sp.Matrix([[0, 0, 0, 1]])
    t = sp.Matrix.vstack(sp.Matrix.hstack(rot, trans), last_row)

    t01 = t.subs({a: 0, alpha: sp.pi/2, d: l1, theta: theta1})
    t12 = t.subs({a: a2, alpha: 0, d: 0, theta: theta2})
    t23 = t.subs({a: a3, alpha: 0, d: 0, theta: theta3})


    T = [t01, t01*t12, t01*t12*t23]
    return T

def dh_for_jacobian(q):
    # Constant D-H parameters
    l1 = 1.0
    a2 = 1.0
    a3 = 1.0

    theta1 = q[0]
    theta2 = q[1]
    theta3 = q[2]

    rot = sp.Matrix([[scos(theta), -ssin(theta)*scos(alpha), ssin(theta)*ssin(alpha)],
                     [ssin(theta), scos(theta)*scos(alpha), -
                      scos(theta)*ssin(alpha)],
                     [0, ssin(alpha), scos(alpha)]])
    trans = sp.Matrix([a*scos(theta), a*ssin(theta), d])
    last_row = sp.Matrix([[0, 0, 0, 1]])
    t = sp.Matrix.vstack(sp.Matrix.hstack(rot, trans), last_row)

    t01 = t.subs({a: 0, alpha: sp.pi/2, d: l1, theta: theta1})
    t12 = t.subs({a: a2, alpha: 0, d: 0, theta: theta2})
    t23 = t.subs({a: a3, alpha: 0, d: 0, theta: theta3})


    T = [t01, t01*t12, t01*t12*t23]
    return T

# Jacobian calculation


def jacobian(T):

    # Z vectors
    z = [sp.Matrix([0, 0, 1])]
    for i in T:
        z.append((i[:3, 2]))

    # Origins
    o = [sp.Matrix([0, 0, 0])]
    for i in T:
        o.append((i[:3, 3]))

    # Build the Jacobian matrix
    J = sp.zeros(6, 3)

    # The first three rows of the Jacobian are the cross product of z vectors and difference of end-effector and joint origins
    for i in range(3):
        J[0, i] = sp.Matrix(
            cross(z[i], [o[-1][0] - o[i][0], o[-1][1] - o[i][1], o[-1][2] - o[i][2]]))

    # The last three rows of the Jacobian are simply the z vectors for rotational joints
    for i in range(3):
        J[3:6, i] = z[i]
        # sp.pprint(J)
    return J


def generic_gravity_matrix():
  l1 = 1.0
  a2 = 1.0
  a3 = 1.0

  # Mass of the LINKS
  m1 = ((0.09*0.09-0.08*0.08)) *7600
  m2 = ((0.09*0.09-0.08*0.08)) *7600
  m3 = ((0.09*0.09-0.08*0.08)) *7600

  # Height of Centre of Mass of all the LINKS
  h1 = 0.2+l1/2
  h2 = 0.2+l1 + (a2*ssin(theta2))/2
  h3 = 0.2+l1 + (a2*ssin(theta2)) + (a3*ssin(theta2+theta3))/2

  #Total Potential Energy
  p = 9.8*((m1*h1)+(m2*h2)+(m3*h3))

  g = sp.Matrix([0, 0, 0])

  g[0] = sp.diff(p,theta1)
  g[1] = sp.diff(p,theta2)
  g[2] = sp.diff(p,theta3)

  return g

def gravity_matrix(q):
    g = generic_gravity_matrix()
    g[0] = g[0].subs(theta1, q[0]).subs(theta2, q[1]).subs(theta3, q[2]).evalf()
    g[1] = g[1].subs(theta1, q[0]).subs(theta2, q[1]).subs(theta3, q[2]).evalf()
    g[2] = g[2].subs(theta1, q[0]).subs(theta2, q[1]).subs(theta3, q[2]).evalf()

    return g

def inertia_matrix():
    D = 0.09
    d = 0.08
    moi = (D**4-d**4)/12
    I2 = sp.Matrix([[0, 0, 0],
     [0,moi,0],
      [0,0,moi]])
    
    I3 = sp.Matrix([[0, 0, 0],
     [0,moi,0],
      [0,0,moi]])
    I = [I2, I3]

    m = ((0.09**2-0.08**2)) *7600
    m = sp.Matrix([ [m], [m], [m]])

    transformations = dh_generic_jacobian()
    r2 = transformations[1]
    R2=r2[0:3,0:3]

    r3 = transformations[2]
    R3=r3[0:3,0:3]

    R = [R2,R3]

    J_w2 = sp.Matrix([[0, 0, 0],
                    [0, 1 ,0],
                     [1, 0 ,0]]) 
    
    J_w3 = sp.Matrix([[ 0, 0 ,0],
                     [ 0, 1, 1 ],
                     [1, 0 ,0]])
    
    J_v2 = sp.Matrix([[0, -ssin(theta2), 0],
                     [scos(theta2)**2, -ssin(theta2)**2, 0],
                      [scos(theta1), 0, 0]])
    J_v3 = sp.Matrix([[0, -ssin(theta2), -ssin(theta3)/2],
                      [scos(theta1)*scos(theta2)+scos(theta1)*scos(theta2+theta3), 
                                                        -ssin(theta1)*ssin(theta2)-ssin(theta1)*ssin(theta2+theta3),
                                                        ssin(theta2)*ssin(theta2+theta3)],
                     [0, scos(theta2), scos(theta3)/2]])
    
    J_w = [J_w2, J_w3]
    J_v = [J_v2, J_v3]
    D = np.zeros((3,3))

    for i in range(2):
      D = D + m[i]*J_v[i].transpose()*J_v[i] + J_w[i].transpose()*R[i]*I[i]*R[i].transpose()*J_w[i]
    return D
def sub_inertia_matrix(q_val):
  D = inertia_matrix()
  D_substituted = D.subs({theta1:q_val[0], theta2:q_val[1], theta3:q_val[2]})    
  return D_substituted

def coriolis_matrix(q_val,q_dot):
    q = sp.Matrix([[theta1],[theta2], [theta3]])
    D = inertia_matrix()
    C = sp.zeros(3, 3)  # 3x3 matrix for a 3 DOF manipulator
    n = len(q)

    # Compute Coriolis matrix
    for i in range(n):
        for j in range(n):
            for k in range(n):
                # The term C_ijk
                C_ijk = 0.5 * (sp.diff(D[i, j], q[k]) + sp.diff(D[i, k], q[j]) - sp.diff(D[j, k], q[i]))
                C[i, j] += C_ijk * q_dot[k]
   
    C_substituted = C.subs({theta1:q_val[0], theta2:q_val[1], theta3:q_val[2]})
    return C_substituted
def print_joint_torques(q):
    f = sp.Matrix([-10, 0, 0, 0, 0, 0])
    dt = 1
    total_time = np.arange(0, 20, dt)

    x_points = list()
    y_points = list()
    z_points = list()

    t_joint1 = list()
    t_joint2 = list()
    t_joint3 = list()

    tau = list()
    for i in total_time:
        x_dot = 0.0
        y_dot = 0.0
        z_dot = -0.707/20
        epsilon = sp.Matrix([x_dot, y_dot, z_dot, 0.0, 0.0, 0.0])

        T = dh_for_jacobian(q)
        J = jacobian(T)
        J_trans = J.transpose()

        j_inv = np.linalg.pinv(np.array(J, dtype=float))
        q_dot = j_inv * epsilon
        q = q + q_dot * dt
        q[0]= 0
        q_ddot = q_dot/dt

        D = sub_inertia_matrix(q)
        C = coriolis_matrix(q, q_dot)
        # sp.pprint(C)

        tau = D * q_ddot + C * q_dot + gravity_matrix(q) - (J_trans * f)
        # tau = gravity_matrix(q) - (J_trans * f)
        T_end_effector = dh_trans(q)

        t_joint1.append(tau[0])
        t_joint2.append(tau[1])
        t_joint3.append(tau[2])

        x_points.append(0)
        y_points.append(T_end_effector[1, 3])
        z_points.append(T_end_effector[2, 3])

    plt.rcParams['figure.figsize'] = [20, 10]
    fig, axs = plt.subplots(3,1)
    axs[0].plot(total_time, t_joint1)
    axs[0].set_title("JOINT 1 TORQUE")


    axs[1].plot(total_time, t_joint2)
    axs[1].set_title("JOINT 2 TORQUE")

    axs[2].plot(total_time, t_joint3)
    axs[2].set_title("JOINT 3 TORQUE")

    fig.tight_layout(pad=3.0)

    for ax in axs.flat:
      ax.set(xlabel='TIME', ylabel='TORQUE')

    plt.show()


    fig = plt.figure()
    plt.plot(x_points, z_points)
    plt.title("y-z plane")
    plt.xlabel("Y")
    plt.ylabel("Z")
    plt.show()

def print_analytical_jacobian(q_A):
  T = dh_for_jacobian(q_A)
  print("The Analytical Jacobian for point A:")
  sp.pprint(jacobian(T))
# Jabobian for point A
q_A = sp.Matrix([0, 0, 0])
print_analytical_jacobian(q_A)
# Home Position Of End Effector
q_initial = sp.Matrix([0, sp.pi/4, -sp.pi/4])

#Plots JOINT TORQUES and PRINTS THE CIRCLE TRACED
print_joint_torques(q_initial)