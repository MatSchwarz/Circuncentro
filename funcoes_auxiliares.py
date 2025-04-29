import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import pandas as pd
from matplotlib.patches import Polygon
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from numpy.linalg import solve
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.optimize import linprog

def circuncentro(x):

    x = np.array(x)
    x = x / np.linalg.norm(x, axis=1)[:, np.newaxis]
    p = len(x) 
    n = x.shape[1]

    if p <= 1:
        return x[0]

    if p == 2:
        return (x[0] + x[1]) / 2
    
    A = np.zeros((p-1, p-1))
    b = np.zeros(p-1)

    for i in range(p-1):
        for j in range(p-1):

            A[i, j] = np.dot(x[j+1] - x[0], x[i+1] - x[0])
        

        b[i] = 0.5 * np.linalg.norm(x[i+1] - x[0])**2
    

    alphas = solve(A, b)
    
    alpha_1 = 1.0 - np.sum(alphas)
    all_alphas = np.insert(alphas, 0, alpha_1)
    
    circ = np.zeros(n)
    for i in range(p):
        if i == 0:
            circ += all_alphas[i] * x[i]
        else:
            circ += all_alphas[i] * x[i]
    
    return circ



def verifica(x, circ):
    distances = np.linalg.norm(x - circ, axis=1)
    return distances



def check_active_constraints(x, A, b, tol=1e-6):
    active_constraints = []
    for i, restricao in enumerate(A):
        # Verifica se |Ax - b| ≤ tol
        if abs(np.dot(restricao, x) - b[i]) <= 1e-10:
            active_constraints.append(i)
    return active_constraints