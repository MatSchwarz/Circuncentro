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

def gerar_problema_otimizacao_linear_factivel_limitado(m, n):
    while True:
        c = np.random.randint(-10, 10, size=n)
        A_aleatorio = np.random.randint(-10, 10, size=(m, n))
        b_aleatorio = np.random.randint(1, 20, size=m)

        # Adicionando restrições de sinal -x <= 0
        A_sinal = -np.eye(n)
        b_sinal = np.zeros(n)

        # Adicionando restrições para limitar o conjunto viável
        A_limite = np.eye(n)
        b_limite = np.random.randint(5, 15, size=n)

        # Combinando as restrições
        A = np.vstack([A_aleatorio, A_sinal, A_limite])
        b = np.hstack([b_aleatorio, b_sinal, b_limite])

        # Testar se o problema é factível e limitado
        result = linprog(c, A_ub=A, b_ub=b, bounds=(0, None), method="highs")
        if result.success:
            return c, A, b
