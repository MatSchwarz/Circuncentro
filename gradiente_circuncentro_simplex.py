import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Polygon
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.optimize import linprog
from funcoes_auxiliares import *

def gradiente_descendente_circuncentrico_simplex(x0, c, A, b, tol=1e-6, max_iter=1000, passo_minimo=1e-4):
    x = np.array(x0, dtype=float)
    n = len(c)
    m = len(b)
    grad = np.array(c)

    historico_solucao = [x.copy()]
    restricoes_ativas = []
    vetores_direcao = []
    lista_alphas = []
    tamanho_passo = []
    valores_funcao_objetivo = [np.dot(c, x)]
    index_ativos = []

    for i in range(max_iter):
        i_ativos = check_active_constraints(x, A, b, tol)
        restricoes = A[i_ativos].astype(float)
        restricoes_ativas.append(restricoes)
        index_ativos.append(i_ativos)

        for j in range(len(restricoes)):
            restricoes[j] = restricoes[j] / np.linalg.norm(restricoes[j])

        grad_normalizado = grad / np.linalg.norm(grad)
        if len(restricoes) > 0:
            restricoes = np.vstack([restricoes, grad_normalizado])
        else:
            restricoes = np.array([grad_normalizado])
        direcao = circuncentro(-restricoes)

        if np.linalg.norm(direcao) < tol:
            break
        vetores_direcao.append(direcao)

        alphas = []
        for j in range(m):
            a_j = A[j]
            prod = np.dot(a_j, direcao)
            if prod <= 0:
                continue
            alpha = (b[j] - np.dot(a_j, x)) / prod
            alphas.append(alpha)

        if not alphas:
            print("Problema não limitado na direção de descida")
            break

        alpha = min(alphas)
        lista_alphas.append(alpha)

        x_old = x.copy()
        x = x + alpha * direcao
        passo = np.linalg.norm(x - x_old)
        tamanho_passo.append(passo)
        historico_solucao.append(x.copy())

        f_val = np.dot(c, x)
        valores_funcao_objetivo.append(f_val)

        if passo < passo_minimo:
            print(f"Passo menor que {passo_minimo}, acionando simplex a partir de x = {x}, na iteração {i}")
            res = linprog(c, A_ub=A, b_ub=b, bounds=(0, None), method="simplex", x0=x)
            if res.success:
                historico_solucao.append(res.x)
                valores_funcao_objetivo.append(np.dot(c, res.x))
                i_ativos = check_active_constraints(res.x, A, b, tol)
                restricoes = A[i_ativos].astype(float)
                index_ativos.append(i_ativos)
                restricoes_ativas.append(restricoes)
                vetores_direcao.append(np.zeros_like(res.x)) 
                lista_alphas.append(0)  
                tamanho_passo.append(res.nit) 
            break

    return historico_solucao, valores_funcao_objetivo, index_ativos, restricoes_ativas, vetores_direcao, lista_alphas, tamanho_passo, grad