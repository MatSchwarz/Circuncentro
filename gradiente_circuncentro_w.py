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
from funcoes_auxiliares import *

def gradiente_descendente_circuncentrico2(x0, c, A,b, tol=1e-6, max_iter=1000):

    # Inicializa alguns parâmetros
    x = np.array(x0, dtype=float)
    n = len(c)
    m = len(b)
    grad = np.array(c)

    #Fatores que estou interessado em guardar 
    historico_solucao = [x.copy()]
    restricoes_ativas = []
    vetores_direcao = []
    lista_alphas = []
    tamanho_passo = []
    valores_funcao_objetivo = [np.dot(c, x)]


    for i in range(max_iter):
        
        i_ativos = check_active_constraints(x, A, b, tol)
        # Deixa os vetores das restrições unitários
        restricoes = A[i_ativos].astype(float)
        restricoes_ativas.append(restricoes)
    
        for j in range(len(restricoes)):
            restricoes[j] = restricoes[j] / np.linalg.norm(restricoes[j])

        grad_normalizado = grad / np.linalg.norm(grad)
        if len(restricoes) > 0:
            restricoes = np.vstack([restricoes, grad_normalizado])
        else:
            restricoes = np.array([grad_normalizado])
        direcao = circuncentro(-restricoes)

        # Verificar convergência
        if np.linalg.norm(direcao) < tol:
            break
        vetores_direcao.append(direcao)

        alphas = []

        # Para restrições Ax <= b
        for k in range(m):
            a_k = A[k]
            prod = np.dot(a_k, direcao)
            
            # Se a direção não viola a restrição, o passo pode ser infinito
            if prod <= 0:
                continue
            
            # Caso contrário, calculamos o máximo passo possível
            alpha = (b[k] - np.dot(a_k, x)) / prod
            alphas.append(alpha)


        # Se não houver restrições ativas na direção de descida
        if not alphas:
            print("Problema não limitado na direção de descida")
            break

        # O tamanho do passo é determinado pela restrição mais próxima
        alpha = min(alphas)
        lista_alphas.append(alpha)
        
        # Atualizar a solução
        x_old = x.copy()
        if i%2 != 0:
            p = 1
        else:
            p = 0.9
        x = x + alpha * direcao * p
        tamanho_passo.append(np.linalg.norm(x - x_old))
        historico_solucao.append(x.copy())
        
        # Calcular o valor da função objetivo
        f_val = np.dot(c, x)
        valores_funcao_objetivo.append(f_val)
        

        
    return historico_solucao, valores_funcao_objetivo, restricoes_ativas, vetores_direcao, lista_alphas,tamanho_passo, grad