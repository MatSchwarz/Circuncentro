import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Polygon
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from scipy import optimize
from funcoes_auxiliares import *

def gradiente_descendente_circuncentrico(x0, c, A,b, tol=1e-6, max_iter=1000):

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
    index_ativos = []


    for i in range(max_iter):
        
        i_ativos = check_active_constraints(x, A, b, tol)
        # Deixa os vetores das restrições unitários
        restricoes = A[i_ativos].astype(float)
        restricoes_ativas.append(restricoes)
        index_ativos.append(i_ativos)

        for i in range(len(restricoes)):
            restricoes[i] = restricoes[i] / np.linalg.norm(restricoes[i])

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
        for i in range(m):
            a_i = A[i]
            prod = np.dot(a_i, direcao)
            
            # Se a direção não viola a restrição, o passo pode ser infinito
            if prod <= 0:
                continue
            
            # Caso contrário, calculamos o máximo passo possível
            alpha = (b[i] - np.dot(a_i, x)) / prod
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
        x = x + alpha * direcao
        tamanho_passo.append(np.linalg.norm(x - x_old))
        historico_solucao.append(x.copy())
        
        # Calcular o valor da função objetivo
        f_val = np.dot(c, x)
        valores_funcao_objetivo.append(f_val)
        


    return historico_solucao, valores_funcao_objetivo, index_ativos, restricoes_ativas, vetores_direcao, lista_alphas, tamanho_passo, grad