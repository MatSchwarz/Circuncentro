import numpy as np
from scipy.optimize import linprog
from funcoes_auxiliares import *
from funcoes_graficos import *
from gerador_de_problemas import *
from gradiente_circuncentro import *
from funcoes_tabelas import *
from gradiente_circuncentro_w import *
from gerador_pl_Sokolinsky import *
from scipy.linalg import null_space

def gradiente_descendente_circuncentrico_modificado(x0, c, A, b, tol=1e-6, max_iter=1000):

    x = np.array(x0, dtype=float)
    n = len(c)
    m = len(b)
    grad = np.array(c)
    historico_solucao = [x.copy()]
    restricoes_ativas_hist = []
    vetores_direcao = []
    lista_alphas = []
    tamanho_passo = []
    valores_funcao_objetivo = [np.dot(c, x)]
    index_ativos_hist = [] 

    for i in range(max_iter):
        
        i_ativos = check_active_constraints(x, A, b, tol)
        index_ativos_hist.append(set(i_ativos)) 

        tomar_passo_simplex = False
        if i >= 2:
            if index_ativos_hist[-1] == index_ativos_hist[-3]:
                print(f"Iteração {i}: Ciclagem detectada! Acionando passo do simplex.")
                tomar_passo_simplex = True

        A_ativos_raw = A[i_ativos].astype(float)
        restricoes_ativas_hist.append(A_ativos_raw.copy())

        if tomar_passo_simplex:

            direcao = calcular_direcao_aresta(A_ativos_raw, grad)
        else:

            restricoes = A_ativos_raw.copy()
            for j in range(len(restricoes)):
                norma_r = np.linalg.norm(restricoes[j])
                if norma_r > tol:
                    restricoes[j] = restricoes[j] / norma_r

            grad_normalizado = grad / np.linalg.norm(grad)
            if len(restricoes) > 0:
                restricoes = np.vstack([restricoes, grad_normalizado])
            else:
                restricoes = np.array([grad_normalizado])
            
            direcao = circuncentro(-restricoes)

        if np.linalg.norm(direcao) < tol:
            print(f"Iteração {i}: Convergência alcançada. Norma da direção < {tol}")
            break
        vetores_direcao.append(direcao)

        alphas = []
        for k in range(m):
            a_k = A[k]
            prod = np.dot(a_k, direcao)
            if prod <= 1e-8:
                continue
            alpha = (b[k] - np.dot(a_k, x)) / prod
            alphas.append(alpha)

        if not alphas:
            print("Problema não limitado na direção de descida")
            break

        alpha = min(alphas)
        if alpha < 0 and abs(alpha) < tol: 
            alpha = 0
            
        lista_alphas.append(alpha)
        
        x_old = x.copy()
        

        p = 1.0 
        x = x + alpha * direcao * p

        tamanho_passo.append(np.linalg.norm(x - x_old))
        historico_solucao.append(x.copy())
        
        f_val = np.dot(c, x)
        valores_funcao_objetivo.append(f_val)


    return historico_solucao, valores_funcao_objetivo, index_ativos_hist, restricoes_ativas_hist, vetores_direcao, lista_alphas, tamanho_passo, grad