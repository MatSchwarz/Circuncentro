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
from funcoes_graficos import *
from gerador_de_problemas import *
from gradiente_circuncentro import *


def cria_tabela(historico_solucao, valores_f, restricoes_ativas, vetores_direcao, lista_alphas, tamanhos_passo):
    max_length = len(historico_solucao)

    # Preenchendo listas menores com np.nan
    valores_f = valores_f + [np.nan] * (max_length - len(valores_f))
    restricoes_ativas = restricoes_ativas + [[np.nan]] * (max_length - len(restricoes_ativas))
    vetores_direcao = vetores_direcao + [[np.nan]] * (max_length - len(vetores_direcao))
    lista_alphas = lista_alphas + [np.nan] * (max_length - len(lista_alphas))
    tamanhos_passo = tamanhos_passo + [np.nan] * (max_length - len(tamanhos_passo))

    # Criando o DataFrame
    df = pd.DataFrame({
        'Iteração': range(max_length),
        'Solução (x)': [list(x) for x in historico_solucao],
        'valores f': valores_f,
        'Restrições Ativas': [list(r) for r in restricoes_ativas],
        'Vetores Direção': [list(v) for v in vetores_direcao],
        'Alphas': lista_alphas,
        'Tamanhos Passo': tamanhos_passo
    })

    # Arredondando valores nas restrições ativas, se aplicável
    df['Restrições Ativas'] = df['Restrições Ativas'].apply(
        lambda x: [np.round(i, 3) if isinstance(i, (int, float)) else i for i in x]
    )

    return df
