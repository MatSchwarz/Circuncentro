import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import pandas as pd
from matplotlib.patches import Polygon
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from numpy.linalg import solve
from scipy import optimize
from scipy.optimize import linprog
from funcoes_auxiliares import *

def grafico1(xfinal, A, b):
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = list(mcolors.TABLEAU_COLORS.values())
    xfinal_array = np.array(xfinal)
    x_min, y_min = np.min(xfinal_array, axis=0) - 2
    x_max, y_max = np.max(xfinal_array, axis=0) + 2
    x_vals = np.linspace(x_min, x_max, 500)
    for i in range(len(A)):
        if A[i, 1] != 0:
            y_vals = (b[i] - A[i, 0] * x_vals) / A[i, 1]
            ax.plot(x_vals, y_vals, color=colors[i % len(colors)], 
                    linewidth=1.3, alpha=0.7, 
                    label=f'Restrição {i+1}: {A[i, 0]}x + {A[i, 1]}y ≤ {b[i]}')
            
        else:
            x_const = b[i] / A[i, 0]
            ax.axvline(x=x_const, color=colors[i % len(colors)], 
                    linewidth=1.3, alpha=0.7,
                    label=f'Restrição {i+1}: {A[i, 0]}x ≤ {b[i]}')



    ax.plot([xfinal_array[0, 0], xfinal_array[1, 0]], 
        [xfinal_array[0, 1], xfinal_array[1, 1]], 
        color='darkred', linewidth=1.5, linestyle='-',)
    ax.scatter(xfinal_array[0, 0], xfinal_array[0, 1], 
        color='black', marker='x', linewidth=2, label='Ponto Inicial',alpha=1)

    ax.plot(xfinal_array[1:, 0], xfinal_array[1:, 1], 'o-', 
        color='darkred', linewidth=1.5, markersize=3, 
        markerfacecolor='white', markeredgewidth=1, label='Iterações de x_final')

    # Parte do sombreado no gráfico
    # Criar uma grade para verificar a região factível
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    feasible_region = np.zeros_like(xx, dtype=bool)

    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            feasible_region[i, j] = is_feasible(xx[i, j], yy[i, j], A, b)

    ax.contourf(xx, yy, feasible_region, levels=[0.5, 1.5], colors=['gray'], alpha=0.2)

    # Configurações do gráfico
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_axisbelow(True)  # Colocar grid atrás dos dados

    # Título e rótulos com estilo melhorado
    ax.set_title('Restrições e Região Factível', fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('x', fontsize=14, fontweight='bold')
    ax.set_ylabel('y', fontsize=14, fontweight='bold')

    # Adicionar legenda para a região factível
    feasible_patch = mpatches.Patch(color='gray', alpha=0.2, label='Região Factível')

    # Criar handles e labels personalizados
    handles, labels = ax.get_legend_handles_labels()
    handles.append(feasible_patch)

    # Legenda com estilo melhorado
    legend = ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=12, 
                       frameon=True, facecolor='white', framealpha=0.9)
    legend.set_title('Elementos do Gráfico', prop={'size': 14, 'weight': 'bold'})

    
    plt.show()



    def grafico2(xfinal, A, b):
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(10, 8))

        colors = list(mcolors.TABLEAU_COLORS.values())

        x_vals = np.linspace(-1, 10, 500)
        for i in range(len(A)):
            if A[i, 1] != 0:
                y_vals = (b[i] - A[i, 0] * x_vals) / A[i, 1]
                ax.plot(x_vals, y_vals, color='blue', 
                        linewidth=1.3, alpha=0.7)
                
            else:
                x_const = b[i] / A[i, 0]
                ax.axvline(x=x_const, color='blue', 
                        linewidth=1.3, alpha=0.7)

        ax.plot([], [], color='blue', linewidth=1.3, alpha=0.7, label='Restrições')

        xfinal_array = np.array(xfinal)

        ax.plot([xfinal_array[0, 0], xfinal_array[1, 0]], 
            [xfinal_array[0, 1], xfinal_array[1, 1]], 
            color='darkred', linewidth=1.5, linestyle='-',)
        ax.scatter(xfinal_array[0, 0], xfinal_array[0, 1], 
            color='black', marker='x', linewidth=2, label='Ponto Inicial',alpha=1)

        ax.plot(xfinal_array[1:, 0], xfinal_array[1:, 1], 'o-', 
            color='darkred', linewidth=1.5, markersize=3, 
            markerfacecolor='white', markeredgewidth=1, label='Iterações de x_final')

        # Parte do sombreado no gráfico
        # Criar uma grade para verificar a região factível
        xx, yy = np.meshgrid(np.linspace(-1, 10, 200), np.linspace(-1, 10, 200))
        feasible_region = np.zeros_like(xx, dtype=bool)

        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                feasible_region[i, j] = is_feasible(xx[i, j], yy[i, j], A, b)

        ax.contourf(xx, yy, feasible_region, levels=[0.5, 1.5], colors=['gray'], alpha=0.4)

        # Configurações do gráfico
        ax.set_xlim(-1, 10)
        ax.set_ylim(-1, 10)
        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_axisbelow(True)  # Colocar grid atrás dos dados

        # Título e rótulos com estilo melhorado
        ax.set_title('Restrições e Região Factível', fontsize=18, fontweight='bold', pad=20)
        ax.set_xlabel('x', fontsize=14, fontweight='bold')
        ax.set_ylabel('y', fontsize=14, fontweight='bold')

        # Adicionar legenda para a região factível
        feasible_patch = mpatches.Patch(color='gray', alpha=0.2, label='Região Factível')

        # Criar handles e labels personalizados
        handles, labels = ax.get_legend_handles_labels()
        handles.append(feasible_patch)

        # Legenda com estilo melhorado
        legend = ax.legend(handles=handles, loc='upper right', fontsize=12, frameon=True, facecolor='white', framealpha=0.9)
        legend.set_title('Elementos do Gráfico', prop={'size': 14, 'weight': 'bold'})

        plt.show()