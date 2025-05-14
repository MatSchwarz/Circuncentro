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
from funcoes_tabelas import *
from gradiente_circuncentro_w import *
from scipy import optimize

def dist(a,b,h):
     return np.abs(np.dot(a,h) - b)/np.linalg.norm(a)

def proj(a,b,h):
    return h - (np.dot(a,h) - b) * a / np.linalg.norm(a)**2

def like(a,b,at,bt,L,S):

     if np.linalg.norm(a/np.linalg.norm(a) - at/np.linalg.norm(at)) < L:
          if np.linalg.norm(b/np.linalg.norm(a) - bt/np.linalg.norm(at)) < S:
               return False
     return True

def add_support(A,b,alpha):
    A0 = A.copy()
    b0 = b.copy()
    n = A.shape[1]
    A1 = np.zeros((n,n))
    A2 = np.zeros((n,n))
    A3 = np.zeros((1,n))
    b1 = np.zeros(n)
    b2 = np.zeros(n)
    b3 = np.zeros(1)
    for i in range(n):
        A1[i,i] = 1
        A2[i,i] = -1
        A3[0,i] = 1
        b1[i] = alpha
        b2[i] = 0
        b3[0] = (n - 1) * alpha + alpha / 2
    A0 = np.vstack((A0,A1))
    A0 = np.vstack((A0,A2))
    A0 = np.vstack((A0,A3))
    b0 = np.hstack((b0,b1))
    b0 = np.hstack((b0,b2))
    b0 = np.hstack((b0,b3))

    return A0, b0 


def gera_pl(n,d,alpha,theta,pho,Smin,Lmax,amax,bmax):
    A = np.zeros((d,n))
    b = np.zeros(d)
    c = np.zeros(n)
    A, b = add_support(A,b,alpha)
    h = np.full(n, alpha/2)
    for j in range(1,n+1):
        c[j-1] = - theta * j
    if d == 0:
        return A, b, c
    i = 0
    while i < d:
        A[i,:] = np.random.uniform(0,amax,n) * np.random.choice([1, -1], size=n)
        b[i] = np.random.uniform(0,bmax,1) * np.random.choice([1, -1], size=1)
        if np.dot(A[i,:],h) > b[i]:
            A[i,:] = -A[i,:]
            b[i] = -b[i]
        if dist(A[i,:],b[i],h) < pho or dist(A[i,:],b[i],h) > theta:
            continue
        if np.dot(c,proj(A[i,:],b[i],h)) > np.dot(c,h):
            continue
        if i > 0:
            for k in range(i):
                if like(A[i,:],b[i],A[k,:],b[k],Lmax,Smin) == False:
                    continue
        i += 1
    return A, b, c, h