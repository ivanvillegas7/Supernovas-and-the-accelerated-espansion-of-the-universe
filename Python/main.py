# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 15:04:05 2022

@author: Iván Villegas Pérez
"""
#%%Importamos los módulos que se van a utilizar

from typing import List

import numpy as np

import matplotlib.pyplot as plt

import urllib.request

import pandas as pd

import scipy.integrate as integrate

from MyN_python_siempre_contigo import Minimiza

#%%Importamos los datos

M_v: float = -19.3

j: int = 0

with  open("datos1.dat", "r") as infile:

    lines = infile.readlines()

    z1: List[str] = []

    r1: List[float] = []

    r_err1: List[float] = []

    for  line in  lines:

        #Nos saltamos la primera fila del archivo

        if j==0:

            j+=1

        else:

            vals = line.split()

            z1.append(float(vals[1]))

            r1.append(float(vals[8])-M_v)

            r_err1.append((float(vals[8])-M_v)*float(vals[9])/float(vals[8]))

k: int = 0

with  open("datos2.dat", "r") as infile:

    lines = infile.readlines()

    z2: List[str] = []

    r2: List[float] = []

    r_err2: List[float] = []

    for  line in  lines:

        #Nos saltamos la primera fila del archivo

        if k==0:

            k+=1

        else:

            vals = line.split()

            z2.append(float(vals[1]))

            r2.append(float(vals[8])-M_v)

            r_err2.append((float(vals[8])-M_v)*float(vals[9])/float(vals[8]))
            
targetURL = "https://supernova.lbl.gov/Union/figures/SCPUnion2.1_mu_vs_z.txt"

i = 0

z = []

r = []

r_er = []

for line in urllib.request.urlopen(targetURL):

    #Nos saltamos las 5 primeras filas del archivo  

    if i <= 5:

        column_names = line

        i+=1

    else:

        z.append(float(line.split()[1]))

        r.append(float(line.split()[2]))

        r_er.append(float(line.split()[3]))

output_df = pd.DataFrame({column_names[:17]: z, column_names[27:-1]: r,\
                          column_names[27:-1]: r_er})

output_df.to_csv('datos.dat')

#%%Tarea1

plt.figure()


plt.plot(z1, r1, '.', label=r'SCP SN_E I_A', color='purple')

for i in range(len(r1)):

    plt.errorbar(z1[i],r1[i], yerr = r_err1[i], capsize = 3, color='purple')

plt.plot(z2, r2, '.', label=r'CALÁN/TOLOLO SN_E I_A', color='blue')

for i in range(len(r2)):

    plt.errorbar(z2[i],r2[i], yerr = r_err2[i], capsize = 3, color='blue')

plt.xlabel(r'$z$')

plt.ylabel(r'$\mu$ [pc]')

plt.title(r'Distance modulus ($\mu$) as a function of redshift ($z$)')

plt.legend()

plt.grid(True)


plt.savefig('r frente a z - sin modelos.pdf')

#%%Tarea2

H0: float = (67.31+67.81+67.9+67.27+67.51+67.74)*1e-3/6 # km/s Mpc -> m/s pc

c: float = 3e8 #m/s

Vacio: float = (0.685+0.692+0.6935+0.6844+0.6879+0.6911)/6

Materia: float = (0.315+0.308+0.3065+0.3156+0.3121+0.3089)/6

Curvatura: float = (0.0008-0.052-0.005-0.0001-0.04-0.004)/6

Radiacion: float = 1-Vacio-Materia-Curvatura

CDM: List[float] = []

def E_CDM(x: np.array)-> np.array:

    return 1/np.sqrt(Materia*(1+x)**3+Vacio+Radiacion*(1+x)**4+Curvatura*(1+x)**2)

EdS: List[float] = []

def E_EdS(x: np.array)-> np.array:

    return 1/np.sqrt((1+x)**3)

C_C: List[float] = []

def E_C_C(x: np.array)-> np.array:

    return 1/np.sqrt(0.5*(1+x)**3+0.5)

z_o: np.array = np.linspace(0, 1.45)

for i in range(len(z_o)):

    x: np.array = np.array(np.linspace(0, z_o[i]))

    CDM.append(5*np.log10(((c/H0)*(1+z_o[i])*integrate.simpson(E_CDM(x), x)))-5)

    EdS.append(5*np.log10(((c/H0)*(1+z_o[i])*integrate.simpson(E_EdS(x), x)))-5)
    
    C_C.append(5*np.log10(((c/H0)*(1+z_o[i])*integrate.simpson(E_C_C(x), x)))-5)


plt.figure()


plt.plot(z1, r1, '.', label='Supernovas', color='blue')

for i in range(len(r1)):

    plt.errorbar(z1[i],r1[i], yerr = r_err1[i], capsize = 3, color='blue')

plt.plot(z2, r2, '.', color='blue')

for i in range(len(r2)):

    plt.errorbar(z2[i],r2[i], yerr = r_err2[i], capsize = 3, color='blue')

plt.plot(z_o, CDM, color='red', label='$\Lambda$CDM concordance')

plt.plot(z_o, EdS, color='black', label=r'Einstein-de Sitter model')

plt.xlabel(r'$z$')

plt.ylabel(r'$\mu$ [pc]')

plt.title(r'Distance modulus ($\mu$) as a function of redshift ($z$)')

plt.legend()

plt.xlim(0, 0.85)

plt.grid(True)


plt.savefig('r frente a z.pdf')

#%%Objetivos avanzados 1

plt.figure()


plt.plot(z1, r1, '.', label='Supernovas', color='blue')

for i in range(len(r1)):

    plt.errorbar(z1[i],r1[i], yerr = r_err1[i], capsize = 3, color='blue')

plt.plot(z2, r2, '.', color='blue')

for i in range(len(r2)):

    plt.errorbar(z2[i],r2[i], yerr = r_err2[i], capsize = 3, color='blue')

plt.plot(z, r, '.', color='blue')

for i in range(len(r)):

    plt.errorbar(z[i],r[i], yerr = r_er[i], capsize = 3, color='blue')

plt.plot(z_o, CDM, color='red', label='$\Lambda$CDM concordance')

plt.plot(z_o, EdS, color='black', label=r'Einstein-de Sitter model')

plt.xlabel(r'$z$')

plt.ylabel(r'$\mu$ [pc]')

plt.title(r'Distance modulus ($\mu$) as a function of redshift ($z$)')

plt.legend()

plt.grid(True)


plt.savefig('r frente a z extra datos.pdf')

#%%Objetivos avanzados 2

plt.figure()


plt.plot(z1, r1, '.', label='Supernovas', color='blue')

for i in range(len(r1)):

    plt.errorbar(z1[i],r1[i], yerr = r_err1[i], capsize = 3, color='blue')

plt.plot(z2, r2, '.', color='blue')

for i in range(len(r2)):

    plt.errorbar(z2[i],r2[i], yerr = r_err2[i], capsize = 3, color='blue')

plt.plot(z_o, CDM, color='red', label='$\Lambda$CDM concordance')

plt.plot(z_o, EdS, color='black', label=r'Einstein-de Sitter model')

plt.plot(z_o, C_C, color='green', label='$\Omega_\Lambda=0.5$, $\Omega_m=0.5$')

plt.xlabel(r'$z$')

plt.ylabel(r'$\mu$ [pc]')

plt.title(r'Distance modulus ($\mu$) as a function of redshift ($z$)')

plt.legend()

plt.xlim(0, 0.85)

plt.grid(True)


plt.savefig('r frente a z extra modelos.pdf')

#%%Objetivos avanzados 3

# Buscamos valores  Ω_m y Ω_Λ

z_exp = z1 + z2

r_exp = r1 + r2

def E_aj(x: np.array, o_m: float, o_v: float):
    
    return 1/np.sqrt(o_m*(1+x)**3+o_v)

def f(o_m):

    o_v = 1-o_m

    r_aj = 5*np.log10((c/H0)*(1+z_o)*integrate.simpson(E_aj(z_o, o_m, o_v), z_o))-5

    diferencia = 0
    
    for iexp in range(len(z_exp)):
        
        dif = np.inf
        
        for iaj in range(len(z_o)):
            
            difx = z_exp[iexp]-z_o[iaj]
            
            dify = r_exp[iexp]-r_aj[iaj]

            d = difx**2+dify**2 # Sin raíz para minimizar errores de cálculo
            
            if d<dif:
                
                dif = d
                
        diferencia+=dif
        
    return diferencia

Materia_aj = 0

h = 1e-3

epsilon = 1e-8

N = 200

Materia_aj, f_Materia_aj = Minimiza(f, np.array([Materia_aj]), h, epsilon, N)

Vacio_aj = 1 - Materia_aj

omega_k_ajuste = 0

omega_r_ajuste = 0

r_ajuste1 = 5*np.log10((c/H0)*(1+z_o)*integrate.simpson(E_aj(z_o, Materia_aj, Vacio_aj), z_o))-5


plt.figure()


plt.plot(z_exp, r_exp, '.', label='Supernovas', color='blue')

for i in range(len(r1)):

    plt.errorbar(z1[i],r1[i], yerr = r_err1[i], capsize = 3, color='blue')

plt.plot(z2, r2, '.', color='blue')

for i in range(len(r2)):

    plt.errorbar(z2[i],r2[i], yerr = r_err2[i], capsize = 3, color='blue')
    
plt.plot(z_o, CDM, color='red', label='$\Lambda$CDM concordance')

plt.plot(z_o, EdS, color='black', label=r'Einstein-de Sitter model')

plt.plot(z_o, C_C, color='green', label='$\Omega_\Lambda=0.5$, $\Omega_m=0.5$')

plt.plot(z_o, CDM, color='orange', label=f'Ajuste: $\Omega_\Lambda=${Vacio_aj}, $\Omega_m=${Materia_aj}')

plt.xlabel(r'$z$')

plt.ylabel(r'$\mu$ [pc]')

plt.title(r'Distance modulus ($\mu$) as a function of redshift ($z$)')

plt.legend()

plt.xlim(0, 0.85)

plt.grid(True)


plt.savefig('r frente a z ajuste.pdf')
