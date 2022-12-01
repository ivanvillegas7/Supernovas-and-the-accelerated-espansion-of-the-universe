# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 15:04:05 2022

@author: Iván
"""
#%%Importamos los módulos que se van a utilizar

from typing import List

import numpy as np

import matplotlib.pyplot as plt

import urllib.request

import pandas as pd

import scipy.integrate as integrate

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


plt.plot(z1, r1, '.', label='Supernovas', color='blue')

for i in range(len(r1)):

    plt.errorbar(z1[i],r1[i], yerr = r_err1[i], capsize = 3, color='blue')

plt.plot(z1, r1, '.', color='blue')

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

Radiación: float = 1-Vacio-Materia-Curvatura

CDM: List[float] = []

def E_CDM(x: np.array)-> np.array:

    return 1/np.sqrt(Materia*(1+x)**3+Vacio)

EdS: List[float] = []

def E_EdS(x: np.array)-> np.array:

    return 1/np.sqrt((1+x)**3)

z_o: np.array = np.linspace(0, 1.45)

for i in range(len(z_o)):

    x: np.array = np.array(np.linspace(0, z_o[i]))

    CDM.append(5*np.log10(((c/H0)*(1+z_o[i])*integrate.simpson(E_CDM(x), x)))-5)

    EdS.append(5*np.log10(((c/H0)*(1+z_o[i])*integrate.simpson(E_EdS(x), x)))-5)


plt.figure()


plt.plot(z1, r1, '.', label='Supernovas', color='blue')

for i in range(len(r1)):

    plt.errorbar(z1[i],r1[i], yerr = r_err1[i], capsize = 3, color='blue')

plt.plot(z1, r1, '.', color='blue')

for i in range(len(r2)):

    plt.errorbar(z2[i],r2[i], yerr = r_err2[i], capsize = 3, color='blue')

plt.plot(z_o, CDM, color='red', label='$\Lambda$CDM concordance')

plt.plot(z_o, EdS, color='purple', label=r'Einstein-de Sitter model')

plt.xlabel(r'$z$')

plt.ylabel(r'$\mu$ [pc]')

plt.title(r'Distance modulus ($\mu$) as a function of redshift ($z$)')

plt.legend()

plt.xlim(0, 0.85)

plt.grid(True)


plt.savefig('r frente a z.pdf')

#%%Objetivos avanzados

plt.figure()


plt.plot(z1, r1, '.', label='Supernovas', color='blue')

for i in range(len(r1)):

    plt.errorbar(z1[i],r1[i], yerr = r_err1[i], capsize = 3, color='blue')

plt.plot(z1, r1, '.', color='blue')

for i in range(len(r2)):

    plt.errorbar(z2[i],r2[i], yerr = r_err2[i], capsize = 3, color='blue')

plt.plot(z, r, '.', color='blue')

for i in range(len(r)):

    plt.errorbar(z[i],r[i], yerr = r_er[i], capsize = 3, color='blue')

plt.plot(z_o, CDM, color='red', label='$\Lambda$CDM concordance')

plt.plot(z_o, EdS, color='purple', label=r'Einstein-de Sitter model')

plt.xlabel(r'$z$')

plt.ylabel(r'$\mu$ [pc]')

plt.title(r'Distance modulus ($\mu$) as a function of redshift ($z$)')

plt.legend()

plt.grid(True)


plt.savefig('r frente a z extra datos.pdf')