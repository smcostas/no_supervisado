# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 18:48:58 2024

@author: santi
"""

import os 
os.chdir('downloads/ap no supervisado')
import pandas as pd 
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import normalize, StandardScaler
import scipy.cluster.hierarchy as shc


#%%
df = pd.read_csv('female_players.csv')

#%% version 23
df['fifa_version'].value_counts()
df = df[df['fifa_version'] == 23]
df['fifa_update'].value_counts()
len(df['short_name'].unique())


'''
def get_index_bv(name):
    index = df[df['short_name'] == name].fifa_update.idxmax()
    return index
ids = [get_index_bv(name) for name in df['short_name'].unique()]
df_uniques = df.iloc[ids, :]
'''
df_uniques = df.loc[df.groupby('short_name')['fifa_update'].idxmax()]

df.info(verbose = True, show_counts=True)
df.describe()
## las medias y los desvios cambian mucho entre variables, a pesar de tener la misma escala esta varaibilidad puede imponer que algunas variables tengan mas peso que otras por lo que será necesario estandarizar
#%%
df.player_positions.value_counts()
## a sumimos que la primera que sale es la posicion principal

def split_positions(fila):
    elementos = fila.replace(' ', '').split(',')
    principal_position = elementos[0]
    secondary_positions = elementos[1:]
    
    return pd.Series([principal_position, secondary_positions])

df_uniques[['principal_position', 'secondary_positions']] = df_uniques['player_positions'].apply(split_positions)
df_uniques.principal_position.value_counts()
df_uniques.secondary_positions.value_counts()
'''
forwards=['ST']
midfielders=['CDM','CM']
attackmidfielders = ['CAM', 'CF']
defenders=['CB']
fullbacks = ['LB','RB', 'RWB','LWB']
wings = ['RM', 'LM', 'LW', 'RW']
goalkeepers=['GK']

def pos2(position):
    if position in forwards:
        return 'Forward'
    elif position in wings:
        return 'Wing'

    elif position in midfielders:
        return 'Midfielder'
    
    elif position in attackmidfielders:
        return 'Attackmidfielder'

    elif position in defenders:
        return 'Defender'
    
    elif position in fullbacks:
        return 'Fullback'

    elif position in goalkeepers:
        return 'GK'

    else:
        return 'nan'
'''
forwards=['ST', 'CF', ]
midfielders=['CDM','CM', 'CAM']
defenders=['CB']
wings = ['LW', 'RW', 'RM', 'LM' ]
fullbacks = ['LB','RB', 'RWB','LWB']
goalkeepers=['GK']
def pos2(position):
    if position in forwards:
        return 'Forward'

    elif position in wings:
        return 'Wing'

    elif position in midfielders:
        return 'Midfielder'

    elif position in defenders:
        return 'Defender'
    
    elif position in fullbacks:
        return 'Fullback'

    elif position in goalkeepers:
        return 'GK'

    else:
        return 'nan'

df_uniques['pos2'] = df_uniques['principal_position'].apply(pos2)

#%% pair plots

df_int = df_uniques.iloc[:, 47:75]

df_int['pos2'] = df_uniques['pos2']

sns.pairplot(df_int, hue='pos2', corner = True)
plt.show()
#%%%
## por otra parte vamos a sacar las arqueras, dado que ya hemos mostrado que todas las variables separan muy bien este grupo, pero vemos que podría agregar ruido


sub_df = df_int[df_int['pos2'] != 'GK']

sns.pairplot(sub_df, hue='pos2', 
             x_vars=['defending_marking_awareness', 'mentality_positioning', 'mentality_interceptions',
                     'mentality_vision', 'skill_long_passing', 'attacking_volleys', 'attacking_finishing'],
             y_vars=['defending_marking_awareness', 'mentality_positioning', 'mentality_interceptions',
                     'mentality_vision', 'skill_long_passing', 'attacking_volleys', 'attacking_finishing'], corner = True)
plt.show()

#%% ploteo exploratorio por posicion
df_uniques2 = df_uniques[df_uniques['pos2'] != 'GK']
sub_df2 = sub_df.copy()
sub_df2[['pace','shooting','passing', 'dribbling', 'defending', 'physic']] = df_uniques2[['pace','shooting','passing', 'dribbling', 'defending', 'physic']]
sub_df2['short_name'] = df_uniques2['short_name']

# Function
def plot_radar_chart(ax,data, metrics, title):
    N = len(metrics)
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False) ## defino los angulos de la escala polar de acuerdo a N variables
    theta = np.concatenate([theta, [theta[0]]])
    
    
    ax.set_title(title, y=1.05, fontsize=20)
    ax.set_theta_zero_location("N") ##0 en el norte (arriba)
    ax.set_theta_direction(-1) ## para que el aumento del angulo sea en direccion de las agujas del reloj
    ax.set_rlabel_position(90) ## para marcar los porcentajes en el lado derecho (90 grados)
    ax.xaxis.set_tick_params(grid_color='lightgrey', grid_linewidth=2, zorder=1)
    ax.yaxis.set_tick_params(grid_color='lightgrey', grid_linewidth=2, zorder=1)
    
        
    for idx, (i, row) in enumerate(data.iterrows()): ## crea tantas capas como filas tenga el data set (lo que se va a plotear)
        values = row[metrics].values.flatten().tolist()
        values = values + [values[0]] ## agregar el primer valor al final de la fila para cerrar el plot
        ax.plot(theta, values, linewidth=1.75, linestyle='solid', marker='o', markersize=10, color = 'skyblue')
    
    median_values = data[metrics].mean().tolist()
    median_values = median_values + [median_values[0]]
    ax.plot(theta, median_values, linewidth=2.5, linestyle='solid', marker='o', label = 'mean values', markersize=15, color = 'green')
    
    ax.set_yticks([-2.5, -1.25, 0, 1.25, 2.5])
    ax.set_yticklabels(["2.5", "-1.25", "0", "1.25", "2.5"], color="black", size=10)
    ax.set_xticks(theta)
    ax.set_xticklabels(metrics + [metrics[0]], color='black', size=12)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1)) 


metrics = ['pace','shooting','passing', 'dribbling', 'defending', 'physic']

fig, axs = plt.subplots(2, 3, figsize=(20, 20), subplot_kw={'projection': 'polar'})

plot_radar_chart(axs[0,0], sub_df2[sub_df2['pos2'] == 'Fullback'], metrics, 'Fullbacks features scores')
plot_radar_chart(axs[0,2], sub_df2[sub_df2['pos2'] == 'Defender'], metrics, 'Defenders features scores')
plot_radar_chart(axs[1, 0], sub_df2[sub_df2['pos2'] == 'Midfielder'], metrics, 'Midfielders features scores')
plot_radar_chart(axs[1, 1], sub_df2[sub_df2['pos2'] == 'Wing'], metrics, 'Wing features scores')
plot_radar_chart(axs[1, 2], sub_df2[sub_df2['pos2'] == 'Forward'], metrics, 'Forwards features scores')
axs[0, 1].axis('off')
plt.show()





    
#%%%%%%%%%%%%%%%%%%%%%
#### hierichal clustring
df_cluster = sub_df.drop('pos2', axis = 1)
scaler = StandardScaler()

df_cluster_s = scaler.fit_transform(df_cluster)
df_cluster_s = pd.DataFrame(df_cluster_s, columns=df_cluster.columns)



plt.figure(figsize=(10, 7))  
plt.title("Dendrograms using Ward")  
dend = shc.dendrogram(shc.linkage(df_cluster_s, method='ward'))
plt.show()
## parece ser demasiado consevador

plt.figure(figsize=(10, 7))
plt.title("Dendrograms using Complete")
dend1 = shc.dendrogram(shc.linkage(df_cluster_s, method='complete'))
plt.show()

### este si remifica mucho internamente cada nodo pero muestra muchos grupos
plt.figure(figsize=(10, 7))
plt.title("Dendrograms using Single")
dend2 = shc.dendrogram(shc.linkage(df_cluster_s, method='single'))
plt.show()

### detecta demsiados grupos a mi entender, es demasiado poco conservador
plt.figure(figsize=(10, 7))
plt.title("Dendrograms using Average")
dend3 = shc.dendrogram(shc.linkage(df_cluster_s, method='average'))
plt.show()

Z = shc.linkage(df_cluster_s, method='complete')

max_distance = np.max(Z[:, 2]) ## distancia maxima entre los puntos mas extremos
color_threshold = 0.7 * max_distance
clusters = shc.fcluster(Z, t = color_threshold, criterion = 'distance')

df_cluster_s['clusters'] = clusters


#%%%%
df_cluster_s2 = df_cluster_s.copy()
df_cluster_s2[['pace','shooting','passing', 'dribbling', 'defending', 'physic']] = df_uniques2[['pace','shooting','passing', 'dribbling', 'defending', 'physic']]
metrics = ['attacking_finishing','skill_dribbling', 'movement_acceleration', 'mentality_composure', 'defending_marking_awareness' ]
df_cluster_s2.clusters.value_counts()



fig, axs = plt.subplots(2, 3, figsize=(20, 20), subplot_kw={'projection': 'polar'})

plot_radar_chart(axs[0,0], df_cluster_s2[df_cluster_s2['clusters'] == 1], metrics, 'cluster 1 features scores')
plot_radar_chart(axs[0,2], df_cluster_s2[df_cluster_s2['clusters'] == 2], metrics, 'cluster 2 features scores')
plot_radar_chart(axs[1, 0], df_cluster_s2[df_cluster_s2['clusters'] == 3], metrics, 'cluster 3 features scores')
plot_radar_chart(axs[1, 1],df_cluster_s2[df_cluster_s2['clusters'] == 4], metrics, 'cluster 4 scores')
plot_radar_chart(axs[1, 2], df_cluster_s2[df_cluster_s2['clusters'] == 5], metrics, 'cluster 5 features scores')
axs[0, 1].axis('off')
plt.show()



i_columns = df_uniques.iloc[:, 47:75].columns
sns.pairplot(df_cluster_s2, hue='clusters', 
             x_vars= i_columns,
             y_vars= i_columns,palette = 'viridis', corner = True)

plt.show()


