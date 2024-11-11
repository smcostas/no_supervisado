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
from matplotlib.cm import viridis
from matplotlib.colors import to_hex

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
df_uniques['age'].max()
df_uniques['age'].min()

intervalos = [17, 23, 30, 35,float('inf')]
etiquetas = ['rookie', 'prime', 'experienced', 'in_decline']
df_uniques['carreer_state'] = pd.cut(df_uniques['age'], bins=intervalos, labels=etiquetas, right=False)
df_uniques['carreer_state'].value_counts()


df_int = df_uniques.iloc[:, 47:76]


df_int['pos2'] = df_uniques['pos2']
df_int['carreer_state'] = df_uniques['carreer_state']

sns.pairplot(df_int, hue='pos2', corner = True)
plt.show()

sns.pairplot(df_int, hue='carreEr_state', corner = True) ### aca no se va a apreciar ya que estan todas las posiciones mezcladas por si hiciera boxplots por posicion
plt.show()

list(df_int.columns)

g = sns.FacetGrid(df_int, col = 'pos2', col_wrap = 3, sharey=True)
g.map_dataframe(sns.boxplot, y= 'attacking_finishing', x = 'carreer_state') 
plt.show()

g = sns.FacetGrid(df_int, col = 'pos2', col_wrap = 3, sharey=True)
g.map_dataframe(sns.boxplot, y= 'skill_long_passing', x = 'carreer_state') 
plt.show()

g = sns.FacetGrid(df_int, col = 'pos2', col_wrap = 3, sharey=True)
g.map_dataframe(sns.boxplot,y = 'mentality_positioning', x = 'carreer_state')
plt.show()

g = sns.FacetGrid(df_int, col = 'pos2', col_wrap = 3, sharey=True)
g.map_dataframe(sns.boxplot,y = 'defending_marking_awareness', x = 'carreer_state')

plt.show()


sns.boxplot(df_uniques, x = 'carreer_state', y = 'overall')
plt.show()
## dependiendo de la posicion hay un efecto de la edad en los stats

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
df_uniques2 = df_uniques.copy()
df_uniques2 = df_uniques2[df_uniques2['pos2'] != 'GK']
sub_df2 = sub_df.copy()
sub_df2[['pace','shooting','passing', 'dribbling', 'defending', 'physic']] = df_uniques2[['pace','shooting','passing', 'dribbling', 'defending', 'physic']]


sub_df2['carreer_state'] = df_uniques2['carreer_state']


# Function
def plot_radar_chart(ax,data, metrics, title):
    N = len(metrics)
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False) ## defino los angulos de la escala polar de acuerdo a N variables
    theta = np.concatenate([theta, [theta[0]]])
    
    ax.set_title(title, y=1.05, fontsize=20)
    ax.set_theta_zero_location("N") ##0 en el norte (arriba)
    ax.set_theta_direction(-1) ## para que el aumento del angulo sea en direccion de las agujas del reloj
    ax.set_rlabel_position(90) ## para marcar los porcentajes en el lado derecho (65 grados
    ax.xaxis.set_tick_params(grid_color='#A9A9A9', grid_linewidth=2, zorder=3) ## para que vayan por encima
    ax.yaxis.set_tick_params(grid_color='#A9A9A9', grid_linewidth=2, zorder=3)
    ax.grid(True, color='#A9A9A9', linewidth=2, zorder=3)
    
    career_colors = {
        'rookie': '#2ECC71',      # Verde
        'prime': '#E67E22',       # Naranja
        'experienced': '#3498DB', # Azul
        'in_decline': '#E74C3C'   # Rojo
    }
    
    data = data.sort_values(by='carreer_state')
    
    for idx, (i, row) in enumerate(data.iterrows()): ## crea tantas capas como filas tenga el data set (lo que se va a plotear)
        values = row[metrics].values.flatten().tolist()
        values = values + [values[0]] ## agregar el primer valor al final de la fila para cerrar el plot
        
        carreer_state = row['carreer_state']
        color = career_colors.get(carreer_state, '#000000') 
        
        ax.plot(theta, values, linewidth=1, alpha = 0.35, linestyle='solid', label= carreer_state, marker='o', markersize=10, color=color, zorder = 1)

    
    median_values = data[metrics].mean().tolist()
    median_values = median_values + [median_values[0]]
    ax.plot(theta, median_values, linewidth=1.5, linestyle='solid', marker='o', label = 'mean values', markersize=15, color = 'black', zorder = 2)
    
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0", "25", "50", "75", "100"], color="black", size=10, zorder=5)
    ax.set_ylim(0,100)
    ax.set_xticks(theta)
    ax.set_xticklabels(metrics + [metrics[0]], color='black', size=12, zorder=5)



df_uniques2.carreer_state.value_counts()

metrics = ['pace','shooting','passing', 'dribbling', 'defending', 'physic']


fig, axs = plt.subplots(2, 3, figsize=(20, 20), subplot_kw={'projection': 'polar'})

plot_radar_chart(axs[0,0], sub_df2[sub_df2['pos2'] == 'Fullback'], metrics, 'Fullbacks features scores')
plot_radar_chart(axs[0,2], sub_df2[sub_df2['pos2'] == 'Defender'], metrics, 'Defenders features scores')
plot_radar_chart(axs[1, 0], sub_df2[sub_df2['pos2'] == 'Midfielder'], metrics, 'Midfielders features scores')
plot_radar_chart(axs[1, 1], sub_df2[sub_df2['pos2'] == 'Wing'], metrics, 'Wing features scores')
plot_radar_chart(axs[1, 2], sub_df2[sub_df2['pos2'] == 'Forward'], metrics, 'Forwards features scores')
axs[0, 1].axis('off')
handles, labels = axs[0, 2].get_legend_handles_labels()
unique = dict(zip(labels, handles))
fig.legend(unique.values(), unique.keys(), loc= 'center', fontsize=14)
plt.show()
### que pasa si filtro por overall 

sub_df2['Overall'] = df_uniques2['overall']

sub_df2 = sub_df2[sub_df2['Overall'] > 75]


sub_df2.carreer_state.value_counts()

fig, axs = plt.subplots(2, 3, figsize=(20, 20), subplot_kw={'projection': 'polar'})

plot_radar_chart(axs[0,0], sub_df2[sub_df2['pos2'] == 'Fullback'], metrics, 'Fullbacks features scores')
plot_radar_chart(axs[0,2], sub_df2[sub_df2['pos2'] == 'Defender'], metrics, 'Defenders features scores')
plot_radar_chart(axs[1, 0], sub_df2[sub_df2['pos2'] == 'Midfielder'], metrics, 'Midfielders features scores')
plot_radar_chart(axs[1, 1], sub_df2[sub_df2['pos2'] == 'Wing'], metrics, 'Wing features scores')
plot_radar_chart(axs[1, 2], sub_df2[sub_df2['pos2'] == 'Forward'], metrics, 'Forwards features scores')
axs[0, 1].axis('off')
handles, labels = axs[1, 1].get_legend_handles_labels()
unique = dict(zip(labels, handles))
fig.legend(unique.values(), unique.keys(), loc= 'center', fontsize=14)
plt.show()

sub_df3 = sub_df2[metrics]
sub_df3[['pos2', 'carreer_state']] = sub_df2[['pos2', 'carreer_state']]
stats_defenders = sub_df3[sub_df3['pos2'] == 'Defender'].groupby('carreer_state').describe()
for metric in metrics:
    print('\n',metric)
    print(stats_defenders[metric])

    
#%%
#### hierichal clustring
df_cluster = df_uniques2.iloc[:, 47:76]
df_cluster['age'] = df_uniques2.loc[:, 'age']
scaler = StandardScaler()

df_cluster_s = scaler.fit_transform(df_cluster)
df_cluster_s = pd.DataFrame(df_cluster_s, columns=df_cluster.columns, index=df_cluster.index)



plt.figure(figsize=(10, 7))  
plt.title("Dendrograms using Ward")  
dend = shc.dendrogram(shc.linkage(df_cluster_s, method='ward'))
plt.show()
## congifurando la distancia de separacion a 45 podria encontrar 5 grupos .. probar

plt.figure(figsize=(10, 7))
plt.title("Dendrograms using Complete")
dend1 = shc.dendrogram(shc.linkage(df_cluster_s, method='complete'))
plt.show()
## si bien encuentra 4 grupos, con un pequeño ajuste de distancia encontraria 8 grupos , ojo!
### este si ramifica mucho internamente cada nodo pero muestra muchos grupos
plt.figure(figsize=(10, 7))
plt.title("Dendrograms using Single")
dend2 = shc.dendrogram(shc.linkage(df_cluster_s, method='single'))
plt.show()

### detecta demsiados grupos a mi entender, es demasiado poco conservador
plt.figure(figsize=(10, 7))
plt.title("Dendrograms using Average")
dend3 = shc.dendrogram(shc.linkage(df_cluster_s, method='average'))
plt.show()

#%%% Evaluacion con metricas del clustering
# usando silluette score
from scipy.spatial.distance import pdist
from sklearn.metrics import silhouette_score

linkage_ward = shc.linkage(df_cluster_s, method='ward')
linkage_complete = shc.linkage(df_cluster_s, method='complete')

coph_dist_ward = shc.cophenet(linkage_ward, pdist(df_cluster_s))
coph_dist_complete = shc.cophenet(linkage_complete, pdist(df_cluster_s))
print(f"Cophenetic Correlation Coefficient (Ward): {coph_dist_ward[0]:.4f}")
print(f"Cophenetic Correlation Coefficient (Complete): {coph_dist_complete[0]:.4f}")

'''
inconsistency_ward = shc.inconsistent(linkage_ward)
inconsistency_complete = shc.inconsistent(linkage_complete)
print("Inconsistency Coefficients (Ward):")
print(inconsistency_ward)
print("\nInconsistency Coefficients (Complete):")
print(inconsistency_complete)
'''
distances_comp = [12.3, 11.25, 10]
distances_ward = [80, 50, 40,]
for dist in distances_comp:
    # Obtener clusters con corte de linkage basado en la distancia
    labels_complete = shc.fcluster(linkage_complete, t=dist, criterion='distance')
    
    # Contar el número de clústeres generados
    num_clusters_complete = len(set(labels_complete))

    if num_clusters_complete > 1:
        silhouette_complete = silhouette_score(df_cluster_s, labels_complete)
    else:
        silhouette_complete = -1  # Indicador de no cálculo válido
    
    print(f"\nDistancia de corte: {dist}")
    print(f"  Complete - Número de Clústeres: {num_clusters_complete}, Silhouette Score: {silhouette_complete:.4f}")
    


for dist in distances_ward:
    # Obtener clusters con corte de linkage basado en la distancia
    labels_ward = shc.fcluster(linkage_ward, t=dist, criterion='distance')
    
    # Contar el número de clústeres generados
    num_clusters_ward = len(set(labels_ward))

    if num_clusters_ward > 1:
        silhouette_ward = silhouette_score(df_cluster_s, labels_ward)
    else:
        silhouette_ward = -1  # Indicador de no cálculo válido
    
    print(f"\nDistancia de corte: {dist}")
    print(f"  Complete - Número de Clústeres: {num_clusters_ward}, Silhouette Score: {silhouette_ward:.4f}")



#%%%
Z = shc.linkage(df_cluster_s, method='complete')
max_distance = np.max(Z[:, 2]) ## distancia maxima entre los puntos mas extremos
color_threshold = 0.7 * max_distance ## asi como da 
clusters = shc.fcluster(Z, t = color_threshold, criterion = 'distance')
df_uniques2['clusters'] = clusters

#%%%%

sns.boxplot(df_uniques2, x = 'clusters', y = 'overall')
plt.show()
sns.boxplot(df_uniques2, x = 'pos2', y = 'overall')
plt.show()


g = sns.FacetGrid(df_uniques2, col = 'clusters', sharey = True, sharex = False)
g.map_dataframe(sns.boxplot, y= 'overall', x = 'pos2') 
plt.show()


sns.boxplot(df_uniques2, x = 'clusters', y = 'defending_marking_awareness') ## el gurpo 2 y 3 tienen jugadores que tienen buenos stats defensivos.
plt.show()
sns.boxplot(df_uniques2, x = 'clusters', y = 'defending_standing_tackle')
plt.show()
sns.boxplot(df_uniques2, x = 'clusters', y = 'defending_sliding_tackle')
plt.show()
sns.boxplot(df_uniques2, x = 'clusters', y = 'attacking_short_passing')
plt.show()
sns.boxplot(df_uniques2, x = 'clusters', y = 'attacking_crossing')
plt.show()
sns.boxplot(df_uniques2, x = 'clusters', y = 'skill_long_passing')
plt.show()
sns.boxplot(df_uniques2, x = 'clusters', y = 'power_stamina')
plt.show()
sns.boxplot(df_uniques2, x = 'clusters', y = 'mentality_positioning')
plt.show()
## el grupo 1 contiene los jugadores mas rapidos con mejor dribbling y mejor definicion... hay patrones
sns.boxplot(df_uniques2, x = 'clusters', y = 'skill_dribbling')
plt.show()
sns.boxplot(df_uniques2, x = 'clusters', y = 'attacking_finishing')
plt.show()
sns.boxplot(df_uniques2, x = 'clusters', y = 'attacking_volleys')
plt.show()
sns.boxplot(df_uniques2, x = 'clusters', y = 'attacking_heading_accuracy') ## no parece haber diferencias en heading
plt.show()
sns.boxplot(df_uniques2, x = 'clusters', y = 'movement_sprint_speed')
plt.show()



df_uniques2.clusters.value_counts()
df_uniques2.pos2.value_counts()
df_uniques2.groupby(['clusters']).pos2.value_counts() ## el grupo 1 no contiene defensores, tiene sentido por estar vinculado mas a lo ofensivo(y pocos fullbacks, supongo que solo los defensivos) 
## el grupo 2 contiene mas que nada midfielders y fullbacks  entiendo que son jugadores con ritmo de caracter defensivo
## el grupo 3 contiene casi todos los defensores, contiene a todos los que tiene los mejores stats defensivos pero que son lentos, con peor pase, menos stamina. etc  Los mejores defensores van a caer en el 2
## el grupo 4 contiene a delanteros menos versatiles diria. Aca caeran los peores wing y mco, pero muchos buenos 9 de area



def plot_radar_chart(ax,data, metrics, title):
    N = len(metrics)
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False) ## defino los angulos de la escala polar de acuerdo a N variables
    theta = np.concatenate([theta, [theta[0]]])
    
    
    ax.set_title(title, y=1.05, fontsize=20)
    ax.set_theta_zero_location("N") ##0 en el norte (arriba)
    ax.set_theta_direction(-1) ## para que el aumento del angulo sea en direccion de las agujas del reloj
    ax.set_rlabel_position(65) ## para marcar los porcentajes en el lado derecho (65 grados
    ax.xaxis.set_tick_params(grid_color='#A9A9A9', grid_linewidth=2, zorder=3) ## para que vayan por encima
    ax.yaxis.set_tick_params(grid_color='#A9A9A9', grid_linewidth=2, zorder=3)
    ax.grid(True, color='#A9A9A9', linewidth=2, zorder=3)
    pos_colors = {
    'Defender': '#87CEEB',   # skyblue
    'Fullback': '#3498DB',   # azul
    'Midfielder': '#2ECC71', # verde
    'Wing': '#FFA500',       # naranja
    'Forward': '#E74C3C'     # rojo
    }
    
    data = data.sort_values(by='pos2')
    
    for idx, (i, row) in enumerate(data.iterrows()): ## crea tantas capas como filas tenga el data set (lo que se va a plotear)
        values = row[metrics].values.flatten().tolist()
        values = values + [values[0]] ## agregar el primer valor al final de la fila para cerrar el plot
        
        pos = row['pos2']
        color = pos_colors.get(pos, '#000000') 
        
        ax.plot(theta, values, linewidth=1, alpha = 0.35, linestyle='solid', label= pos, marker='o', markersize=10, color=color, zorder = 1 )

    
    median_values = data[metrics].median().tolist()
    median_values = median_values + [median_values[0]]
    ax.plot(theta, median_values, linewidth=1.5, linestyle='solid', marker='o', label = 'median values', markersize=15, color = 'black')
    
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0", "25", "50", "75", "100"], color="black", size=10, zorder=5)
    ax.set_ylim(0,100)
    ax.set_xticks(theta)
    ax.set_xticklabels(metrics + [metrics[0]], color='black', size=12, zorder=5)



metrics = ['attacking_finishing', 'attacking_crossing','skill_dribbling', 'attacking_short_passing','movement_acceleration', 'movement_sprint_speed', 'defending_marking_awareness', 'defending_standing_tackle']


fig, axs = plt.subplots(2, 3, figsize=(20, 20), subplot_kw={'projection': 'polar'})

plot_radar_chart(axs[0,0], df_uniques2[df_uniques2['clusters'] == 1], metrics, 'cluster 1 features scores')
plot_radar_chart(axs[0,2], df_uniques2[df_uniques2['clusters'] == 2], metrics, 'cluster 2 features scores')
plot_radar_chart(axs[1, 0], df_uniques2[df_uniques2['clusters'] == 3], metrics, 'cluster 3 features scores')
plot_radar_chart(axs[1, 1],df_uniques2[df_uniques2['clusters'] == 4], metrics, 'cluster 4 scores')
plot_radar_chart(axs[1, 2],df_uniques2[df_uniques2['clusters'] == 5], metrics, 'cluster 4 scores')
handles, labels = axs[0, 0].get_legend_handles_labels()
unique = dict(zip(labels, handles))
fig.legend(unique.values(), unique.keys(), loc= 'center', fontsize=14)
plt.subplots_adjust(wspace=0.3, hspace=0.4)
plt.show()



i_columns = df_uniques2.iloc[:, 47:76].columns
sns.pairplot(df_uniques2, hue='clusters', 
             x_vars= i_columns,
             y_vars= i_columns,palette = 'viridis', corner = True)

plt.show()




#%%% Embeedings 

from sklearn.decomposition import PCA

pca = PCA(n_components=2)  # Reducimos a 2 componentes principales
pca_result = pca.fit_transform(df_cluster_s) ## ya esta estandarizado
explained_variance_ratio = pca.explained_variance_ratio_ ## los dos primeros ejes explican el 60 por ciento de la variacion

pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'], index=df_cluster_s.index)

pca_df['Cluster'] = clusters

plt.figure(figsize=(12, 8))
sns.scatterplot(
    x='PC1', y='PC2', 
    hue='Cluster', 
    palette='viridis',  
    data=pca_df, 
    s=100, alpha=0.7
)


loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
threshold = 0.3 ## me quedo solo con aquellos que tengan carga  por encima de 0.3 

for i, (var, (x, y)) in enumerate(zip(df_cluster_s.columns, loadings)):
    # Solo grafica las flechas y etiquetas si el loading es mayor al umbral
    if np.abs(x) > threshold or np.abs(y) > threshold:
        plt.arrow(0, 0, x * 5, y * 5, color='red', alpha=0.7, head_width=0.2, head_length=0.3)
        plt.text(x * 5.5, y * 5.5, var, color='red', ha='center', va='center', fontsize=12)
plt.title('PCA con Clusters y Vectores de las Variables Originales')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.2f}%)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.2f}%)')
plt.grid(True)
plt.show()