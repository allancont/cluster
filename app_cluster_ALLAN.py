import pandas as pd
import streamlit  as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import altair as alt
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from gower import gower_matrix
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title = 'Agrupamento Hierárquico', \
                   layout="wide",
                   initial_sidebar_state='expanded')

header = st.container()
dataset = st.container()
features = st.container()
model_training  = st.container()
features2 = st.container()

#################
# FUNÇÕES CACHE #
#################

@st.cache_data
def get_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00468/online_shoppers_intention.csv"
    df = pd.read_csv(url)
    return df

@st.cache_data()
def gerar_grafico(file_name):
    fig = sns.pairplot(file_name[['Administrative','Informational','ProductRelated','Administrative_Duration','Informational_Duration','ProductRelated_Duration','Revenue']], hue='Revenue')
    fig.fig.suptitle("Relações entre variáveis do conjunto de dados", y=1.05, x=0.5, ha='center', fontsize=16)
    st.pyplot(fig)



with header:
    # Título principal da aplicação
    st.title('Projeto de clusterização de sessões de acesso ao portal de compras online')
    st.write("O presente projeto foi baseado no trabalho publicado por Sakar, C.O., Polat, S.O., Katircioglu, M. et al - **Real-time prediction of online shoppers' purchasing intention using multilayer perceptron and LSTM recurrent neural networks** - cuja proposta inicial seria a proposição de um sistema de análise do comportamento do comprador online em tempo real capaz de prevê simultaneamente a intenção de compra do visitante e a probabilidade de abandono do site.")
    st.write("### Objetivos:")
    st.write("Agrupar as sessões de acesso ao portal considerando o comportamento de acesso e informações de sazonalidade, como a proximidade a uma data especial ou fim de semana.\n Para isso será utilizada a mesma base de registros de 12.330 sessões de acesso a páginas, cada sessão sendo de um único usuário em um período de 12 meses.")
    st.markdown("---")

with dataset:
    # with st.spinner('Carregando dados...'):
    dados = get_data()
    meses_dict = {'Jan': 1,'Feb': 2,'Mar': 3,'Apr': 4,'May': 5,'June': 6,'Jul': 7,'Aug': 8,'Sep': 9,'Oct': 10,'Nov': 11,'Dec': 12}
    dados['DtEspecial'] = dados['SpecialDay'].apply(lambda x: 'Não' if x == 0 else 'Sim')
    dados['Mes'] = dados['Month'].map(meses_dict).astype('int')

    #Padronizando variáveis
    var_pad1 = dados.select_dtypes(include="number")
    var_pad2 = dados.select_dtypes(exclude="number")
    colunas=var_pad1.columns.tolist()
    scaler = StandardScaler(with_mean=True,with_std=True)
    scaler.fit(var_pad1)

    # Ajustando valores da escala
    dados_pad = scaler.fit_transform(var_pad1)
    dados_numpad = pd.DataFrame(dados_pad,columns=colunas)
    dados_pad = pd.concat([dados_numpad,var_pad2],axis=1)
    st.write(dados.head())

with features:
    # with st.spinner('Gerando gráfico das variáveis...'):
    fig = gerar_grafico(dados_pad)
    st.markdown("---")

with model_training:
    sel_col, disp_col = st.columns(2)
    select_var=dados_pad.select_dtypes(include='float').columns.values.tolist()
    var_options = sel_col.multiselect('Selecione as variáveis desejadas',select_var)
    variaveis = var_options+['DtEspecial', 'Weekend', 'Revenue']    
    variaveis_cat = ['Weekend', 'Revenue', 'DtEspecial']
    df = dados_pad[variaveis].dropna()
    df = pd.get_dummies(dados_pad[variaveis].dropna())
    categorias=df.select_dtypes(exclude=float).columns.values
    vars_cat = [True if x in categorias else False for x in df.columns]
    d_col=df.columns.values.tolist()
    cat= {'variavel': d_col, 'categorica': vars_cat}
    n_grupos = sel_col.slider("Selecione o número de grupos",1,10,3)
with features2:
    def calcular_e_exibir():
        distancia_gower = gower_matrix(df, cat_features=vars_cat)
        gdv = squareform(distancia_gower,force='tovector')
        Z = linkage(gdv, method='complete')
        Z_df = pd.DataFrame(Z,columns=['id1','id2','dist','n'])
        # disp_col.write('Amostra de distâncias calculadas')
        # disp_col.write(Z_df.head())
        df['grupo_n3'] = fcluster(Z, n_grupos, criterion='maxclust')
        df3 = dados_pad.reset_index().merge(df.reset_index(), how='left')
        disp_col.write('Clusters')
        disp_col.write(df3.groupby([ 'Revenue','DtEspecial', 'grupo_n3'])['index'].count().unstack().fillna(0).style.format(precision=0))
        df_cat = pd.DataFrame(cat)
        col_scope = df_cat.loc[df_cat['categorica'] == False, 'variavel'].tolist()
        # col_scope = n_grupos+['grupo_n3','Revenue',']
        fig2 = sns.pairplot(df3[col_scope+['grupo_n3']], hue='grupo_n3',palette='tab10')
        fig2.fig.suptitle("Relações entre variáveis do conjunto de dados", y=1.05, x=0.5, ha='center', fontsize=16)
        st.pyplot(fig2)

    if st.button("Calcular"):
        calcular_e_exibir()  