import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
# import warnings
from io import BytesIO

# warnings.filterwarnings('ignore')

@st.cache_data(show_spinner=True)
def load_data(file_data):
    try:
        return pd.read_csv(file_data, sep=';')
    except:
        return pd.read_excel(file_data)

# @st.cache_data
# def multiselect_filter(relatorio, col, selecionados):
#     if 'all' in selecionados:
#         return relatorio
#     else:
#         return relatorio[relatorio[col].isin(selecionados)].reset_index(drop=True)

@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

@st.cache_data
def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, header=False,sheet_name='Sheet1')
    writer.save()
    processed_data = output.getvalue()
    return processed_data

def main():
    st.title("Importação de Folha de Pagamento")

    # Diretório onde estão localizados os arquivos .xlsx
    diretorio = 'C:/Users/Allan/OneDrive/Área de Trabalho'  # Substitua pelo caminho correto
    os.chdir(diretorio) # Diretório dos arquivos

    # Nome da pasta que você deseja importar
    nome_pasta = 'Integração'

    # Obtendo o início raiz dos arquivos
    file_type = st.text_input("Digite o início dos arquivos no formato AAAAMM:")
    file_extension ='.' + 'xlsx'

    # Localiza todos os arquivos que correspondem ao padrão de nome
    arquivos = [i for i in glob.glob(os.path.join(diretorio, f"{file_type}*{file_extension}"))]

    # Loop para importar a pasta 'integração' de cada arquivo encontrado
    dados = []
    for arquivo in arquivos:
        df = load_data(arquivo)
        dados.append(df)

    # Concatena todos os DataFrames em um único DataFrame
    dados = pd.concat(dados, ignore_index=True)
    df = dados[dados['CAMPO7'] != 0]

    # Definir as colunas desejadas
    colunas = ['Tipo', 'Sequência', 'Código Reduzido', 'Data Lançamento', 'Natureza', 'Documento', 'Valor', 'Cod_Hist', 'Histórico', 'Conciliado', 'Branco', 'Filial']
    df['CAMPO3'] = df['CAMPO3'].astype(str)
    df['CAMPO4'] = df['CAMPO4'].astype(str)
    df['CAMPO6'] = df['CAMPO6'].astype(str)
    df = df.rename(columns=dict(zip(df.columns, colunas)))
    # df = df[colunas]

    if st.button("Exibir informações"):
        st.subheader("Dados Importados")
        st.dataframe(df)

        st.subheader("Contagem dos lançamentos por filial")
        st.text(df.Filial.value_counts())

    # Exportação do DataFrame
    if st.button("Exportar para Excel"):
        arq_exp = f'integra_{file_type}.xlsx'
        excel_data = to_excel(df)
        st.download_button(f'Baixar {arq_exp}', excel_data, file_name=arq_exp, mime='application/octet-stream')

if __name__ == '__main__':
    main()