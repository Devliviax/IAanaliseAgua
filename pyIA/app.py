import streamlit as st
import psycopg2
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

# Desativar o aviso PyplotGlobalUseWarning
st.set_option('deprecation.showPyplotGlobalUse', False)

# Defina as informações de conexão ao banco de dados
hostname = "dpg-ckonrk41tcps73b8raj0-a.oregon-postgres.render.com"
port = 5432
database = "dashboarddatabase"
username = "dashboardusa"
password = "age085yQwL1W3ZXs2pJS1Tk3QLKxr4LL"

try:
    # Conecte ao banco de dados com SSL
    conn = psycopg2.connect(
        host=hostname,
        port=port,
        database=database,
        user=username,
        password=password,
        sslmode="require",
    )

    cursor = conn.cursor()

    # Execute uma consulta SQL na tabela do banco de dados
    cursor.execute("SELECT * FROM public.water_potability")  # Certifique-se de que a tabela está no esquema "public"

    dados = cursor.fetchall()

    # Exiba os dados em uma tabela Streamlit
    st.header('Tabela de Dados do Banco de Dados')
    df = pd.DataFrame(dados, columns=[desc[0] for desc in cursor.description])
    st.dataframe(df)

    conn.close()

except Exception as e:
    st.error(f"Erro na conexão ao banco de dados: {e}")

# Resto do código para análise de dados e modelo de aprendizado de máquina

# Adicione widgets no Streamlit para análise e exibição de gráficos
if st.checkbox("Mostrar Gráfico de Correlação"):
    plt.figure(figsize=(15, 9))
    sns.heatmap(df.corr(), annot=True)
    st.pyplot()

if st.checkbox("Mostrar Estatísticas"):
    st.write(df.info())
    st.write(df.isnull().sum())

# Treinamento de modelos e exibição de precisões
if st.button("Treinar Modelos"):
    X = df.drop(['potability'], axis=1)  # Use 'potability' em letras minúsculas
    y = df['potability']

    # Trate os valores ausentes com o SimpleImputer (substituindo NaN pela média)
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    # Padronize os dados após o tratamento dos valores ausentes
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    knn = KNeighborsClassifier(n_neighbors=35)
    dt = DecisionTreeClassifier(criterion='entropy', max_depth=9, min_samples_leaf=30)
    rf = RandomForestClassifier(min_samples_leaf=2, n_estimators=500)
    ada = AdaBoostClassifier(learning_rate=0.2, n_estimators=50)

    knn_model = knn.fit(X_train, y_train)
    y_pred = knn_model.predict(X_test)
    accuracy_knn = accuracy_score(y_test, y_pred)

    dt_model = dt.fit(X_train, y_train)
    y_pred = dt_model.predict(X_test)
    accuracy_dt = accuracy_score(y_test, y_pred)

    rf_model = rf.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    accuracy_rf = accuracy_score(y_test, y_pred)

    ada_model = ada.fit(X_train, y_train)
    y_pred = ada_model.predict(X_test)
    accuracy_ada = accuracy_score(y_test, y_pred)

    st.write('A precisão do modelo knn é:', accuracy_knn)
    st.write('A precisão do modelo Decision Tree é:', accuracy_dt)
    st.write('A precisão do modelo Random Forest é:', accuracy_rf)
    st.write('A precisão do modelo Ada Boost é:', accuracy_ada)