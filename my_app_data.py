import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from st_pages import Page, show_pages, add_page_title

class LogReg:
    def __init__(self, learning_rate, n_epochs):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.coef_ = np.random.uniform(-1, 1, size = X.shape[1]) # Cгенерируется столько весов, сколько у нас столбцов 
        self.intercept_ = np.random.uniform(-1, 1) # w0


        for epoch in range(self.n_epochs):

            y_pred = self.sigmoid(X@self.coef_ + self.intercept_) # 768 - предсказаний

            error =  (y_pred - y)

            grad_w0 = error # 768 
            grad_w = X * error.reshape(-1, 1) # (768, 2) Для всех элементов выборки посчитаны частные производны по w1, w2

            self.coef_ = self.coef_ - self.learning_rate * grad_w.mean(axis=0)
            self.intercept_ = self.intercept_ - self.learning_rate * grad_w0.mean()

    def predict(self, X):
        y_pred = self.sigmoid(X@self.coef_ + self.intercept_)
        return y_pred
    
    def score(self, X, y):
        y_pred = np.round(self.predict(X)) # Добавил еще округление вероятности к ближайшему классу
        return (y == y_pred).mean()

st.title("""
Логистическая регрессия
""")

st.write("""
Вы можете получить предсказание о том, был ли одобрен кредит клиенту, 
на основе данных о его доходе и кредитному рейтингу
""")

# Загрузка обучающей выборки
uploaded_files = st.file_uploader("Загрузите обучающую выборку CSV", accept_multiple_files=True)
if uploaded_files:
    for uploaded_file in uploaded_files:
        bytes_data = uploaded_file.read()
        uploaded_file.seek(0)
        train = pd.read_csv(uploaded_file)
        st.write(f"Загружен файл: {uploaded_file.name}")
        st.dataframe(data=train)

# Выбор колонок
        selected_columns = st.sidebar.multiselect(
            "Выберите параметры для обучения",
            train.columns
        )
        st.sidebar.write("Вы выбрали:", selected_columns)

        target = st.sidebar.selectbox(
            "Выберите таргет",
            train.columns
        )
        
        st.sidebar.write("Вы выбрали:", target, ". Наша модель должна предсказать этот параметр")

        # Выбор признаков для scatter-графика
        feature1 = st.sidebar.selectbox("Выберите первый признак для scatter-графика", train.columns)
        feature2 = st.sidebar.selectbox("Выберите второй признак для scatter-графика", train.columns)

        # Предобработка данных
        if selected_columns:
            scaler = StandardScaler()
            train[selected_columns] = scaler.fit_transform(train[selected_columns])
            X_train, y_train = train[selected_columns], train[target]


        agree = st.sidebar.checkbox("Стандартизировать данные")
        if agree:
            st.sidebar.write("Хорошо, данные будут стандартизированы")
            # Предобработка данных
            if selected_columns:
                scaler = StandardScaler()
                train[selected_columns] = scaler.fit_transform(train[selected_columns])
                X_train, y_train = train[selected_columns], train[target]


uploaded_files = st.file_uploader("Загрузите тестовую выборку CSV", accept_multiple_files=True)
if uploaded_files:
    for uploaded_file in uploaded_files:
        bytes_data = uploaded_file.read()
        uploaded_file.seek(0)
        test = pd.read_csv(uploaded_file)
        st.write(f"Загружен файл: {uploaded_file.name}")
        st.dataframe(data=test)

        agree = st.sidebar.checkbox("Если вы стандартизировали данные для обучения, рекомендуем стандартизировать эти данные тоже")
        if agree:
            st.sidebar.write("Хорошо, данные будут стандартизированы")
            scaler = StandardScaler()
            # Предобработка данных
            test[selected_columns] = scaler.fit_transform(test[selected_columns])
            X_test, y_test = test[selected_columns], test[target]

st.write("""
Тестовая выборка должна содержать доход, рейтинг по кредитной карте
""")

# Параметры обучения
learning_rate = st.sidebar.slider(
    "Выберите learning rate",
    min_value=0.0,
    max_value=1.0,
    step=0.01,
    value=0.01,
    format="%f")
st.sidebar.write("Ваш learning rate равен", learning_rate)

n_epochs = st.sidebar.slider("Количество эпох", min_value=1, max_value=1000, value=1)
st.sidebar.write("Количество эпох равно", n_epochs)

# Кнопка "Начать обучение"
if st.sidebar.button("Начать обучение", type="primary"):
    # Проверяем, загружены ли данные
    if 'train' in locals() and 'test' in locals(): 
        # Создаем модель
        my_model = LogReg(learning_rate, n_epochs)

        # Обучаем модель
        my_model.fit(X_train, y_train)

        # Создаем словарь весов
        weights = dict(zip(selected_columns, my_model.coef_))
        weights['Свободный член'] = my_model.intercept_

        st.write("Веса модели:")
        st.write(weights)
        
        # Оцениваем точность
        train_accuracy = my_model.score(X_train, y_train)
        test_accuracy = my_model.score(X_test, y_test)
        st.write(f"Точность на обучающей выборке: {train_accuracy}")
        st.write(f"Точность на тестовой выборке: {test_accuracy}")

        # Строим scatter-график
        fig, ax = plt.subplots()
        ax.scatter(test[feature1], test[feature2], c=test[target])
        x = ax.set_xlabel(feature1)
        ax.set_ylabel(feature2)
        ax.set_title("Scatter-график с выделением таргета")
        st.pyplot(fig)

        
    else:
        st.error("Загрузите обучающую и тестовую выборки!")