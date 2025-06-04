import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc
)

st.set_page_config(layout="wide")  # Широкая раскладка

st.title("Классификация диабета — анализ с помощью моделей ML")

@st.cache_data
def load_data():
    return pd.read_csv("D:/Загрузки/diabetes.csv")

df = load_data()

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Две колонки: слева — настройки, справа — результаты
col_controls, col_results = st.columns([1, 2])

with col_controls:
    st.subheader("Настройки модели")
    model_type = st.selectbox("Выберите модель", ["SVC", "Случайный лес", "Градиентный бустинг", "Дерево решений"])

    if model_type == "SVC":
        C = st.slider("Обратный коэффициент регуляризации (C)", 0.01, 10.0, 1.0)
        model = SVC(C=C, probability=True, random_state=42)
        param_str = f"C = {C:.2f}"
    elif model_type == "Случайный лес":
        n_estimators = st.slider("Количество деревьев", 10, 200, 100, step=10)
        max_depth = st.slider("Максимальная глубина", 1, 30, 5)
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        param_str = f"n_estimators = {n_estimators}, max_depth = {max_depth}"
    elif model_type == "Градиентный бустинг":
        n_estimators = st.slider("Количество итераций", 10, 200, 100, step=10)
        learning_rate = st.slider("Скорость обучения", 0.01, 1.0, 0.1)
        max_depth = st.slider("Максимальная глубина", 1, 30, 3)
        model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=42)
        param_str = f"n_estimators={n_estimators}, lr={learning_rate:.2f}, max_depth={max_depth}"
    else:  # Дерево решений
        max_depth = st.slider("Максимальная глубина", 1, 30, 5)
        min_samples_split = st.slider("Минимальное число образцов для разбиения", 2, 20, 2)
        model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
        param_str = f"max_depth={max_depth}, min_samples_split={min_samples_split}"

# Обучаем модель и считаем метрики без кнопки
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

with col_results:
    st.subheader("Метрики модели")
    met_col1, met_col2, met_col3, met_col4 = st.columns(4)
    met_col1.metric("Accuracy", f"{acc:.2f}")
    met_col2.metric("Precision", f"{prec:.2f}")
    met_col3.metric("Recall", f"{rec:.2f}")
    met_col4.metric("F1-score", f"{f1:.2f}")



    # Барплот распределения TP, TN, FP, FN
    results_df = pd.DataFrame({'y_true': y_test, 'y_pred': y_pred})
    results_df["type"] = results_df.apply(lambda row:
        "TP" if row.y_true == 1 and row.y_pred == 1 else
        "TN" if row.y_true == 0 and row.y_pred == 0 else
        "FP" if row.y_true == 0 and row.y_pred == 1 else
        "FN", axis=1)


    fig, ax = plt.subplots(figsize=(6,4))
    sns.countplot(x="type", data=results_df, palette="Set2", order=["TP", "TN", "FP", "FN"], ax=ax)
    ax.set_xlabel("Тип предсказания")
    ax.set_ylabel("Количество")
    ax.set_title("Типы предсказаний модели")
    st.pyplot(fig)

    # ROC-кривая
    st.subheader("ROC-кривая")
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    fig_roc, ax_roc = plt.subplots(figsize=(6,4))
    ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax_roc.plot([0, 1], [0, 1], 'r--')
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC-кривая")
    ax_roc.legend(loc="lower right")
    st.pyplot(fig_roc)
