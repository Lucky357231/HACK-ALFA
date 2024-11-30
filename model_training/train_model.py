import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import json

# Загрузка данных из JSON
with open("clients.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Подготовка данных
def preprocess_data(data):
    df = pd.json_normalize(data)
    df["segment"] = df["segment"].astype("category").cat.codes
    df["role"] = df["role"].astype("category").cat.codes
    df["current_method"] = df["current_method"].astype("category").cat.codes
    df["target"] = df["available_methods"].apply(lambda x: x[0])  # Первый метод как цель
    df["target"] = df["target"].astype("category").cat.codes
    # Добавляем признаки
    df["signatures_mobile_common"] = df["signatures.common.mobile"]
    df["signatures_web_common"] = df["signatures.common.web"]
    df["signatures_mobile_special"] = df["signatures.special.mobile"]
    df["signatures_web_special"] = df["signatures.special.web"]
    df["available_methods_count"] = df["available_methods"].apply(len)
    return df

processed_data = preprocess_data(data)

# Выбор признаков
X = processed_data[[
    "segment", "role", "organizations_count", "current_method", "claims",
    "signatures_mobile_common", "signatures_web_common",
    "signatures_mobile_special", "signatures_web_special", "available_methods_count"
]]
y = processed_data["target"]

# Стандартизация признаков
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Настройка гиперпараметров для RandomForestClassifier
param_grid = {
    'n_estimators': [100, 200, 300],  # Количество деревьев
    'max_depth': [10, 20, 30],  # Максимальная глубина деревьев
    'min_samples_split': [2, 5, 10],  # Минимальное количество выборок для разбиения
    'min_samples_leaf': [1, 2, 4],  # Минимальное количество выборок на листе
    'class_weight': ['balanced']  # Балансировка классов
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    scoring='accuracy',
    cv=3,  # Кросс-валидация
    verbose=2,
    n_jobs=-1  # Использование всех процессоров
)

# Обучение модели
grid_search.fit(X_train, y_train)

# Лучшая модель и её параметры
best_model = grid_search.best_estimator_
print(f"Лучшие параметры: {grid_search.best_params_}")

# Оценка модели
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Сохранение модели и стандартизатора
joblib.dump(best_model, "recommendation_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Модель и стандартизатор сохранены.")
