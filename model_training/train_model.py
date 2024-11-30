from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import pandas as pd
import json
import joblib

# Загрузка данных
with open("clients.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Подготовка данных
def preprocess_data(data):
    df = pd.json_normalize(data)

    # Кодируем категориальные признаки
    df["segment"] = df["segment"].astype("category").cat.codes
    df["role"] = df["role"].astype("category").cat.codes
    df["current_method"] = df["current_method"].astype("category").cat.codes

    # Бизнес-логика: определение приоритетных рекомендаций с помощью словаря
    priority_methods = {
        "Малый бизнес": "PayControl",
        "Средний бизнес": "КЭП на токене",
        "Крупный бизнес": "КЭП в приложении"
    }

    # Удаляем SMS из доступных методов
    df["available_methods"] = df["available_methods"].apply(
        lambda x: [m for m in x if m != "SMS"]
    )

    # Применяем бизнес-логику для определения целевой переменной
    df["target"] = df["segment"].apply(lambda x: priority_methods.get(x, None))

    # Удаляем строки с пустыми значениями в target (если бизнес-логика не сработала)
    df = df[df["target"].notna()]

    # Проверка, что после фильтрации остались данные
    if df.shape[0] == 0:
        raise ValueError("Нет данных для обучения после применения бизнес-логики")

    # Кодируем целевую переменную
    df["target"] = df["target"].astype("category").cat.codes

    # Новые признаки
    df["mobile_web_ratio_common"] = df["signatures.common.mobile"] / (df["signatures.common.web"] + 1)
    df["total_signatures"] = df["signatures.common.mobile"] + df["signatures.common.web"]
    df["available_methods_count"] = df["available_methods"].apply(len)
    df["is_mobile_user"] = df["mobile_app"].astype(int)
    df["role_segment_interaction"] = df["role"] * df["segment"]

    return df

# Предобработка данных
df = preprocess_data(data)

# Выбор признаков
X = df[[
    "segment", "role", "organizations_count", "current_method", "claims",
    "mobile_web_ratio_common", "total_signatures",
    "available_methods_count", "is_mobile_user", "role_segment_interaction"
]]
y = df["target"]

# Проверка, что в данных есть классы для балансировки
if y.value_counts().min() == 0:
    raise ValueError("Нельзя выполнить балансировку классов: один из классов отсутствует")

# Балансировка классов
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)

# Стандартизация признаков
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_balanced)

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_balanced, test_size=0.2, random_state=42)

# Подбор гиперпараметров для RandomForestClassifier
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'class_weight': ['balanced', None]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,
    verbose=2,
    n_jobs=-1
)

# Обучение модели
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# Оценка модели
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Сохранение модели и масштабатора
joblib.dump(best_model, "recommendation_model_rf_with_logic.pkl")
joblib.dump(scaler, "scaler_rf_with_logic.pkl")
