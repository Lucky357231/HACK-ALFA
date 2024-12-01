import joblib
import pandas as pd
import json
from django.shortcuts import render

# Загрузка обученной модели и скейлера
model = joblib.load('./recommendation_model.pkl')
scaler = joblib.load('./scaler.pkl')

# Словарь для отображения методов
method_mapping = {
    0: "PayControl",
    1: "КЭП на токене",
    2: "КЭП в приложении"
}

# Загрузка данных клиента
def load_client_data(client_id=None):
    with open('clients.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
    if client_id:
        client = next((c for c in data if c["client_id"] == client_id), None)
        if not client:
            raise ValueError(f"Клиент с ID {client_id} не найден")
        return client
    return data

# Предобработка данных для модели
def preprocess_client_data(client_data):
    try:
        df = pd.DataFrame([{
            "segment": client_data["segment"],
            "role": client_data["role"],
            "organizations_count": client_data["organizations_count"],
            "current_method": client_data["current_method"],
            "claims": client_data["claims"],
            "mobile_web_ratio_common": client_data["signatures"]["common"]["mobile"] / (client_data["signatures"]["common"]["web"] + 1),
            "total_signatures": client_data["signatures"]["common"]["mobile"] + client_data["signatures"]["common"]["web"],
            "available_methods_count": len(client_data["available_methods"]),
            "is_mobile_user": int(client_data["mobile_app"]),
            "role_segment_interaction": hash(client_data["role"]) * hash(client_data["segment"])  # Уникальное взаимодействие
        }])

        # Кодирование категориальных признаков
        df["segment"] = df["segment"].astype("category").cat.codes
        df["role"] = df["role"].astype("category").cat.codes
        df["current_method"] = df["current_method"].astype("category").cat.codes

        return df

    except KeyError as e:
        raise ValueError(f"Отсутствует ключ в данных клиента: {e}")

# Основной обработчик
def index(request):
    if request.method == "POST":
        client_id = request.POST.get("client_id")
        if not client_id:
            return render(request, "recommendation_app/index.html", {
                "error": "Пожалуйста, укажите client_id"
            })

        try:
            client_data = load_client_data(client_id)
            df = preprocess_client_data(client_data)
        except ValueError as e:
            return render(request, "recommendation_app/index.html", {
                "error": str(e)
            })

        # Масштабирование данных
        try:
            df_scaled = scaler.transform(df)
        except ValueError as e:
            return render(request, "recommendation_app/index.html", {
                "error": f"Ошибка масштабирования данных: {e}"
            })

        # Предсказание метода
        prediction = model.predict(df_scaled)[0]
        predicted_method = method_mapping.get(prediction, "Неизвестный метод")

        # Сохраняем данные в сессии
        request.session['client_data'] = client_data
        request.session['predicted_method'] = predicted_method

        return render(request, "recommendation_app/result.html", {
            "client_data": client_data,
            "predicted_method": predicted_method
        })

    return render(request, "recommendation_app/index.html")

# Настройки сессии
def settings(request):
    client_data = request.session.get('client_data', None)
    predicted_method = request.session.get('predicted_method', None)

    return render(request, "recommendation_app/settings.html", {
        "client_data": client_data,
        "predicted_method": predicted_method
    })

def sms_metrics(request):
    # Логика получения метрик метода SMS
    with open("clients.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.json_normalize(data)

    # Фильтрация данных: только те, у которых метод подписания - SMS
    sms_claims_df = df[df["current_method"] == "SMS"]

    # Подсчет общего количества claims для метода SMS
    sms_claims = sms_claims_df["claims"].sum()

    # Подсчет общего количества клиентов, использующих SMS
    sms_clients_count = sms_claims_df.shape[0]

    # Подсчет клиентов, использующих другие методы (кроме SMS)
    other_methods_df = df[df["current_method"] != "SMS"]
    other_methods_clients_count = other_methods_df.shape[0]

    # Подготовка данных для графиков
    clients_count_data = {
        'sms_clients_count': sms_clients_count,
        'other_methods_clients_count': other_methods_clients_count
    }

    claims_data = {
        'sms_claims': sms_claims
    }

    return render(request, "recommendation_app/sms_metrics.html", {
        "clients_count_data": clients_count_data,
        "claims_data": claims_data
    })

