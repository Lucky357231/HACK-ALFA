import joblib
import pandas as pd
import json
from django.shortcuts import render

# Загрузка обученной модели
model = joblib.load('./recommendation_model.pkl')

# Словарь для отображения методов
method_mapping = {
    0: "SMS",
    1: "PayControl",
    2: "КЭП на токене",
    3: "КЭП в приложении"
}

# Загрузка данных из clients.json
def load_client_data(client_id=None):
    with open('clients.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
    if client_id:
        # Ищем данные конкретного клиента
        client = next((c for c in data if c["client_id"] == client_id), None)
        if not client:
            raise ValueError(f"Клиент с ID {client_id} не найден")
        return client
    return data

# Форма для ввода данных клиента
def index(request):
    if request.method == "POST":
        # Получаем client_id из формы
        client_id = request.POST.get("client_id")
        if not client_id:
            return render(request, "recommendation_app/index.html", {
                "error": "Пожалуйста, укажите client_id"
            })

        try:
            # Загружаем данные клиента из файла clients.json
            client_data = load_client_data(client_id)
        except ValueError as e:
            return render(request, "recommendation_app/index.html", {
                "error": str(e)
            })

        # Преобразуем данные для предсказания
        df = pd.DataFrame([{
            "segment": client_data["segment"],
            "role": client_data["role"],
            "organizations_count": client_data["organizations_count"],
            "current_method": client_data["current_method"],
            "claims": client_data["claims"],
            "signatures_mobile_common": client_data["signatures"]["common"]["mobile"],
            "signatures_web_common": client_data["signatures"]["common"]["web"],
            "signatures_mobile_special": client_data["signatures"]["special"]["mobile"],
            "signatures_web_special": client_data["signatures"]["special"]["web"],
            "available_methods_count": len(client_data["available_methods"])
        }])

        # Преобразуем категориальные данные в числовые
        df["segment"] = df["segment"].astype("category").cat.codes
        df["role"] = df["role"].astype("category").cat.codes
        df["current_method"] = df["current_method"].astype("category").cat.codes

        # Предсказание
        prediction = model.predict(df)[0]
        predicted_method = method_mapping[prediction]
        
        request.session['client_data'] = client_data
        request.session['predicted_method'] = predicted_method
        
        # Отправляем результат в шаблон
        return render(request, "recommendation_app/result.html", {
            "client_data": client_data,
            "predicted_method": predicted_method
        })

    # Если метод GET, показываем пустую форму
    return render(request, "recommendation_app/index.html")

def settings(request):
    # Получаем данные из сессии
    client_data = request.session.get('client_data', None)
    predicted_method = request.session.get('predicted_method', None)

    return render(request, "recommendation_app/settings.html", {
        "client_data": client_data,
        "predicted_method": predicted_method
    })
