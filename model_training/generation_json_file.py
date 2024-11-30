import json
import random


# Функция для генерации данных клиентов
def generate_clients_data(num_records=1000):
    segments = ["Крупный бизнес", "Средний бизнес", "Малый бизнес"]
    roles = ["Сотрудник", "Менеджер", "Руководитель", "ЕИО"]
    methods = ["КЭП на токене", "PayControl", "SMS", "КЭП в приложении"]

    data = []

    for i in range(num_records):
        segment = random.choice(segments)
        role = random.choice(roles)
        current_method = random.choice(methods)
        available_methods = random.sample(methods, k=random.randint(1, len(methods)))
        claims = random.randint(0, 20)
        organizations_count = random.randint(1, 50)
        mobile_app = random.choice([True, False])

        signatures_common_mobile = random.randint(0, 50)
        signatures_common_web = random.randint(0, 50)
        signatures_special_mobile = random.randint(0, 20)
        signatures_special_web = random.randint(0, 20)

        data.append({
            "client_id": f"client{i + 1:04d}",
            "organization_id": f"org{i + 1:04d}",
            "segment": segment,
            "role": role,
            "organizations_count": organizations_count,
            "current_method": current_method,
            "mobile_app": mobile_app,
            "signatures": {
                "common": {
                    "mobile": signatures_common_mobile,
                    "web": signatures_common_web
                },
                "special": {
                    "mobile": signatures_special_mobile,
                    "web": signatures_special_web
                }
            },
            "available_methods": available_methods,
            "claims": claims
        })

    return data


# Генерация данных для 1000 клиентов
clients_data = generate_clients_data(1000)

# Сохранение данных в файл clients.json
with open("clients.json", "w", encoding="utf-8") as f:
    json.dump(clients_data, f, ensure_ascii=False, indent=4)

print("Данные клиентов успешно сгенерированы и сохранены в файл clients.json.")