from django.db import models

# Эта модель будет использоваться для структурирования входных данных
# Даже если данные не сохраняются в базе данных, мы можем использовать её как структуру данных

class Client(models.Model):
    # Данные клиента, которые будут переданы в модель
    client_id = models.CharField(max_length=255, unique=True)
    organization_id = models.CharField(max_length=255)
    segment = models.CharField(max_length=50)
    role = models.CharField(max_length=50)
    organizations_count = models.PositiveIntegerField()
    current_method = models.CharField(max_length=50)
    claims = models.PositiveIntegerField()

    # Подписи (common и special) — это просто поля, которые хранят статистику
    signatures_mobile_common = models.PositiveIntegerField()
    signatures_web_common = models.PositiveIntegerField()
    signatures_mobile_special = models.PositiveIntegerField()
    signatures_web_special = models.PositiveIntegerField()

    # Доступные методы — это список методов, которые использует клиент
    available_methods = models.JSONField()

    def __str__(self):
        return f"Client {self.client_id} ({self.segment})"
