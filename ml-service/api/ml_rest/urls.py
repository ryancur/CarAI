from django.urls import path

from .views import api_predict_price

urlpatterns = [
    path(
        "automobile/predict-price/",
        api_predict_price,
        name="api_predict_price",
    ),
]
