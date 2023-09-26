from django.urls import path

from .views import api_predict_price, api_suggested_prices

urlpatterns = [
    path(
        "automobile/predict-price/",
        api_predict_price,
        name="api_predict_price",
    ),
    path(
        "automobile/suggested-prices/",
        api_suggested_prices,
        name="api_suggested_prices",
    ),
]
