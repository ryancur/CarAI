from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
import json

from .price_predict import predict_price
from .models import SuggestedPrice
from .encoders import SuggestedPriceEncoder


@require_http_methods(["POST"])
def api_predict_price(request):
    try:
        content = json.loads(request.body)
        auto_price_prediction = predict_price(input_data=content)
        content["suggested_price"] = auto_price_prediction
        return JsonResponse({"auto": content})
    except:
        response = JsonResponse({"message": "Could not predict price"})
        response.status_code = 400
        return response


@require_http_methods(["GET"])
def api_suggested_prices(request):
    if request.method == "GET":
        suggested_prices = SuggestedPrice.objects.all()
        return JsonResponse(
            {"suggested_prices": suggested_prices},
            encoder=SuggestedPriceEncoder,
        )
