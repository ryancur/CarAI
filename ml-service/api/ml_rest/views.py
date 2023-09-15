from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
import json

from .price_predict import predict_price


@require_http_methods(["POST"])
def api_predict_price(request):
    try:
        content = json.loads(request.body)
        auto_price_prediction = predict_price(input_data=content)
        content["suggested_price"] = auto_price_prediction
        return JsonResponse({"auto": content})
    except:
        response = JsonResponse(
            {"message": "Could not predict price"}
        )
        response.status_code = 400
        return response
