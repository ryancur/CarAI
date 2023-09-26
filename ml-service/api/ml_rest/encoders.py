from common.json import ModelEncoder

from .models import SuggestedPrice


class SuggestedPriceEncoder(ModelEncoder):
    model = SuggestedPrice
    properties = ["id", "suggested_price"]

    def get_extra_data(self, o):
        return {
            "automobile": {
                "vin": o.automobile.vin,
            }
        }
