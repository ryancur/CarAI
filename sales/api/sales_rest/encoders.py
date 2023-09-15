from common.json import ModelEncoder
from .models import Salesperson, Sale, Customer


class SalesPersonListEncoder(ModelEncoder):
    model = Salesperson
    properties = [
        "first_name",
        "last_name",
        "employee_id",
    ]


class CustomerListEncoder(ModelEncoder):
    model = Customer
    properties = [
        "id",
        "first_name",
        "last_name",
        "address",
        "phone_number",
    ]


class SaleListEncoder(ModelEncoder):
    model = Sale
    properties = [
        "id",
    ]

    def get_extra_data(self, o):
        return {
            "automobile": {
                "vin": o.automobile.vin,
                "model_name": o.automobile.model_name,
                "manufacturer_name": o.automobile.manufacturer_name,
            },
            "salesperson": {
                "first_name": o.salesperson.first_name,
                "last_name": o.salesperson.last_name,
                "employee_id": o.salesperson.employee_id,
            },
            "customer": {
                "first_name": o.customer.first_name,
                "last_name": o.customer.last_name,
                "address": o.customer.address,
                "phone_number": o.customer.phone_number,
            },
            "price": str(o.price),
        }
