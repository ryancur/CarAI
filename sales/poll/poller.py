import django
import os
import sys
import time
import json
import requests

sys.path.append("")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "sales_project.settings")
django.setup()


from sales_rest.models import AutomobileVO


def get_automobiles():
    response = requests.get("http://carai-inventory-api-1:8000/api/automobiles/")
    content = json.loads(response.content)
    for automobile in content["autos"]:
        AutomobileVO.objects.update_or_create(
            vin=automobile["vin"],
            model_name=automobile["model"]["name"],
            manufacturer_name=automobile["model"]["manufacturer"]["name"],
            defaults={
                "vin": automobile["vin"],
            },
        )


def poll(repeat=True):
    while True:
        print("Sales poller polling for data")
        try:
            get_automobiles()
            print("SALES poller success")
        except Exception as e:
            print(e, file=sys.stderr)

        if not repeat:
            break

        time.sleep(10)


if __name__ == "__main__":
    poll()
