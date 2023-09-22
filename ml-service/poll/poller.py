import django
import os
import sys
import time
import json
import requests

sys.path.append("")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ml_project.settings")
django.setup()


from ml_rest.models import AutomobileVO


def get_automobiles():
    response = requests.get("http://carai-inventory-api-1:8000/api/automobiles/")
    content = json.loads(response.content)
    for automobile in content["autos"]:
        # TODO: put price prediction here
        AutomobileVO.objects.update_or_create(
            vin=automobile["vin"],
            defaults={
                "vin": automobile["vin"],
            },
        )


def poll(repeat=True):
    while True:
        print("ML poller polling for data")
        try:
            get_automobiles()
            print("ML poller success")
        except Exception as e:
            print(e, file=sys.stderr)

        if not repeat:
            break

        time.sleep(10)


if __name__ == "__main__":
    poll()
