import django
import os
import sys
import time
import json
import requests

sys.path.append("")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "inventory_project.settings")
django.setup()


from inventory_rest.models import SuggestedPriceVO


def get_automobiles():
    response = requests.get("http://project-beta-inventory-api-1:8000/api/automobiles/")
    content = json.loads(response.content)
    for price in content["price"]:
        SuggestedPriceVO.objects.update_or_create(
            suggested_price=price["price"],
            defaults={
                "suggested_price": price["price"],
            },
        )


def poll(repeat=True):
    while True:
        print("Inventory poller polling for data")
        try:
            get_automobiles()
        except Exception as e:
            print(e, file=sys.stderr)

        if not repeat:
            break

        time.sleep(60)


if __name__ == "__main__":
    poll()
