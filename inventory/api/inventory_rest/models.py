from django.db import models
from django.urls import reverse


class SuggestedValueVO(models.Model):
    suggested_value = models.DecimalField(max_digits=12, decimal_places=2)


class Manufacturer(models.Model):
    name = models.CharField(max_length=100, unique=True)

    def get_api_url(self):
        return reverse("api_manufacturer", kwargs={"pk": self.id})


class VehicleModel(models.Model):
    name = models.CharField(max_length=100)
    picture_url = models.URLField()

    manufacturer = models.ForeignKey(
        Manufacturer,
        related_name="models",
        on_delete=models.CASCADE,
    )

    def get_api_url(self):
        return reverse("api_vehicle_model", kwargs={"pk": self.id})


class Automobile(models.Model):
    color = models.CharField(max_length=50)
    year = models.PositiveSmallIntegerField()
    vin = models.CharField(max_length=17, unique=True)
    sold = models.BooleanField(default=False)

    suggested_value = models.ForeignKey(
        SuggestedValueVO,
        related_name="automobiles",
        on_delete=models.CASCADE,
        null=True,
    )

    model = models.ForeignKey(
        VehicleModel,
        related_name="automobiles",
        on_delete=models.CASCADE,
    )

    def get_api_url(self):
        return reverse("api_automobile", kwargs={"vin": self.vin})
