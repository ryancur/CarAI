from django.db import models


class AutomobileVO(models.Model):
    vin = models.CharField(unique=True, max_length=17)


class SuggestedPrice(models.Model):
    automobile = models.ForeignKey(
        AutomobileVO,
        related_name="sale",
        on_delete=models.CASCADE,
    )
    suggested_price = models.DecimalField(max_digits=12, decimal_places=2)
    created_at = models.DateTimeField(auto_now_add=True)
