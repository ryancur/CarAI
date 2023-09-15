from django.db import models


class AutomobileVO(models.Model):
    vin = models.CharField(unique=True, max_length=17)
    model_name = models.CharField(max_length=100, null=True)
    manufacturer_name = models.CharField(max_length=100, null=True)


class Salesperson(models.Model):
    first_name = models.CharField(max_length=200)
    last_name = models.CharField(max_length=200)
    employee_id = models.BigAutoField(primary_key=True)


class Customer(models.Model):
    first_name = models.CharField(max_length=200)
    last_name = models.CharField(max_length=200)
    address = models.CharField(max_length=300)
    phone_number = models.PositiveBigIntegerField()


class Sale(models.Model):
    automobile = models.ForeignKey(
        AutomobileVO,
        related_name="sale",
        on_delete=models.CASCADE,
    )
    salesperson = models.ForeignKey(
        Salesperson,
        related_name="sale",
        on_delete=models.PROTECT,
    )
    customer = models.ForeignKey(
        Customer,
        related_name="sale",
        on_delete=models.PROTECT,
    )
    price = models.DecimalField(max_digits=8, decimal_places=2)
