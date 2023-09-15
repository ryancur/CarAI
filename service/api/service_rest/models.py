from django.db import models
from django.urls import reverse


class AutomobileVO(models.Model):
    vin = models.CharField(max_length=17, unique=True)


class Technician(models.Model):
    first_name = models.CharField(max_length=100)
    last_name = models.CharField(max_length=100)
    employee_id = models.CharField(max_length=25, unique=True)

    def __str__(self):
        return f"{self.first_name} {self.last_name}"

    class Meta:
        ordering = ("id",)


class Appointment(models.Model):
    @classmethod
    def create(cls, **kwargs):
        # set vip status
        inventory_vins = AutomobileVO.objects.all()
        vins = [inv_vin.vin for inv_vin in inventory_vins]
        service_vin = kwargs["vin"]
        if service_vin in vins:
            kwargs["vip_status"] = True
        else:
            kwargs["vip_status"] = False

        # create the appointment
        appointment = cls(**kwargs)
        appointment.save()
        return appointment

    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)
    date_time = models.DateTimeField()
    reason = models.CharField(max_length=200)
    vin = models.CharField(max_length=17)
    customer = models.CharField(max_length=200)
    vip_status = models.BooleanField(default=False)
    status = models.CharField(max_length=10, default="CREATED")
    technician = models.ForeignKey(
        Technician,
        related_name="appointments",
        on_delete=models.PROTECT,
    )

    def cancel(self):
        status = "CANCELED"
        self.status = status
        self.save()

    def finish(self):
        status = "FINISHED"
        self.status = status
        self.save()

    def get_api_url(self):
        return reverse("api_show_appointment", kwargs={"pk": self.pk})

    def __str__(self):
        return f"{self.vin}-{self.date_time}"

    class Meta:
        ordering = ("date_time",)
