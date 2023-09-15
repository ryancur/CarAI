from django.urls import path

# from .views import
from .views import (
    api_technician,
    api_technicians,
    api_appointment,
    api_appointments,
    api_cancel_appointment,
    api_finish_appointment,
)

urlpatterns = [
    path("technicians/", api_technicians, name="api_technicians"),
    path("technicians/<int:pk>/", api_technician, name="api_technician"),
    path("appointments/", api_appointments, name="api_appointments"),
    path("appointments/<int:pk>/", api_appointment, name="api_appointment"),
    path(
        "appointments/<int:pk>/cancel/",
        api_cancel_appointment,
        name="api_cancel_appointment",
    ),
    path(
        "appointments/<int:pk>/finish/",
        api_finish_appointment,
        name="api_finish_appointment",
    ),
]
