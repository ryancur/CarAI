from django.shortcuts import get_object_or_404
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
import json

from .encoders import (
    TechnicianEncoder,
    AppointmentEncoder,
)

from .models import Technician, Appointment


# technician views
@require_http_methods(["GET", "POST"])
def api_technicians(request):
    if request.method == "GET":
        technicians = Technician.objects.all()
        return JsonResponse(
            {"technicians": technicians},
            encoder=TechnicianEncoder,
        )
    else:
        try:
            content = json.loads(request.body)
            technician = Technician.objects.create(**content)
            return JsonResponse(
                technician,
                encoder=TechnicianEncoder,
                safe=False,
            )
        except:
            return JsonResponse({"message": "Employee ID already exists."}, status=400)


@require_http_methods(["GET", "DELETE"])
def api_technician(request, pk):
    if request.method == "DELETE":
        technician = get_object_or_404(Technician, id=pk)
        technician_id = technician.id
        technician.delete()
        message = f"Deleted technician: {technician.first_name} {technician.last_name} with id {technician_id}"
        return JsonResponse({"message": message})
    else:
        # GET response for individual technician id
        technician = get_object_or_404(Technician, id=pk)
        return JsonResponse(
            {"technician": technician}, encoder=TechnicianEncoder, safe=False
        )


# appointment views
@require_http_methods(["GET", "POST"])
def api_appointments(request):
    if request.method == "GET":
        appointments = Appointment.objects.all()
        return JsonResponse({"appointments": appointments}, encoder=AppointmentEncoder)
    else:
        content = json.loads(request.body)
        try:
            # get the technician object from the db
            tech_id = content["technician"]
            technician = Technician.objects.get(id=tech_id)
            # add the technician object to the content dict before creation
            content["technician"] = technician
        except Technician.DoesNotExist:
            return JsonResponse({"message": "Invalid technician id"}, status=400)
        appointment = Appointment.create(**content)
        return JsonResponse(
            appointment,
            encoder=AppointmentEncoder,
            safe=False,
        )


@require_http_methods(["GET", "DELETE"])
def api_appointment(request, pk):
    if request.method == "DELETE":
        appointment = get_object_or_404(Appointment, id=pk)
        appointment_id = appointment.id
        appointment.delete()
        message = f"Deleted appointment: {appointment_id} for VIN: {appointment.vin}"
        return JsonResponse({"message": message})
    else:
        # GET response for individual appointment id
        appointment = get_object_or_404(Appointment, id=pk)
        return JsonResponse(
            {"appointment": appointment},
            encoder=AppointmentEncoder,
            safe=False,
        )


@require_http_methods(["PUT"])
def api_cancel_appointment(request, pk):
    appointment = get_object_or_404(Appointment, id=pk)
    appointment.cancel()
    return JsonResponse(
        appointment,
        encoder=AppointmentEncoder,
        safe=False,
    )


@require_http_methods(["PUT"])
def api_finish_appointment(request, pk):
    appointment = get_object_or_404(Appointment, id=pk)
    appointment.finish()
    return JsonResponse(
        appointment,
        encoder=AppointmentEncoder,
        safe=False,
    )
