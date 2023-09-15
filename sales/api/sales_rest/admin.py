from django.contrib import admin

from .models import AutomobileVO


@admin.register(AutomobileVO)
class AutomobileVOAdmin(admin.ModelAdmin):
    pass
