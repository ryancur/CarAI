from django.contrib import admin

from .models import SuggestedPrice, AutomobileVO

@admin.register(SuggestedPrice)
class SuggestedPriceAdmin(admin.ModelAdmin):
    pass

@admin.register(AutomobileVO)
class AutomobileVOAdmin(admin.ModelAdmin):
    pass
