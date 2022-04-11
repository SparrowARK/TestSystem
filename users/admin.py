from django.contrib import admin
from .models import Time,Present
from django.contrib.admin.models import LogEntry

# Register your models here.
admin.site.register(Time)
admin.site.register(Present)
LogEntry.objects.all().delete()