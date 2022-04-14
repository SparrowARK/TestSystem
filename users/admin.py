from django.contrib import admin
from django.contrib.admin.models import LogEntry

from .models import Present, Time

# Register your models here.
admin.site.register(Time)
admin.site.register(Present)
LogEntry.objects.all().delete()
