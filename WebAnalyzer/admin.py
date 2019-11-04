# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.contrib import admin
from .models import ImageModel, ResultModel, ResultPositionModel, ResultLabelModel

# Register your models here.
admin.site.register(ImageModel)
admin.site.register(ResultModel)
admin.site.register(ResultPositionModel)
admin.site.register(ResultLabelModel)