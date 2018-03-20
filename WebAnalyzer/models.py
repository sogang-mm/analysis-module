# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models

# Create your models here.
from Modules.manager import analyzer_module

class ImageModel(models.Model):
    image = models.ImageField()
    token = models.AutoField(primary_key=True)
    uploaded_date = models.DateTimeField(auto_now_add=True)
    updated_date = models.DateTimeField(auto_now=True)
    result = models.TextField(null=True)

    def save(self, *args, **kwargs):
        super(ImageModel, self).save(*args, **kwargs)
        self.result = analyzer_module.get_result_by_path(self.image.path)
        super(ImageModel, self).save()