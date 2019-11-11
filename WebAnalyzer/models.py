# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models

# Create your models here.
from rest_framework import exceptions
from AnalysisModule.config import DEBUG
from WebAnalyzer.tasks import analyzer_by_path
from WebAnalyzer.utils import filename
import ast


class ImageModel(models.Model):
    file = models.FileField(upload_to=filename.default)
    token = models.AutoField(primary_key=True)
    uploaded_date = models.DateTimeField(auto_now_add=True)
    updated_date = models.DateTimeField(auto_now=True)
    result = models.TextField()

    def save(self, *args, **kwargs):
        super(ImageModel, self).save(*args, **kwargs)

        if DEBUG:
            task_get = str(analyzer_by_path(self.file.path))
        else:
            task_get = str(analyzer_by_path.delay(self.file.path).get())

        self.result = task_get
        super(ImageModel, self).save()
