# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models

# Create your models here.
from rest_framework import exceptions
from AnalysisModule.config import DEBUG
from WebAnalyzer.tasks import analyzer_by_path
from WebAnalyzer.utils import filename
from django_mysql.models import JSONField
import ast


class ImageModel(models.Model):
    image = models.TextField()
    file = models.FileField(upload_to=filename.default)
    token = models.AutoField(primary_key=True)
    uploaded_date = models.DateTimeField(auto_now_add=True)
    updated_date = models.DateTimeField(auto_now=True)
    results = JSONField(null=True)

    def save(self, *args, **kwargs):
        super(ImageModel, self).save(*args, **kwargs)

        if DEBUG:
            task_get = ast.literal_eval(str(analyzer_by_path(self.image, self.file.path)))
        else:
            task_get = ast.literal_eval(str(analyzer_by_path.delay(self.image, self.file.path).get()))

        self.results = [{"module_result": task_get}]
        super(ImageModel, self).save()
