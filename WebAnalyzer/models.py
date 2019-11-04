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
    image = models.TextField()
    token = models.AutoField(primary_key=True)
    uploaded_date = models.DateTimeField(auto_now_add=True)
    updated_date = models.DateTimeField(auto_now=True)

    def save(self, *args, **kwargs):
        super(ImageModel, self).save(*args, **kwargs)

        if DEBUG:
            task_get = ast.literal_eval(str(analyzer_by_path(self.image)))
        else:
            task_get = ast.literal_eval(str(analyzer_by_path.delay(self.image).get()))

        for result in task_get:
            self.result.create(values=result)
        #super(ImageModel, self).save()


class ResultModel(models.Model):
    result_model = models.ForeignKey(ImageModel, related_name='result', on_delete=models.CASCADE)
    values = models.TextField()

    def save(self, *args, **kwargs):
        if not (isinstance(self.values[0], list) or isinstance(self.values[0], tuple)):
            raise exceptions.ValidationError("Module return values(0) Error. Please contact the administrator")
        if not (isinstance(self.values[1], dict)):
            raise exceptions.ValidationError("Module return values(1) Error. Please contact the administrator")

        super(ResultModel, self).save(*args, **kwargs)
        x, y, w, h = self.values[0]
        ResultPositionModel.objects.create(result_detail_model=self, x=x, y=y, w=w, h=h)
        for item in self.values[1].items():
            self.label.create(description=item[0], score=float(item[1]))
        super(ResultModel, self).save()


class ResultPositionModel(models.Model):
    result_detail_model = models.OneToOneField(ResultModel, related_name='position', on_delete=models.CASCADE)
    x = models.FloatField(null=True, unique=False)
    y = models.FloatField(null=True, unique=False)
    w = models.FloatField(null=True, unique=False)
    h = models.FloatField(null=True, unique=False)

    class Meta:
        ordering = ['x', 'y', 'w', 'h']


class ResultLabelModel(models.Model):
    result_detail_model = models.ForeignKey(ResultModel, related_name='label', on_delete=models.CASCADE)
    description = models.TextField(null=True, unique=False)
    score = models.FloatField(null=True, unique=False)

    class Meta:
        ordering = ['-score']
