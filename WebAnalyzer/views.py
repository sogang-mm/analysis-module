# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.shortcuts import render

# Create your views here.
from AnalysisModule.config import VIEWSET_NUMBER
from WebAnalyzer.models import *
from WebAnalyzer.serializers import *
from rest_framework import viewsets, generics


class ImageViewSet(viewsets.ModelViewSet):
    queryset = ImageModel.objects.all()
    serializer_class = ImageSerializer

    def get_queryset(self):
        return self.queryset.order_by('-token')[:VIEWSET_NUMBER]
