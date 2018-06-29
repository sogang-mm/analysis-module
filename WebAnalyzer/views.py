# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.shortcuts import render

# Create your views here.
from WebAnalyzer.models import *
from WebAnalyzer.serializers import *
from rest_framework import viewsets, generics


class ImageViewSet(viewsets.ModelViewSet):
    queryset = ImageModel.objects.all()
    serializer_class = ImageSerializer

    def get_queryset(self):
        queryset = self.queryset
        queryset = queryset.order_by('-token')

        token = self.request.query_params.get('token', None)
        if token is not None:
            queryset = queryset.filter(token=token)

        return queryset
