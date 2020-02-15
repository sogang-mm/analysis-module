from rest_framework import serializers
from WebAnalyzer.models import *


class ImageSerializer(serializers.HyperlinkedModelSerializer):

    class Meta:
        model = ImageModel
        fields = ('image', 'file', 'token', 'uploaded_date', 'updated_date', 'results')
        read_only_fields = ('token', 'uploaded_date', 'updated_date', 'results')

