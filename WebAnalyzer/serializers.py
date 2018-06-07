from rest_framework import serializers
from WebAnalyzer.models import *


class ResultPositionSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = ResultPositionModel
        fields = ('x', 'y', 'w', 'h')


class ResultLabelSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = ResultLabelModel
        fields = ('description', 'score')


class ResultSerializer(serializers.HyperlinkedModelSerializer):
    position = ResultPositionSerializer(read_only=True)
    label = ResultLabelSerializer(many=True, read_only=True)

    class Meta:
        model = ResultModel
        fields = ('position', 'label')


class ImageSerializer(serializers.HyperlinkedModelSerializer):
    result = ResultSerializer(many=True, read_only=True)

    class Meta:
        model = ImageModel
        fields = ('image', 'token', 'uploaded_date', 'updated_date', 'result')
        read_only_fields = ('token', 'uploaded_date', 'updated_date', 'result')
