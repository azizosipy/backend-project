from rest_framework import serializers
from .models import Documents


class DocumentSerializer(serializers.ModelSerializer):
    class Meta:
        fields = '__all__'
        model = Documents