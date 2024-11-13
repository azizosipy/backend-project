from django.shortcuts import render
from rest_framework import generics
from .serializers import DocumentSerializer
from .models import Documents
# Create your views here.


class DocumentAPIView(generics.ListCreateAPIView):
    queryset = Documents.objects.all()
    serializer_class = DocumentSerializer
