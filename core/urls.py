from django.urls import path
from .views import DocumentAPIView


urlpatterns = [
    path('upload/',DocumentAPIView.as_view(),name='upload')
]