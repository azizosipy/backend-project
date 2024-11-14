from django.urls import path
from .views import process_pdf


urlpatterns = [
    path('analyze/', process_pdf, name='process_pdf'),
]