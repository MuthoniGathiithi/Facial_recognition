from django.urls import path
from . import views

urlpatterns = [
    path('enroll/', views.enroll_view, name='enroll'),
]

