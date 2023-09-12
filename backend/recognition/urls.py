from django.urls import path
from . import views

urlpatterns = [
    path('images-get/', views.getUserImages ),
    path('images-upload/', views.uploadUserImages )
]