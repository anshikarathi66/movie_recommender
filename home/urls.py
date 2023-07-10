from home import views
from django.contrib import admin
from django.urls import path,include

urlpatterns = [
    path('', views.index, name='home'),
    path('external', views.external),
    path('recommender', views.recommender)
    ]