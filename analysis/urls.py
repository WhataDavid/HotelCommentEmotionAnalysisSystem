from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name="home"),
    path('user/', views.user, name="user"),
    path('analysis/', views.analysis, name="analysis"),
    path('file/', views.file, name="file"),
    path('crawl/', views.crawl, name="crawl")

]
