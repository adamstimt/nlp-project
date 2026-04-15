from django.urls import path
from . import views

app_name = 'nlp_app'

urlpatterns = [
    path('', views.index, name='index'),
    
    path('analyze/', views.analyze, name='analyze'),
    
    path('stats/', views.stats_view, name='stats'),
]