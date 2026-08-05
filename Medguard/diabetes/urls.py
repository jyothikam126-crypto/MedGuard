from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('heart-disease/', views.heart_disease, name='heart_disease'),
    path('symptom-checker/', views.symptom_checker, name='symptom_checker'),
    path('docgpt/', views.docgpt, name='docgpt'),
    path('api/docgpt/', views.api_docgpt, name='api_docgpt'),
    path('history/', views.prediction_history, name='prediction_history'),
    path('daily-log/', views.daily_log, name='daily_log'),
    path('analytics/', views.health_analytics, name='health_analytics'),
    path('export-health-data/', views.export_health_data, name='export_health_data'),
    path('privacy/', views.privacy_policy, name='privacy_policy'),
]