from django.urls import path, include
from chat import views

urlpatterns = [
    path('ai/', views.AIchatViews, name='ai-chat')
]
