from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('auth/', include('users.urls')),
    path('healthcare/', include('healthcare.urls')),    
    path('chat/', include('chat.urls')),
]
