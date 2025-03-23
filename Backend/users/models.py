import random
from django.contrib.auth.models import AbstractUser
from django.db import models

class CustomUser(AbstractUser):
    is_active = models.BooleanField(default=False)  # User is inactive until OTP is verified

