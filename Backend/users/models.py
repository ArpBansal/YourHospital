from django.contrib.auth.models import AbstractUser
from django.db import models

class CustomUser(AbstractUser):
    is_active = models.BooleanField(default=False)  # User is inactive until OTP is verified


# is_active is present in default User, but just to create a custom user model, i have used it here
# i have used custom user so that, there will be no problem adding anything else to the user later on 
