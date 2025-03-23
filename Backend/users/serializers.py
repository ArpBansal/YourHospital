from rest_framework import serializers
from users.models import CustomUser
from django.contrib.auth.password_validation import validate_password
from users.utils import send_otp_email  # Import the send_otp_email function

class CustomUserRegistrationSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True, required=True, validators=[validate_password])
    confirm_password = serializers.CharField(write_only=True, required=True)

    class Meta:
        model = CustomUser
        fields = ('id', 'username', 'email', 'password', 'confirm_password')

    def validate(self, attrs):
        if attrs['password'] != attrs['confirm_password']:
            raise serializers.ValidationError({"password": "Passwords do not match."})
        return attrs

    def create(self, validated_data):
        validated_data.pop('confirm_password')  # Remove extra password field
        user = CustomUser.objects.create_user(**validated_data)
        user.is_active = False  # User cannot log in until email is verified
        user.save()

        send_otp_email(user)  # Send OTP after registration
        return user
