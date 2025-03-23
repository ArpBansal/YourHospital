from django.contrib.auth.hashers import make_password
from django.shortcuts import get_object_or_404
from rest_framework.response import Response
from rest_framework import status, permissions
from rest_framework.views import APIView
from users.models import CustomUser
from rest_framework_simplejwt.views import TokenObtainPairView
from rest_framework_simplejwt.tokens import RefreshToken
from users.serializers import CustomUserRegistrationSerializer
from users.utils import generate_otp, store_otp, get_stored_otp, delete_otp, send_otp_email

class RegistrationView(APIView):
    def post(self, request):
        serializer = CustomUserRegistrationSerializer(data=request.data)
        if serializer.is_valid():
            email = serializer.validated_data['email']

            if CustomUser.objects.filter(email=email, is_active=True).exists():
                return Response({"error": "User with this email already exists."}, status=status.HTTP_400_BAD_REQUEST)

            # Generate and store OTP
            otp = generate_otp()
            store_otp(email, otp)

            # Send OTP email
            send_otp_email(email, otp)

            # Create the user but keep them inactive
            user = CustomUser.objects.create(
                username=serializer.validated_data['username'],
                email=email,
                password=make_password(serializer.validated_data['password']),
                is_active=False  # User remains inactive until OTP verification
            )

            return Response({"message": "OTP sent to your email. Verify to activate your account."}, status=status.HTTP_201_CREATED)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class VerifyOTPView(APIView):
    def post(self, request):
        email = request.data.get("email")
        otp_entered = request.data.get("otp")

        otp_stored = get_stored_otp(email)

        if not otp_stored:
            return Response({"error": "OTP expired or invalid. Please request a new one."}, status=status.HTTP_400_BAD_REQUEST)

        if otp_stored == otp_entered:
            try:
                user = CustomUser.objects.get(email=email, is_active=False)
                user.is_active = True
                user.save()

                delete_otp(email)

                return Response({"message": "Email verified! Your account is now active."}, status=status.HTTP_200_OK)

            except CustomUser.DoesNotExist:
                return Response({"error": "User not found or already active."}, status=status.HTTP_404_NOT_FOUND)

        return Response({"error": "Invalid OTP."}, status=status.HTTP_400_BAD_REQUEST)


class ResendOTPView(APIView):
    def post(self, request):
        email = request.data.get("email")

        try:
            user = CustomUser.objects.get(email=email, is_active=False)

            # Generate and store new OTP
            otp = generate_otp()
            store_otp(email, otp)

            # Send new OTP email
            send_otp_email(email, otp, subject="Your New OTP Code")

            return Response({"message": "New OTP sent to your email."}, status=status.HTTP_200_OK)

        except CustomUser.DoesNotExist:
            return Response({"error": "User not found or already verified."}, status=status.HTTP_404_NOT_FOUND)


class CustomTokenObtainPairView(TokenObtainPairView):
    def post(self, request, *args, **kwargs):
        response = super().post(request, *args, **kwargs)
        email = request.data.get('email')

        user = get_object_or_404(CustomUser, email=email)
        if not user.email_verified:
            return Response({"error": "Email not verified. Please verify your email first."}, status=status.HTTP_403_FORBIDDEN)

        return response



class LogoutView(APIView):
    permission_classes = [permissions.IsAuthenticated]
    def post(self, request):
        try:
            refresh_token = request.data['refresh']
            token = RefreshToken(refresh_token)
            token.blacklist()
            return Response({'message': "Successfully Logged Out !"}, status=status.HTTP_205_RESET_CONTENT)
        except Exception as e:
            return Response({'message': "Invalid Token"}, status=status.HTTP_400_BAD_REQUEST)


