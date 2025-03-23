from rest_framework import generics, permissions
from .models import GeneralHealthForm
from rest_framework.response import Response
from rest_framework.views import APIView
from .serializers import GeneralHealthFormSerializer

# Create and submit a new form
class GeneralHealthFormCreateView(generics.CreateAPIView):
    serializer_class = GeneralHealthFormSerializer
    permission_classes = [permissions.IsAuthenticated]  # Only logged-in users

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)  # Assign form to the logged-in user

# List all forms submitted by the authenticated user
class GeneralHealthFormListView(generics.ListAPIView):
    serializer_class = GeneralHealthFormSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return GeneralHealthForm.objects.filter(user=self.request.user)

# Retrieve a specific form
class GeneralHealthFormDetailView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        try:
            form = GeneralHealthForm.objects.get(user=request.user)  # Get form for logged-in user
            serializer = GeneralHealthFormSerializer(form)
            return Response(serializer.data)
        except GeneralHealthForm.DoesNotExist:
            return Response({"error": "Form not found"}, status=404)

# Update a specific form
class GeneralHealthFormUpdateView(generics.UpdateAPIView):
    serializer_class = GeneralHealthFormSerializer
    permission_classes = [permissions.IsAuthenticated]
    
    def get_object(self):
        return GeneralHealthForm.objects.filter(user=self.request.user).last()
