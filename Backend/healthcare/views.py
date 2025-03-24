from rest_framework import generics, permissions, status
from rest_framework.response import Response
from rest_framework.views import APIView
from .models import GeneralHealthForm
from .serializers import GeneralHealthFormSerializer
from django.http import Http404

# Create and submit a new form
class GeneralHealthFormCreateView(generics.CreateAPIView):
    serializer_class = GeneralHealthFormSerializer
    permission_classes = [permissions.IsAuthenticated]  # Only logged-in users

    def perform_create(self, serializer):
        try:
            serializer.save(user=self.request.user)  # Assign form to the logged-in user
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

# List all forms submitted by the authenticated user
class GeneralHealthFormListView(generics.ListAPIView):
    serializer_class = GeneralHealthFormSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        try:
            return GeneralHealthForm.objects.filter(user=self.request.user)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# Retrieve a specific form
class GeneralHealthFormDetailView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        try:
            form = GeneralHealthForm.objects.get(user=request.user)  # Get form for logged-in user
            serializer = GeneralHealthFormSerializer(form)
            return Response(serializer.data, status=status.HTTP_200_OK)
        except GeneralHealthForm.DoesNotExist:
            return Response({"error": "Form not found"}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# Update a specific form
class GeneralHealthFormUpdateView(generics.UpdateAPIView):
    serializer_class = GeneralHealthFormSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_object(self):
        try:
            form = GeneralHealthForm.objects.filter(user=self.request.user).last()
            if not form:
                raise GeneralHealthForm.DoesNotExist  # Raise the correct exception
            return form
        except GeneralHealthForm.DoesNotExist:
            # Instead of returning Response, we raise a proper Django exception
            raise Http404("No form found to update")
        except Exception as e:
            # Handle unexpected errors properly
            from rest_framework.exceptions import APIException
            raise APIException(f"Unexpected error: {str(e)}")
