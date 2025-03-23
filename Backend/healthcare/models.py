from django.db import models
from django.contrib.auth import get_user_model
import uuid

User = get_user_model()

class GeneralHealthForm(models.Model):

    GENDER_CHOICES = [
        ('Male', 'Male'),
        ('Female', 'Female'),
        ('Other', 'Other'),
        ('Prefer not to say', 'Prefer Not to Say'),
    ]

    LANGUAGE_CHOICES = [
        ('English', 'English'),
        ('Hindi', 'Hindi'),
        ('Other', 'Other'),
    ]

    SYMPTOM_SEVERITY_CHOICES = [
        ('Mild', 'Mild'),
        ('Moderate', 'Moderate'),
        ('Severe', 'Severe'),
        ('Critical', 'Critical'),
    ]

    SYMPTOM_DURATION_CHOICES = [
        ('Less than a day', 'Less than a day'),
        ('1-3 days', '1-3 days'),
        ('More than a week', 'More than a week'),
        ('Chronic', 'Chronic'),
    ]

    PREGNANCY_STATUS_CHOICES = [
        ('Not Pregnant', 'Not Pregnant'),
        ('Pregnant', 'Pregnant'),
        ('Not Applicable', 'Not Applicable'),
    ]

    id = models.UUIDField(editable=False, primary_key=True, default=uuid.uuid4)
    user = models.OneToOneField(User, on_delete=models.CASCADE)  
    name = models.CharField(max_length=100)  
    age = models.PositiveIntegerField()  
    gender = models.CharField(max_length=20, choices=GENDER_CHOICES)  
    contact_details = models.CharField(max_length=15)  

    chronic_conditions = models.TextField(blank=True, null=True)  
    past_surgeries = models.TextField(blank=True, null=True)  
    allergies = models.TextField(blank=True, null=True)  
    medications = models.TextField(blank=True, null=True)  

    symptoms = models.TextField(blank=True, null=True)  
    symptom_severity = models.CharField(max_length=20, choices=SYMPTOM_SEVERITY_CHOICES, blank=True, null=True)  
    symptom_duration = models.CharField(max_length=20, choices=SYMPTOM_DURATION_CHOICES, blank=True, null=True)  

    mental_health_stress = models.BooleanField(default=False)  
    mental_health_anxiety = models.BooleanField(default=False)  
    mental_health_depression = models.BooleanField(default=False)  

    vaccination_history = models.TextField(blank=True, null=True)  
    accessibility_needs = models.TextField(blank=True, null=True)  

    pregnancy_status = models.CharField(max_length=20, choices=PREGNANCY_STATUS_CHOICES, default='Not Applicable')

    emergency_contact = models.JSONField(default=dict)

    health_insurance_provider = models.CharField(max_length=100, blank=True, null=True)  
    health_insurance_policy = models.CharField(max_length=100, blank=True, null=True)  

    preferred_language = models.CharField(max_length=50, choices=LANGUAGE_CHOICES, blank=True, null=True)  
    research_participation = models.BooleanField(default=False)  

    created_at = models.DateTimeField(auto_now_add=True)  
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):  
        return f"{self.name}'s Health Form"
