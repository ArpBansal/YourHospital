# Generated by Django 5.1.4 on 2025-03-23 13:44

import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='GeneralHealthForm',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=100)),
                ('age', models.PositiveIntegerField()),
                ('gender', models.CharField(choices=[('Male', 'Male'), ('Female', 'Female'), ('Other', 'Other'), ('Prefer not to say', 'Prefer Not to Say')], max_length=20)),
                ('contact_details', models.CharField(max_length=15)),
                ('chronic_conditions', models.TextField(blank=True, null=True)),
                ('past_surgeries', models.TextField(blank=True, null=True)),
                ('allergies', models.TextField(blank=True, null=True)),
                ('medications', models.TextField(blank=True, null=True)),
                ('symptoms', models.TextField(blank=True, null=True)),
                ('symptom_severity', models.CharField(blank=True, choices=[('Mild', 'Mild'), ('Moderate', 'Moderate'), ('Severe', 'Severe'), ('Critical', 'Critical')], max_length=20, null=True)),
                ('symptom_duration', models.CharField(blank=True, choices=[('Less than a day', 'Less than a day'), ('1-3 days', '1-3 days'), ('More than a week', 'More than a week'), ('Chronic', 'Chronic')], max_length=20, null=True)),
                ('mental_health_stress', models.BooleanField(default=False)),
                ('mental_health_anxiety', models.BooleanField(default=False)),
                ('mental_health_depression', models.BooleanField(default=False)),
                ('vaccination_history', models.TextField(blank=True, null=True)),
                ('accessibility_needs', models.TextField(blank=True, null=True)),
                ('pregnancy_status', models.CharField(choices=[('Not Pregnant', 'Not Pregnant'), ('Pregnant', 'Pregnant'), ('Not Applicable', 'Not Applicable')], default='Not Applicable', max_length=20)),
                ('emergency_contact', models.JSONField(default=dict)),
                ('health_insurance_provider', models.CharField(blank=True, max_length=100, null=True)),
                ('health_insurance_policy', models.CharField(blank=True, max_length=100, null=True)),
                ('preferred_language', models.CharField(blank=True, choices=[('English', 'English'), ('Hindi', 'Hindi'), ('Other', 'Other')], max_length=50, null=True)),
                ('research_participation', models.BooleanField(default=False)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('user', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
    ]
