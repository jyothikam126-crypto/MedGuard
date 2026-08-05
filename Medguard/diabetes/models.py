from django.db import models
from django.contrib.auth.models import User

# Create your models here.

class DiabetesPrediction(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    
    Age = models.IntegerField()
    Gender = models.CharField(max_length=10)
    symptoms = models.TextField()
    Polyuria = models.BooleanField(default=False)
    Polydipsia = models.BooleanField(default=False)
    sudden_weight_loss = models.BooleanField(default=False)
    weakness = models.BooleanField(default=False)
    Polyphagia = models.BooleanField(default=False) 
    Genital_thrush = models.BooleanField(default=False)
    visual_blurring = models.BooleanField(default=False)
    Itching = models.BooleanField(default=False)
    Irritability = models.BooleanField(default=False)
    delayed_healing = models.BooleanField(default=False)
    partial_paresis = models.BooleanField(default=False)
    muscle_stiffness = models.BooleanField(default=False)
    Alopecia = models.BooleanField(default=False)
    Obesity = models.BooleanField(default=False)    
    prediction = models.BooleanField(default=False)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.user.username} - Diabetes - {self.created_at.strftime('%Y-%m-%d')}"


class HeartDiseasePrediction(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    
    age = models.IntegerField()
    gender = models.CharField(max_length=10)
    chest_pain_type = models.IntegerField(default=0)
    blood_pressure = models.IntegerField(default=120)
    cholesterol = models.IntegerField(default=200)
    fasting_blood_sugar = models.BooleanField(default=False)
    ecg = models.IntegerField(default=0)
    max_heart_rate = models.IntegerField(default=150)
    exercise_angina = models.BooleanField(default=False)
    st_depression = models.FloatField(default=0)
    prediction = models.BooleanField(default=False)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.user.username} - Heart Disease - {self.created_at.strftime('%Y-%m-%d')}"


class SymptomCheck(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    
    symptoms = models.TextField()
    body_area = models.CharField(max_length=50, default='general')
    severity = models.CharField(max_length=20, default='mild')
    duration = models.CharField(max_length=20, default='hours')
    notes = models.TextField(blank=True)
    result = models.TextField(blank=True)  # Stores JSON result
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.user.username} - Symptom Check - {self.created_at.strftime('%Y-%m-%d')}"


class DocGPTConversation(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    
    user_message = models.TextField()
    ai_response = models.TextField()
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.user.username} - DocGPT - {self.created_at.strftime('%Y-%m-%d %H:%M')}"
class DailyHealthLog(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    date = models.DateField(auto_now_add=True)
    
    # Lifestyle Data
    sleep_hours = models.FloatField(default=0)
    exercise_minutes = models.IntegerField(default=0)
    water_ml = models.IntegerField(default=0)
    food_habit = models.CharField(max_length=50, choices=[('healthy', 'Healthy'), ('mixed', 'Mixed'), ('junk', 'Junk')], default='mixed')
    
    # Basic Health Data
    weight = models.FloatField(null=True, blank=True)
    blood_pressure_sys = models.IntegerField(null=True, blank=True)
    blood_pressure_dia = models.IntegerField(null=True, blank=True)
    mood = models.CharField(max_length=20, default='neutral')
    stress_level = models.IntegerField(default=0) # 0-10
    
    # Calculated Score
    health_score = models.IntegerField(default=0)
    
    # AI Suggestions
    suggestions = models.TextField(blank=True)
    
    class Meta:
        ordering = ['-date']
        unique_together = ['user', 'date']
    
    def __str__(self):
        return f"{self.user.username} - {self.date}"
