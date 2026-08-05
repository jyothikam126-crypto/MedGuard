from django import forms
from .models import DailyHealthLog

class HealthLogForm(forms.ModelForm):
    class Meta:
        model = DailyHealthLog
        fields = [
            'sleep_hours', 'exercise_minutes', 'water_ml', 'food_habit',
            'weight', 'blood_pressure_sys', 'blood_pressure_dia', 'mood', 'stress_level'
        ]
        widgets = {
            'sleep_hours': forms.NumberInput(attrs={'class': 'form-input', 'placeholder': 'Hours of sleep'}),
            'exercise_minutes': forms.NumberInput(attrs={'class': 'form-input', 'placeholder': 'Minutes of exercise'}),
            'water_ml': forms.NumberInput(attrs={'class': 'form-input', 'placeholder': 'Water in ml'}),
            'food_habit': forms.Select(attrs={'class': 'form-input'}),
            'weight': forms.NumberInput(attrs={'class': 'form-input', 'placeholder': 'Weight in kg'}),
            'blood_pressure_sys': forms.NumberInput(attrs={'class': 'form-input', 'placeholder': 'Systolic'}),
            'blood_pressure_dia': forms.NumberInput(attrs={'class': 'form-input', 'placeholder': 'Diastolic'}),
            'mood': forms.Select(attrs={'class': 'form-input'}, choices=[
                ('happy', 'Happy'),
                ('neutral', 'Neutral'),
                ('sad', 'Sad'),
                ('stressed', 'Stressed'),
                ('tired', 'Tired')
            ]),
            'stress_level': forms.NumberInput(attrs={'class': 'form-input', 'min': 0, 'max': 10, 'placeholder': '0-10'}),
        }
