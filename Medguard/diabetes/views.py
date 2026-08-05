from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
from django.conf import settings
from django.utils import timezone
from django.db import IntegrityError
import os
import json
import logging
import joblib
import numpy as np
import requests
from .models import DiabetesPrediction, HeartDiseasePrediction, SymptomCheck, DocGPTConversation, DailyHealthLog
from .forms import HealthLogForm
from accounts.models import UserProfile, DoctorProfile, Appointment, ChatMessage
from accounts.views import patient_required

# Create your views here.

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'diabetes', 'model_files')

# Load Diabetes Model
DIABETES_MODEL_PATH = os.path.join(MODEL_DIR, 'clinical_diabetes_model.pkl')
diabetes_model = joblib.load(DIABETES_MODEL_PATH)

# Load Heart Disease Model and Scaler
HEART_MODEL_PATH = os.path.join(MODEL_DIR, 'heart_disease_model.pkl')
HEART_SCALER_PATH = os.path.join(MODEL_DIR, 'heart_disease_scaler.pkl')
heart_model = joblib.load(HEART_MODEL_PATH)
heart_scaler = joblib.load(HEART_SCALER_PATH)

# Symptom checker condition labels
CONDITION_LABELS = [
    'Common Cold', 'Flu', 'COVID-19', 'Migraine', 'Food Poisoning',
    'Anxiety', 'Allergies', 'Stomach Flu', 'Tension Headache'
]

# Gemini API Configuration
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
logger = logging.getLogger(__name__)

symptoms = [
    ('Polyuria', 'Excessive Urine'),
    ('Polydipsia', 'Excessive Thirst'),
    ('sudden_weight_loss', 'Sudden Weight Loss'),
    ('weakness', 'Weakness (tiredness)'),
    ('Polyphagia', 'Excessive Hunger'), 
    ('Genital_thrush', 'Genital Infection'),
    ('visual_blurring', 'Blurry Vision'),
    ('Itching', 'Itchy Skin'),
    ('Irritability', 'Irritability'),
    ('delayed_healing', 'Slow Wound Healing'),
    ('partial_paresis', 'Muscle Weakness'),
    ('muscle_stiffness', 'Muscle Stiffness'),
    ('Alopecia', 'Hair Loss'),
    ('Obesity', 'Obesity'),
]

# General symptoms for symptom checker
general_symptoms = [
    ('fever', 'Fever'),
    ('cough', 'Cough'),
    ('headache', 'Headache'),
    ('fatigue', 'Fatigue'),
    ('body_ache', 'Body Aches'),
    ('nausea', 'Nausea'),
    ('vomiting', 'Vomiting'),
    ('diarrhea', 'Diarrhea'),
    ('chest_pain', 'Chest Pain'),
    ('shortness_breath', 'Shortness of Breath'),
    ('dizziness', 'Dizziness'),
    ('sore_throat', 'Sore Throat'),
    ('runny_nose', 'Runny Nose'),
    ('joint_pain', 'Joint Pain'),
    ('rash', 'Skin Rash'),
    ('abdominal_pain', 'Abdominal Pain'),
    ('loss_appetite', 'Loss of Appetite'),
    ('chills', 'Chills'),
    ('sweating', 'Excessive Sweating'),
    ('insomnia', 'Insomnia'),
]

# mapping the symptoms to their respective field names
field_names = {
    'Polyuria': 'Polyuria',
    'Polydipsia': 'Polydipsia',
    'sudden_weight_loss': 'sudden weight loss',
    'weakness': 'weakness',
    'Polyphagia': 'Polyphagia',
    'Genital_thrush': 'Genital thrush',
    'visual_blurring': 'visual blurring',
    'Itching': 'Itching',
    'Irritability': 'Irritability',
    'delayed_healing': 'delayed healing',
    'partial_paresis': 'partial paresis',
    'muscle_stiffness': 'muscle stiffness',
    'Alopecia': 'Alopecia',
    'Obesity': 'Obesity',
}


@login_required
def dashboard(request):
    """Main dashboard view with stats and overview."""
    user = request.user
    user_profile = UserProfile.objects.get(user=user)

    if user_profile.role == 'doctor':
        # Redirect doctors to their dashboard
        return redirect('doctor_dashboard')

    # Patient dashboard
    # Get prediction counts
    diabetes_count = DiabetesPrediction.objects.filter(user=user).count()
    heart_count = HeartDiseasePrediction.objects.filter(user=user).count()
    symptom_count = SymptomCheck.objects.filter(user=user).count()
    total_predictions = diabetes_count + heart_count + symptom_count

    # Get chat session count
    chat_sessions = DocGPTConversation.objects.filter(user=user).count()

    # Get daily log count
    daily_logs = DailyHealthLog.objects.filter(user=user).count()

    # Calculate days active (days since first prediction)
    days_active = 1

    # Get recent health score
    recent_log = DailyHealthLog.objects.filter(user=user).order_by('-date').first()
    health_score = recent_log.health_score if recent_log else 0

    context = {
        'total_predictions': total_predictions,
        'diabetes_count': diabetes_count,
        'heart_count': heart_count,
        'symptom_count': symptom_count,
        'chat_sessions': chat_sessions,
        'daily_logs': daily_logs,
        'days_active': days_active,
        'health_score': health_score,
        'is_patient': True,
    }

    return render(request, 'dashboard.html', context)


@login_required
@patient_required
def home(request):
    """Diabetes prediction view."""
    if request.method == 'POST':
        Age = int(request.POST.get('age'))
        gender = request.POST.get('Gender')

        Gender = 1 if gender == 'Male' else 0

        symptoms_value = []
        symptoms_dict = {}
        for field, _ in symptoms:
            value = 1 if request.POST.get(field) else 0
            symptoms_dict[field] = bool(value)

            model_field = field_names.get(field, field)
            symptoms_value.append(value)

        final_input = [Age, Gender] + symptoms_value

        prediction = diabetes_model.predict([final_input])[0]
        # Get probability percentage
        try:
            probability = diabetes_model.predict_proba([final_input])[0]
            risk_percentage = int(round(probability[1] * 100))
        except Exception:
            risk_percentage = 100 if prediction == 1 else 0
        
        is_high_risk = prediction == 1
        
        # Generate reasons based on risk factors
        reasons = []
        if is_high_risk:
            if Age > 45:
                reasons.append("Age above 45 increases diabetes risk")
            if symptoms_dict.get('Polyuria'):
                reasons.append("Excessive urination (Polyuria) symptom present")
            if symptoms_dict.get('Polydipsia'):
                reasons.append("Excessive thirst (Polydipsia) symptom present")
            if symptoms_dict.get('sudden_weight_loss'):
                reasons.append("Sudden weight loss detected")
            if symptoms_dict.get('weakness'):
                reasons.append("General weakness/fatigue present")
            if symptoms_dict.get('Polyphagia'):
                reasons.append("Excessive hunger (Polyphagia) symptom present")
            if symptoms_dict.get('visual_blurring'):
                reasons.append("Visual blurring symptom present")
            if symptoms_dict.get('delayed_healing'):
                reasons.append("Slow wound healing detected")
            if symptoms_dict.get('Obesity'):
                reasons.append("Obesity is a major risk factor")
            if not reasons:
                reasons.append("Based on overall symptom assessment")
        else:
            reasons.append("No significant diabetes symptoms detected")
            if Age < 45:
                reasons.append("Age is within lower risk range")
            reasons.append("Overall symptom profile indicates normal function")
        
        result = {
            'risk': 'high' if is_high_risk else 'low',
            'score': risk_percentage,
            'message': "The patient <strong>has diabetes</strong>." if prediction == 1 else "The patient <strong>does not have diabetes</strong>.",
            'reasons': reasons
        }
        prediction_result = bool(prediction)

        # Save the prediction to the database
        DiabetesPrediction.objects.create(
            user=request.user, 
            Age=Age,
            Gender=Gender,
            symptoms=", ".join([label for field, label in symptoms if symptoms_dict[field]]),
            Polyuria=symptoms_dict['Polyuria'],
            Polydipsia=symptoms_dict['Polydipsia'],
            sudden_weight_loss=symptoms_dict['sudden_weight_loss'],
            weakness=symptoms_dict['weakness'],
            Polyphagia=symptoms_dict['Polyphagia'],
            Genital_thrush=symptoms_dict['Genital_thrush'],
            visual_blurring=symptoms_dict['visual_blurring'],
            Itching=symptoms_dict['Itching'],
            Irritability=symptoms_dict['Irritability'],
            delayed_healing=symptoms_dict['delayed_healing'],
            partial_paresis=symptoms_dict['partial_paresis'],
            muscle_stiffness=symptoms_dict['muscle_stiffness'],
            Alopecia=symptoms_dict['Alopecia'],
            Obesity=symptoms_dict['Obesity'],
            prediction=prediction_result
        )

        context = {
            'result': result,
            'age': Age,
            'gender': gender,
            'symptoms': symptoms,
            'selected_symptoms': [label for field, label in symptoms if symptoms_dict[field]],
        }

        return render(request, 'home.html', context)

    return render(request, 'home.html', {'symptoms': symptoms})


@login_required
@patient_required
def heart_disease(request):
    """Heart disease prediction view."""
    result = None

    if request.method == 'POST':
        age = int(request.POST.get('age', 0))
        gender = request.POST.get('gender', '')
        chest_pain = int(request.POST.get('chest_pain', 0))
        blood_pressure = int(request.POST.get('blood_pressure', 120))
        cholesterol = int(request.POST.get('cholesterol', 200))
        fasting_blood_sugar = int(request.POST.get('fasting_blood_sugar', 0))
        ecg = int(request.POST.get('ecg', 0))
        max_heart_rate = int(request.POST.get('max_heart_rate', 150))
        exercise_angina = int(request.POST.get('exercise_angina', 0))
        st_depression = float(request.POST.get('st_depression', 0))
        sex = 1 if gender.lower() == 'male' else 0

        # Predict using trained heart disease ML model.
        model_input = np.array([[
            age,
            sex,
            chest_pain,
            blood_pressure,
            cholesterol,
            fasting_blood_sugar,
            ecg,
            max_heart_rate,
            exercise_angina,
            st_depression
        ]], dtype=float)
        model_input_scaled = heart_scaler.transform(model_input)
        prediction = int(heart_model.predict(model_input_scaled)[0])
        probability = (
            float(heart_model.predict_proba(model_input_scaled)[0][1])
            if hasattr(heart_model, 'predict_proba')
            else float(prediction)
        )
        is_high_risk = prediction == 1

        # Save prediction
        HeartDiseasePrediction.objects.create(
            user=request.user,
            age=age,
            gender=gender,
            chest_pain_type=chest_pain,
            blood_pressure=blood_pressure,
            cholesterol=cholesterol,
            fasting_blood_sugar=bool(fasting_blood_sugar),
            ecg=ecg,
            max_heart_rate=max_heart_rate,
            exercise_angina=bool(exercise_angina),
            st_depression=st_depression,
            prediction=is_high_risk
        )

        # Generate reasons based on risk factors
        reasons = []
        if is_high_risk:
            if age > 55:
                reasons.append("Age above 55 increases heart disease risk")
            if chest_pain == 0:
                reasons.append("Typical angina chest pain detected")
            elif chest_pain == 1:
                reasons.append("Atypical angina symptoms present")
            if blood_pressure > 140:
                reasons.append("High blood pressure (hypertension)")
            elif blood_pressure > 120:
                reasons.append("Elevated blood pressure")
            if cholesterol > 240:
                reasons.append("High cholesterol levels")
            elif cholesterol > 200:
                reasons.append("Borderline high cholesterol")
            if fasting_blood_sugar == 1:
                reasons.append("Elevated fasting blood sugar (diabetes risk)")
            if max_heart_rate < 100:
                reasons.append("Low maximum heart rate capacity")
            if exercise_angina == 1:
                reasons.append("Exercise-induced angina present")
            if st_depression > 2:
                reasons.append("Significant ST depression (heart strain)")
            elif st_depression > 1:
                reasons.append("Moderate ST depression")
            if not reasons:
                reasons.append("Based on overall cardiovascular assessment")
        else:
            reasons.append("Heart health parameters within normal range")
            if age < 50:
                reasons.append("Age is within lower risk range")
            if blood_pressure <= 120:
                reasons.append("Blood pressure is normal")
            if cholesterol < 200:
                reasons.append("Cholesterol levels are healthy")
            if max_heart_rate >= 140:
                reasons.append("Good cardiovascular fitness")
            reasons.append("Overall cardiovascular indicators appear normal")
        
        result = {
            'risk': 'high' if is_high_risk else 'low',
            'score': int(round(probability * 100)),
            'reasons': reasons
        }

    return render(request, 'heart_disease.html', {'result': result})


@login_required
def symptom_checker(request):
    """AI-powered symptom checker view."""
    result = None
    
    if request.method == 'POST':
        selected_symptoms = request.POST.getlist('symptoms')
        body_area = request.POST.get('body_area', 'general')
        severity = request.POST.get('severity', 'mild')
        duration = request.POST.get('duration', 'hours')
        notes = request.POST.get('notes', '')
        
        # Use Gemini API for symptom analysis
        symptom_names = [dict(general_symptoms).get(s, s) for s in selected_symptoms]
        
        prompt = f"""As a medical AI assistant, analyze these symptoms and provide possible conditions.

Symptoms: {', '.join(symptom_names)}
Affected area: {body_area}
Severity: {severity}
Duration: {duration}
Additional notes: {notes}

Provide:
1. Top 3 possible conditions with likelihood percentages
2. Brief recommendations

Format your response as JSON with this structure:
{{
    "conditions": [
        {{"name": "Condition Name", "probability": 75}},
        ...
    ],
    "recommendations": "Your recommendations here"
}}

Be conservative with probabilities and always recommend consulting a doctor."""

        try:
            response = call_gemini_api(prompt)
            
            # Try to parse JSON from response
            try:
                # Find JSON in response
                start = response.find('{')
                end = response.rfind('}') + 1
                if start != -1 and end != 0:
                    json_str = response[start:end]
                    result = json.loads(json_str)
                else:
                    result = {
                        'conditions': [
                            {'name': 'General Assessment Needed', 'probability': 50}
                        ],
                        'recommendations': response
                    }
            except json.JSONDecodeError:
                result = {
                    'conditions': [
                        {'name': 'General Assessment Needed', 'probability': 50}
                    ],
                    'recommendations': response
                }
                
        except Exception as e:
            result = {
                'conditions': [
                    {'name': 'Analysis Error', 'probability': 0}
                ],
                'recommendations': f'Unable to analyze symptoms. Please try again or consult a healthcare provider.'
            }
        
        # Save symptom check
        SymptomCheck.objects.create(
            user=request.user,
            symptoms=', '.join(symptom_names),
            body_area=body_area,
            severity=severity,
            duration=duration,
            notes=notes,
            result=json.dumps(result) if result else ''
        )
    
    return render(request, 'symptom_checker.html', {
        'symptoms': general_symptoms,
        'result': result
    })


@login_required
def docgpt(request):
    """DocGPT chat interface view."""
    return render(request, 'docgpt.html')


@login_required
def api_docgpt(request):
    """API endpoint for DocGPT chat."""
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'Method not allowed'}, status=405)
    
    try:
        data = json.loads(request.body)
        user_message = data.get('message', '')
        history = data.get('history', [])
        
        if not user_message:
            return JsonResponse({'success': False, 'error': 'No message provided'})
        
        # Create system prompt for DocGPT - Advanced Health Assistant
        system_prompt = """You are DocGPT, an advanced AI-powered health assistant integrated into the MedGuard Health Platform. You are designed to provide comprehensive, accurate, and empathetic health guidance to users.

🏥 YOUR IDENTITY:
- Name: DocGPT
- Role: AI Health Advisor & Medical Information Assistant
- Platform: MedGuard Health Dashboard

💡 YOUR CAPABILITIES:
1. **Symptom Analysis**: Help users understand their symptoms and when to seek medical care
2. **Health Education**: Explain medical conditions, treatments, and procedures in simple terms
3. **Medication Information**: Provide general information about common medications (dosages, side effects, interactions)
4. **Wellness Advice**: Offer tips on nutrition, exercise, sleep, stress management, and preventive care
5. **First Aid Guidance**: Provide basic first aid instructions for common situations
6. **Mental Health Support**: Offer supportive conversation and coping strategies (not therapy)

📋 GUIDELINES:
- Always be empathetic, warm, and supportive in your responses
- Use clear, simple language avoiding excessive medical jargon
- Structure responses with bullet points, headings, or numbered lists for clarity
- Provide evidence-based information from reliable medical sources
- Always recommend consulting a healthcare professional for:
  • Persistent or severe symptoms
  • Medication changes or new medications
  • Mental health crises
  • Any emergency situations
- Never provide definitive diagnoses - only educational information
- Include appropriate disclaimers when discussing treatments or medications
- Respect user privacy and maintain a professional tone

⚠️ IMPORTANT DISCLAIMERS:
- You are an AI assistant, NOT a licensed medical professional
- Your responses are for educational purposes only
- In emergencies, always advise calling emergency services immediately
- Do not replace professional medical advice, diagnosis, or treatment

🎯 RESPONSE STYLE:
- Begin with acknowledgment of the user's concern
- Provide clear, actionable information
- End with encouragement and next steps
- Use emojis sparingly to maintain a friendly yet professional tone

Remember: Your goal is to empower users with knowledge while ensuring they seek appropriate professional care when needed."""

        # Build conversation
        messages = [{"role": "user", "parts": [{"text": system_prompt}]}]
        
        for msg in history[-10:]:  # Keep last 10 messages for context
            role = "user" if msg['role'] == 'user' else "model"
            messages.append({"role": role, "parts": [{"text": msg['content']}]})
        
        messages.append({"role": "user", "parts": [{"text": user_message}]})
        
        # Call Gemini API
        response_text = call_gemini_api_chat(messages)
        
        # Save conversation
        DocGPTConversation.objects.create(
            user=request.user,
            user_message=user_message,
            ai_response=response_text
        )
        
        return JsonResponse({
            'success': True,
            'response': response_text
        })
        
    except (json.JSONDecodeError, KeyError, TypeError):
        return JsonResponse({
            'success': False,
            'error': 'Invalid request data.'
        }, status=400)
    except requests.RequestException:
        logger.exception("DocGPT provider request failed")
        return JsonResponse({
            'success': False,
            'error': 'The AI service is temporarily unavailable. Please try again later.'
        }, status=503)
    except Exception:
        logger.exception("Unexpected DocGPT error")
        return JsonResponse({
            'success': False,
            'error': 'Unable to process your request at this time.'
        }, status=500)


def call_gemini_api(prompt):
    """Call Gemini API for single prompt."""
    if not GEMINI_API_KEY:
        raise requests.RequestException("Gemini API is not configured.")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    
    headers = {
        "Content-Type": "application/json"
    }
    
    data = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 1024
        }
    }
    
    response = requests.post(url, headers=headers, json=data, timeout=20)
    response.raise_for_status()
    
    result = response.json()
    return result['candidates'][0]['content']['parts'][0]['text']


def call_gemini_api_chat(messages):
    """Call Gemini API for chat conversations."""
    if not GEMINI_API_KEY:
        raise requests.RequestException("Gemini API is not configured.")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    
    headers = {
        "Content-Type": "application/json"
    }
    
    # Convert messages format for Gemini
    contents = []
    for msg in messages:
        contents.append({
            "role": msg["role"],
            "parts": msg["parts"]
        })
    
    data = {
        "contents": contents,
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 1024
        }
    }
    
    response = requests.post(url, headers=headers, json=data, timeout=20)
    response.raise_for_status()
    
    result = response.json()
    return result['candidates'][0]['content']['parts'][0]['text']


@login_required
@patient_required
def prediction_history(request):
    """View user's prediction history."""
    user = request.user
    
    # Get all predictions
    diabetes_predictions = DiabetesPrediction.objects.filter(user=user).order_by('-id')
    heart_predictions = HeartDiseasePrediction.objects.filter(user=user).order_by('-id')
    symptom_checks = SymptomCheck.objects.filter(user=user).order_by('-id')
    
    # Combine and format predictions
    predictions = []
    
    for p in diabetes_predictions:
        predictions.append({
            'type': 'diabetes',
            'created_at': p.id,  # Using ID as proxy for ordering
            'details': f'Age: {p.Age}, Symptoms: {p.symptoms[:50]}...' if len(p.symptoms) > 50 else f'Age: {p.Age}, Symptoms: {p.symptoms}',
            'positive': p.prediction
        })
    
    for p in heart_predictions:
        predictions.append({
            'type': 'heart',
            'created_at': p.id,
            'details': f'Age: {p.age}, BP: {p.blood_pressure}, Cholesterol: {p.cholesterol}',
            'positive': p.prediction
        })
    
    for p in symptom_checks:
        predictions.append({
            'type': 'symptom',
            'created_at': p.id,
            'details': f'Area: {p.body_area}, Symptoms: {p.symptoms[:50]}...' if len(p.symptoms) > 50 else f'Area: {p.body_area}, Symptoms: {p.symptoms}',
            'positive': False  # Symptom checks don't have positive/negative
        })
    
    # Sort by created_at (id) descending
    predictions.sort(key=lambda x: x['created_at'], reverse=True)
    
    context = {
        'predictions': predictions[:20],  # Show last 20
        'total_predictions': len(predictions),
        'diabetes_count': diabetes_predictions.count(),
        'heart_count': heart_predictions.count(),
        'symptom_count': symptom_checks.count(),
    }
    
    return render(request, 'prediction_history.html', context)


@login_required
def daily_log(request):
    """Daily health logging view."""
    today = timezone.localdate()
    log = DailyHealthLog.objects.filter(user=request.user, date=today).first()
    
    if request.method == 'POST':
        form = HealthLogForm(request.POST, instance=log)
        if form.is_valid():
            cleaned = form.cleaned_data
            
            # Calculate Health Score (Simple algorithm)
            score = 0
            
            # Sleep (ideal 7-9 hours)
            if 7 <= cleaned['sleep_hours'] <= 9: score += 20
            elif 6 <= cleaned['sleep_hours'] < 7 or 9 < cleaned['sleep_hours'] <= 10: score += 15
            else: score += 5
            
            # Exercise (ideal 30+ mins)
            if cleaned['exercise_minutes'] >= 45: score += 20
            elif cleaned['exercise_minutes'] >= 30: score += 15
            elif cleaned['exercise_minutes'] >= 15: score += 10
            else: score += 0
            
            # Water (ideal 2500+ ml)
            if cleaned['water_ml'] >= 3000: score += 20
            elif cleaned['water_ml'] >= 2000: score += 15
            elif cleaned['water_ml'] >= 1000: score += 5
            
            # Food
            if cleaned['food_habit'] == 'healthy': score += 20
            elif cleaned['food_habit'] == 'mixed': score += 10
            else: score += 0
            
            # Stress
            score += (10 - cleaned['stress_level']) * 2
            
            # Smart AI Insights (Simple rule-based for now, can be Gemini)
            suggestions = []
            if cleaned['sleep_hours'] < 7: suggestions.append("Aim for at least 7 hours of sleep.")
            if cleaned['exercise_minutes'] < 30: suggestions.append("Try to include 30 mins of daily activity.")
            if cleaned['water_ml'] < 2000: suggestions.append("Stay hydrated! Aim for 2-3 liters of water.")
            if cleaned['food_habit'] == 'junk': suggestions.append("Incorporate more whole foods into your diet.")
            if cleaned['stress_level'] > 7: suggestions.append("Consider meditation or a walk to reduce stress.")

            update_values = {
                'sleep_hours': cleaned['sleep_hours'],
                'exercise_minutes': cleaned['exercise_minutes'],
                'water_ml': cleaned['water_ml'],
                'food_habit': cleaned['food_habit'],
                'weight': cleaned['weight'],
                'blood_pressure_sys': cleaned['blood_pressure_sys'],
                'blood_pressure_dia': cleaned['blood_pressure_dia'],
                'mood': cleaned['mood'],
                'stress_level': cleaned['stress_level'],
                'health_score': score,
                'suggestions': " | ".join(suggestions),
            }

            try:
                DailyHealthLog.objects.update_or_create(
                    user=request.user,
                    date=today,
                    defaults=update_values,
                )
            except IntegrityError:
                # Concurrent create fallback: update the row that already won.
                DailyHealthLog.objects.filter(
                    user=request.user,
                    date=today
                ).update(**update_values)

            return redirect('health_analytics')
    else:
        form = HealthLogForm(instance=log)
        
    return render(request, 'daily_log.html', {'form': form})


@login_required
@patient_required
def health_analytics(request):
    """Health analytics dashboard with Power BI-style charts."""
    logs = list(DailyHealthLog.objects.filter(user=request.user).order_by('date')[:30])  # Last 30 days for better trends

    # Prepare chart data
    dates = [log.date.strftime('%m/%d') for log in logs]
    sleep_data = [log.sleep_hours for log in logs]
    exercise_data = [log.exercise_minutes for log in logs]
    water_data = [log.water_ml for log in logs]
    scores = [log.health_score for log in logs]

    # Weight and mood data
    weight_data = [log.weight for log in logs if log.weight]
    mood_data = []
    mood_dates = []
    for log in logs:
        if log.mood:
            # Convert mood strings to numbers for charting
            mood_value = 0  # default
            if log.mood == 'sad':
                mood_value = 0
            elif log.mood == 'neutral':
                mood_value = 1
            elif log.mood == 'happy':
                mood_value = 2
            mood_data.append(mood_value)
            mood_dates.append(log.date.strftime('%m/%d'))

    # Food habit distribution
    food_habits = {}
    for log in logs:
        habit = log.food_habit
        food_habits[habit] = food_habits.get(habit, 0) + 1

    # Convert food habits to chart data
    food_labels = []
    food_counts = []
    food_colors = []
    for habit, count in food_habits.items():
        if habit == 'healthy':
            food_labels.append('Healthy')
            food_colors.append('#10b981')  # Green
        elif habit == 'mixed':
            food_labels.append('Mixed')
            food_colors.append('#f59e0b')  # Yellow
        else:  # junk
            food_labels.append('Junk')
            food_colors.append('#ef4444')  # Red
        food_counts.append(count)

    current_log = logs[-1] if logs else None

    # Handle suggestions
    suggestions_list = []
    if current_log and current_log.suggestions:
        suggestions_list = [s.strip() for s in current_log.suggestions.split('|') if s.strip()]

    avg_score = sum(scores) / len(scores) if scores else 0

    context = {
        'dates': json.dumps(dates),
        'sleep_data': json.dumps(sleep_data),
        'exercise_data': json.dumps(exercise_data),
        'water_data': json.dumps(water_data),
        'scores': json.dumps(scores),
        'weight_data': json.dumps(weight_data),
        'mood_data': json.dumps(mood_data),
        'mood_dates': json.dumps(mood_dates),
        'food_labels': json.dumps(food_labels),
        'food_counts': json.dumps(food_counts),
        'food_colors': json.dumps(food_colors),
        'current_log': current_log,
        'suggestions': suggestions_list,
        'avg_score': int(avg_score),
    }

    return render(request, 'health_analytics.html', context)


import csv
from django.http import HttpResponse


@login_required
def export_health_data(request):
    """Export health predictions and symptoms as CSV for medical reports."""
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename="health_report_{request.user.username}.csv"'

    writer = csv.writer(response)
    writer.writerow(['Date', 'Prediction Type', 'Symptoms/Details', 'Medical Data', 'Prediction Result'])

    # Get all predictions ordered by date
    diabetes_predictions = DiabetesPrediction.objects.filter(user=request.user).order_by('-created_at')
    heart_predictions = HeartDiseasePrediction.objects.filter(user=request.user).order_by('-created_at')
    symptom_checks = SymptomCheck.objects.filter(user=request.user).order_by('-created_at')

    # Export Diabetes Predictions
    for pred in diabetes_predictions:
        writer.writerow([
            pred.created_at.strftime('%Y-%m-%d %H:%M'),
            'Diabetes Prediction',
            pred.symptoms,
            f'Age: {pred.Age}, Gender: {pred.Gender}',
            'Positive' if pred.prediction else 'Negative'
        ])

    # Export Heart Disease Predictions
    for pred in heart_predictions:
        medical_data = f'Age: {pred.age}, Gender: {pred.gender}, BP: {pred.blood_pressure}, Cholesterol: {pred.cholesterol}, Max HR: {pred.max_heart_rate}'
        writer.writerow([
            pred.created_at.strftime('%Y-%m-%d %H:%M'),
            'Heart Disease Prediction',
            f'Chest Pain Type: {pred.chest_pain_type}, ECG: {pred.ecg}, Exercise Angina: {pred.exercise_angina}, ST Depression: {pred.st_depression}',
            medical_data,
            'High Risk' if pred.prediction else 'Low Risk'
        ])

    # Export Symptom Checks
    for check in symptom_checks:
        writer.writerow([
            check.created_at.strftime('%Y-%m-%d %H:%M'),
            'Symptom Check',
            check.symptoms,
            f'Body Area: {check.body_area}, Severity: {check.severity}, Duration: {check.duration}',
            'Assessment Completed'
        ])

    return response


@login_required
def privacy_policy(request):
    """Privacy policy view."""
    return render(request, 'privacy_policy.html')
