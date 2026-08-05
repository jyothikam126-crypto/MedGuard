from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth.models import User,auth
from django.contrib.auth.decorators import login_required
from django.db import models
from functools import wraps
from .models import UserProfile, DoctorProfile, Appointment, Prescription, ChatMessage, HealthReport
from diabetes.models import DiabetesPrediction, HeartDiseasePrediction, SymptomCheck, DailyHealthLog

def patient_required(view_func):
    """Decorator to ensure only patients can access the view."""
    @wraps(view_func)
    def _wrapped_view(request, *args, **kwargs):
        try:
            user_profile = UserProfile.objects.get(user=request.user, role='patient')
            return view_func(request, *args, **kwargs)
        except UserProfile.DoesNotExist:
            messages.error(request, "Access denied. This feature is for patients only.")
            return redirect('doctor_dashboard')
    return _wrapped_view

def doctor_required(view_func):
    """Decorator to ensure only doctors can access the view."""
    @wraps(view_func)
    def _wrapped_view(request, *args, **kwargs):
        try:
            user_profile = UserProfile.objects.get(user=request.user, role='doctor')
            return view_func(request, *args, **kwargs)
        except UserProfile.DoesNotExist:
            messages.error(request, "Access denied. This feature is for doctors only.")
            return redirect('dashboard')
    return _wrapped_view

# Create your views here.

def register(request):
    if request.method == "POST":
        first_name = request.POST.get("first_name")
        last_name = request.POST.get("last_name")
        username = request.POST.get("username")
        email = request.POST.get("email")
        password = request.POST.get("password")
        confirm_password = request.POST.get("confirm_password")
        role = request.POST.get("role", "patient")
        license_number = request.POST.get("license_number", "")
        specialization = request.POST.get("specialization", "")

        if password == confirm_password:
            if User.objects.filter(username=username).exists():
                messages.info(request, "Username already exists")
                return render(request, "register.html")
            elif User.objects.filter(email=email).exists():
                messages.info(request, "Email already exists")
                return render(request, "register.html")
            else:
                user = User.objects.create_user(username=username, password=password, email=email, first_name=first_name, last_name=last_name)
                user.save()

                # Create UserProfile
                user_profile = UserProfile.objects.create(user=user, role=role)

                # Create DoctorProfile if role is doctor
                if role == "doctor":
                    if not license_number or not specialization:
                        messages.error(request, "License number and specialization are required for doctors")
                        user.delete()  # Clean up
                        return render(request, "register.html")
                    # Automatically verify doctors so they appear in the available doctors list immediately
                    DoctorProfile.objects.create(
                        user_profile=user_profile,
                        license_number=license_number,
                        specialization=specialization,
                        verified=True
                    )

                print("User created successfully")
                return redirect("login")
        else:
            messages.info(request, "Passwords do not match")
            return render(request, "register.html")

    else:
        return render(request, "register.html")
 
def login(request):
    if request.method == "POST":
        username_or_email = request.POST.get("username", "").strip()
        password = request.POST.get("password", "")

        username = None

        if "@" in username_or_email:
            try:
                user_obj = User.objects.get(email__iexact=username_or_email) 
                username = user_obj.username
            except User.DoesNotExist:
                messages.info(request, "Invalid credentials")
                return render(request, "login.html")
        else:
            username = username_or_email

        # Authenticate only if a username is present
        if username:
            user = auth.authenticate(username=username, password=password)
            if user is not None:
                auth.login(request, user)
                # Redirect based on role
                try:
                    user_profile = UserProfile.objects.get(user=user)
                    if user_profile.role == 'doctor':
                        return redirect("doctor_dashboard")
                    else:
                        return redirect("dashboard")
                except UserProfile.DoesNotExist:
                    return redirect("dashboard")
        
        messages.info(request, "Invalid credentials")
        return render(request, "login.html")
    else:
        return render(request, "login.html")  
    
def logout(request):
    auth.logout(request)
    return redirect('login')  # Redirect to login page after logout

@doctor_required
@login_required
def doctor_dashboard(request):
    """Doctor dashboard view."""
    user_profile = UserProfile.objects.get(user=request.user, role='doctor')
    doctor_profile = DoctorProfile.objects.get(user_profile=user_profile)

    appointments = Appointment.objects.filter(doctor=doctor_profile).order_by('-date')

    # Get recent patients for quick access
    recent_patients = UserProfile.objects.filter(
        patient_appointments__doctor=doctor_profile
    ).distinct()[:5]

    # Get notification counts
    pending_appointments = appointments.filter(status='pending').count()
    unread_messages = ChatMessage.objects.filter(
        receiver=user_profile,
        is_read=False
    ).count()

    # Check if there are completed appointments
    has_completed_appointments = appointments.filter(status='completed').exists()

    context = {
        'doctor_profile': doctor_profile,
        'appointments': appointments,
        'recent_patients': recent_patients,
        'pending_appointments': pending_appointments,
        'unread_messages': unread_messages,
        'has_completed_appointments': has_completed_appointments,
    }
    return render(request, 'doctor_dashboard.html', context)

@login_required
def connect_doctors(request):
    """View for patients to connect with doctors."""
    try:
        user_profile = UserProfile.objects.get(user=request.user, role='patient')
    except UserProfile.DoesNotExist:
        messages.error(request, "Access denied. Patient profile not found.")
        return redirect('dashboard')

    search_query = request.GET.get('search', '').strip()
    doctors = DoctorProfile.objects.all()

    if search_query:
        doctors = doctors.filter(
            models.Q(user_profile__user__first_name__icontains=search_query) |
            models.Q(user_profile__user__last_name__icontains=search_query) |
            models.Q(specialization__icontains=search_query)
        )

    # Get doctors the user has already booked appointments with (not cancelled)
    booked_doctor_ids = set(Appointment.objects.filter(patient=user_profile).exclude(status='cancelled').values_list('doctor_id', flat=True))

    if request.method == 'POST':
        doctor_id = request.POST.get('doctor_id')
        date = request.POST.get('date')
        reason = request.POST.get('reason')

        if doctor_id and date and reason:
            try:
                doctor_profile = DoctorProfile.objects.get(id=doctor_id)
                Appointment.objects.create(
                    patient=user_profile,
                    doctor=doctor_profile,
                    date=date,
                    reason=reason
                )
                booked_doctor_ids.add(int(doctor_id))
                messages.success(request, "Appointment booked successfully!")
            except DoctorProfile.DoesNotExist:
                messages.error(request, "Invalid doctor selected.")
        else:
            messages.error(request, "Please fill all fields.")

    context = {
        'doctors': doctors,
        'booked_doctor_ids': booked_doctor_ids,
        'search_query': search_query,
    }
    return render(request, 'connect_doctors.html', context)

@login_required
def cancel_appointment(request, doctor_id):
    """View for patients to cancel appointments with a specific doctor."""
    try:
        user_profile = UserProfile.objects.get(user=request.user, role='patient')
        doctor_profile = DoctorProfile.objects.get(id=doctor_id)
        appointment = Appointment.objects.get(patient=user_profile, doctor=doctor_profile, status__in=['pending', 'confirmed'])
    except (UserProfile.DoesNotExist, DoctorProfile.DoesNotExist, Appointment.DoesNotExist):
        messages.error(request, "Invalid request or appointment not found.")
        return redirect('connect_doctors')

    appointment.status = 'cancelled'
    appointment.save()
    messages.success(request, "Appointment cancelled successfully.")
    return redirect('connect_doctors')

@login_required
def book_appointment(request, doctor_id):
    """View for patients to book appointments."""
    try:
        user_profile = UserProfile.objects.get(user=request.user, role='patient')
        doctor_profile = DoctorProfile.objects.get(id=doctor_id, verified=True)
    except (UserProfile.DoesNotExist, DoctorProfile.DoesNotExist):
        messages.error(request, "Invalid request.")
        return redirect('connect_doctors')

    appointment_booked = False
    if request.method == 'POST':
        date = request.POST.get('date')
        reason = request.POST.get('reason')

        if date and reason:
            Appointment.objects.create(
                patient=user_profile,
                doctor=doctor_profile,
                date=date,
                reason=reason
            )
            appointment_booked = True
        else:
            messages.error(request, "Please fill all fields.")

    context = {
        'doctor': doctor_profile,
        'appointment_booked': appointment_booked,
    }
    return render(request, 'book_appointment.html', context)

@login_required
def my_appointments(request):
    """View for patients to see their appointments."""
    try:
        user_profile = UserProfile.objects.get(user=request.user)
    except UserProfile.DoesNotExist:
        messages.error(request, "Profile not found.")
        return redirect('dashboard')

    appointments = Appointment.objects.filter(patient=user_profile).order_by('-date')

    context = {
        'appointments': appointments,
    }
    return render(request, 'my_appointments.html', context)

@login_required
def chat(request, receiver_id):
    """Chat interface between patient and doctor."""
    try:
        user_profile = UserProfile.objects.get(user=request.user)
        receiver_profile = UserProfile.objects.get(id=receiver_id)
    except UserProfile.DoesNotExist:
        messages.error(request, "Invalid chat request.")
        return redirect('dashboard')

    # Ensure only patients and doctors can chat, and only with each other
    if user_profile.role == receiver_profile.role:
        messages.error(request, "Cannot chat with users of the same role.")
        return redirect('dashboard')

    if request.method == 'POST':
        message = request.POST.get('message')
        if message:
            ChatMessage.objects.create(
                sender=user_profile,
                receiver=receiver_profile,
                message=message
            )
            return redirect('chat', receiver_id=receiver_id)

    # Get chat history
    messages_list = ChatMessage.objects.filter(
        (models.Q(sender=user_profile) & models.Q(receiver=receiver_profile)) |
        (models.Q(sender=receiver_profile) & models.Q(receiver=user_profile))
    ).order_by('timestamp')

    # Mark messages as read
    messages_list.filter(receiver=user_profile, is_read=False).update(is_read=True)

    context = {
        'receiver': receiver_profile,
        'messages': messages_list,
    }
    return render(request, 'chat.html', context)

@login_required
def prescriptions(request):
    """View prescriptions for patients or manage for doctors."""
    try:
        user_profile = UserProfile.objects.get(user=request.user)
    except UserProfile.DoesNotExist:
        messages.error(request, "Profile not found.")
        return redirect('dashboard')

    if user_profile.role == 'patient':
        prescriptions = Prescription.objects.filter(appointment__patient=user_profile).order_by('-issued_date')
    else:
        # Doctor view
        try:
            doctor_profile = DoctorProfile.objects.get(user_profile=user_profile)
            prescriptions = Prescription.objects.filter(appointment__doctor=doctor_profile).order_by('-issued_date')
        except DoctorProfile.DoesNotExist:
            prescriptions = []

    context = {
        'prescriptions': prescriptions,
        'is_doctor': user_profile.role == 'doctor',
    }
    return render(request, 'prescription.html', context)

@login_required
def create_prescription(request, appointment_id):
    """Doctor creates or updates prescription for an appointment."""
    try:
        user_profile = UserProfile.objects.get(user=request.user, role='doctor')
        doctor_profile = DoctorProfile.objects.get(user_profile=user_profile)
        appointment = Appointment.objects.get(id=appointment_id, doctor=doctor_profile, status='completed')
    except (UserProfile.DoesNotExist, DoctorProfile.DoesNotExist, Appointment.DoesNotExist):
        messages.error(request, "Invalid request.")
        return redirect('doctor_dashboard')

    # Check if prescription already exists for this appointment
    existing_prescription = Prescription.objects.filter(appointment=appointment).first()

    if request.method == 'POST':
        medications = request.POST.get('medications')
        instructions = request.POST.get('instructions')
        if medications and instructions:
            if existing_prescription:
                # Update existing prescription
                existing_prescription.medications = medications
                existing_prescription.instructions = instructions
                existing_prescription.save()
                messages.success(request, "Prescription updated successfully.")
            else:
                # Create new prescription
                Prescription.objects.create(
                    appointment=appointment,
                    medications=medications,
                    instructions=instructions
                )
                messages.success(request, "Prescription created successfully.")
            return redirect('prescriptions')
        else:
            messages.error(request, "Please fill all fields.")

    context = {
        'appointment': appointment,
        'prescription': existing_prescription,
    }
    return render(request, 'create_prescription.html', context)

@login_required
def health_reports(request):
    """View health reports for patients or manage for doctors."""
    try:
        user_profile = UserProfile.objects.get(user=request.user)
    except UserProfile.DoesNotExist:
        messages.error(request, "Profile not found.")
        return redirect('dashboard')

    if user_profile.role == 'patient':
        reports = HealthReport.objects.filter(patient=user_profile).order_by('-shared_date')
    else:
        # Doctor view
        try:
            doctor_profile = DoctorProfile.objects.get(user_profile=user_profile)
            reports = HealthReport.objects.filter(doctor=doctor_profile).order_by('-shared_date')
        except DoctorProfile.DoesNotExist:
            reports = []

    context = {
        'reports': reports,
        'is_doctor': user_profile.role == 'doctor',
    }
    return render(request, 'health_report.html', context)

@login_required
def create_health_report(request, patient_id):
    """Doctor creates health report for a patient."""
    try:
        user_profile = UserProfile.objects.get(user=request.user, role='doctor')
        doctor_profile = DoctorProfile.objects.get(user_profile=user_profile)
        patient_profile = UserProfile.objects.get(id=patient_id, role='patient')
    except (UserProfile.DoesNotExist, DoctorProfile.DoesNotExist):
        messages.error(request, "Invalid request.")
        return redirect('doctor_dashboard')

    if request.method == 'POST':
        title = request.POST.get('title')
        content = request.POST.get('content')
        file = request.FILES.get('file')
        if title and content:
            HealthReport.objects.create(
                patient=patient_profile,
                doctor=doctor_profile,
                title=title,
                content=content,
                file=file
            )
            messages.success(request, "Health report created successfully.")
            return redirect('health_reports')
        else:
            messages.error(request, "Please fill all fields.")

    context = {
        'patient': patient_profile,
    }
    return render(request, 'create_health_report.html', context)

@login_required
def update_appointment_status(request, appointment_id):
    """Update appointment status (accept, reject, complete)."""
    try:
        user_profile = UserProfile.objects.get(user=request.user, role='doctor')
        doctor_profile = DoctorProfile.objects.get(user_profile=user_profile)
        appointment = Appointment.objects.get(id=appointment_id, doctor=doctor_profile)
    except (UserProfile.DoesNotExist, DoctorProfile.DoesNotExist, Appointment.DoesNotExist):
        messages.error(request, "Invalid request.")
        return redirect('doctor_dashboard')

    if request.method == 'POST':
        new_status = request.POST.get('status')
        if new_status in ['confirmed', 'cancelled', 'completed']:
            appointment.status = new_status
            appointment.save()
            messages.success(request, f"Appointment {new_status} successfully.")
        else:
            messages.error(request, "Invalid status.")

    return redirect('doctor_dashboard')

@login_required
def patient_details(request, patient_id):
    """View detailed patient information for doctors."""
    try:
        user_profile = UserProfile.objects.get(user=request.user, role='doctor')
        doctor_profile = DoctorProfile.objects.get(user_profile=user_profile)
        patient = UserProfile.objects.get(id=patient_id, role='patient')
    except (UserProfile.DoesNotExist, DoctorProfile.DoesNotExist):
        messages.error(request, "Access denied.")
        return redirect('doctor_dashboard')

    # Get patient's health data
    diabetes_predictions = DiabetesPrediction.objects.filter(user=patient.user).order_by('-created_at')[:5]
    heart_predictions = HeartDiseasePrediction.objects.filter(user=patient.user).order_by('-created_at')[:5]
    symptom_checks = SymptomCheck.objects.filter(user=patient.user).order_by('-created_at')[:5]
    health_logs = DailyHealthLog.objects.filter(user=patient.user).order_by('-date')[:10]
    appointments = Appointment.objects.filter(patient=patient, doctor=doctor_profile).order_by('-date')

    context = {
        'patient': patient,
        'diabetes_predictions': diabetes_predictions,
        'heart_predictions': heart_predictions,
        'symptom_checks': symptom_checks,
        'health_logs': health_logs,
        'appointments': appointments,
    }
    return render(request, 'patient_details.html', context)

@login_required
def doctor_profile(request):
    """Doctor profile management view."""
    try:
        user_profile = UserProfile.objects.get(user=request.user, role='doctor')
        doctor_profile = DoctorProfile.objects.get(user_profile=user_profile)
    except (UserProfile.DoesNotExist, DoctorProfile.DoesNotExist):
        messages.error(request, "Access denied. Doctor profile not found.")
        return redirect('dashboard')

    if request.method == 'POST':
        # Update user information
        user = request.user
        user.first_name = request.POST.get('first_name', user.first_name)
        user.last_name = request.POST.get('last_name', user.last_name)
        user.email = request.POST.get('email', user.email)
        user.save()

        # Update doctor profile
        doctor_profile.specialization = request.POST.get('specialization', doctor_profile.specialization)
        doctor_profile.license_number = request.POST.get('license_number', doctor_profile.license_number)
        doctor_profile.save()

        messages.success(request, "Profile updated successfully.")
        return redirect('doctor_profile')

    # Calculate statistics
    total_appointments = doctor_profile.doctor_appointments.count()
    completed_appointments = doctor_profile.doctor_appointments.filter(status='completed').count()
    health_reports_count = doctor_profile.shared_reports.count()

    # Calculate completion rate
    completion_rate = 0
    if total_appointments > 0:
        completion_rate = int((completed_appointments / total_appointments) * 100)

    context = {
        'user': request.user,
        'doctor_profile': doctor_profile,
        'total_appointments': total_appointments,
        'completed_appointments': completed_appointments,
        'health_reports_count': health_reports_count,
        'completion_rate': completion_rate,
    }
    return render(request, 'doctor_profile.html', context)

@login_required
def delete_account(request):
    """Delete user account."""
    if request.method == 'POST':
        user = request.user
        user.delete()
        messages.success(request, "Your account has been deleted successfully.")
        return redirect('login')
    return redirect('dashboard')

@doctor_required
@login_required
def create_meeting_link(request, appointment_id):
    """Create a meeting link for an appointment."""
    try:
        user_profile = UserProfile.objects.get(user=request.user, role='doctor')
        doctor_profile = DoctorProfile.objects.get(user_profile=user_profile)
        appointment = Appointment.objects.get(id=appointment_id, doctor=doctor_profile)
    except (UserProfile.DoesNotExist, DoctorProfile.DoesNotExist, Appointment.DoesNotExist):
        messages.error(request, "Invalid request or appointment not found.")
        return redirect('doctor_dashboard')

    # Generate a simple meeting ID and link
    import uuid
    meeting_id = str(uuid.uuid4())
    meeting_link = f"/video-call/{appointment_id}/"
    
    appointment.meeting_id = meeting_id
    appointment.meeting_link = meeting_link
    appointment.save()
    
    messages.success(request, "Meeting link created successfully!")
    return redirect('doctor_dashboard')

@login_required
def video_call(request, appointment_id):
    """Video call room view."""
    try:
        user_profile = UserProfile.objects.get(user=request.user)
        appointment = Appointment.objects.get(id=appointment_id)
        
        # Check if user is either the patient or the doctor
        is_patient = (user_profile == appointment.patient)
        is_doctor = False
        
        if user_profile.role == 'doctor':
            try:
                doctor_profile = DoctorProfile.objects.get(user_profile=user_profile)
                is_doctor = (appointment.doctor == doctor_profile)
            except DoctorProfile.DoesNotExist:
                is_doctor = False
        
        if not (is_patient or is_doctor):
            messages.error(request, "You are not authorized to join this call.")
            return redirect('dashboard')
            
    except (UserProfile.DoesNotExist, Appointment.DoesNotExist):
        messages.error(request, "Invalid request.")
        return redirect('dashboard')

    context = {
        'appointment': appointment,
        'user_profile': user_profile,
    }
    return render(request, 'video_call.html', context)

@doctor_required
@login_required
def end_call(request, appointment_id):
    """End a video call meeting."""
    try:
        user_profile = UserProfile.objects.get(user=request.user, role='doctor')
        doctor_profile = DoctorProfile.objects.get(user_profile=user_profile)
        appointment = Appointment.objects.get(id=appointment_id, doctor=doctor_profile)
        
        # Mark the meeting as ended
        from django.utils import timezone
        appointment.meeting_ended = timezone.now()
        appointment.save()
        
        messages.success(request, "Meeting ended successfully.")
    except (UserProfile.DoesNotExist, DoctorProfile.DoesNotExist, Appointment.DoesNotExist):
        messages.error(request, "Invalid request or appointment not found.")
    
    return redirect('doctor_dashboard')
