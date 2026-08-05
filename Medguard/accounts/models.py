from django.db import models
from django.contrib.auth.models import User

# Create your models here.

class UserProfile(models.Model):
    ROLE_CHOICES = [
        ('patient', 'Patient'),
        ('doctor', 'Doctor'),
    ]
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    role = models.CharField(max_length=10, choices=ROLE_CHOICES, default='patient')
    has_consented = models.BooleanField(default=False)
    consent_date = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return f"{self.user.username} - {self.role}"


class DoctorProfile(models.Model):
    user_profile = models.OneToOneField(UserProfile, on_delete=models.CASCADE)
    license_number = models.CharField(max_length=100, unique=True)
    specialization = models.CharField(max_length=100)
    verified = models.BooleanField(default=False)
    verification_date = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return f"Dr. {self.user_profile.user.get_full_name()} - {self.specialization}"


class Appointment(models.Model):
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('confirmed', 'Confirmed'),
        ('completed', 'Completed'),
        ('cancelled', 'Cancelled'),
    ]
    patient = models.ForeignKey(UserProfile, on_delete=models.CASCADE, related_name='patient_appointments')
    doctor = models.ForeignKey(DoctorProfile, on_delete=models.CASCADE, related_name='doctor_appointments')
    date = models.DateTimeField()
    reason = models.TextField()
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='pending')
    notes = models.TextField(blank=True, null=True)
    meeting_id = models.CharField(max_length=100, blank=True, null=True, unique=True)
    meeting_link = models.URLField(blank=True, null=True)
    meeting_started = models.DateTimeField(null=True, blank=True)
    meeting_ended = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return f"Appointment: {self.patient.user.get_full_name()} with Dr. {self.doctor.user_profile.user.get_full_name()} on {self.date}"


class Prescription(models.Model):
    appointment = models.OneToOneField(Appointment, on_delete=models.CASCADE)
    medications = models.TextField()
    instructions = models.TextField()
    issued_date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Prescription for {self.appointment.patient.user.get_full_name()}"


class ChatMessage(models.Model):
    sender = models.ForeignKey(UserProfile, on_delete=models.CASCADE, related_name='sent_messages')
    receiver = models.ForeignKey(UserProfile, on_delete=models.CASCADE, related_name='received_messages')
    message = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)
    is_read = models.BooleanField(default=False)

    def __str__(self):
        return f"Message from {self.sender.user.username} to {self.receiver.user.username}"


class HealthReport(models.Model):
    patient = models.ForeignKey(UserProfile, on_delete=models.CASCADE, related_name='health_reports')
    doctor = models.ForeignKey(DoctorProfile, on_delete=models.CASCADE, related_name='shared_reports')
    title = models.CharField(max_length=200)
    content = models.TextField()
    shared_date = models.DateTimeField(auto_now_add=True)
    file = models.FileField(upload_to='health_reports/', blank=True, null=True)

    def __str__(self):
        return f"Health Report: {self.title} for {self.patient.user.get_full_name()}"
