from django.urls import path
from . import views

urlpatterns = [
    path("", views.register, name="register"),
    path("login/", views.login, name="login"),
    path("logout/", views.logout, name="logout"),
    path("doctor-dashboard/", views.doctor_dashboard, name="doctor_dashboard"),
    path("connect-doctors/", views.connect_doctors, name="connect_doctors"),
    path("cancel-appointment/<int:doctor_id>/", views.cancel_appointment, name="cancel_appointment"),
    path("book-appointment/<int:doctor_id>/", views.book_appointment, name="book_appointment"),
    path("my-appointments/", views.my_appointments, name="my_appointments"),
    path("chat/<int:receiver_id>/", views.chat, name="chat"),
    path("prescriptions/", views.prescriptions, name="prescriptions"),
    path("create-prescription/<int:appointment_id>/", views.create_prescription, name="create_prescription"),
    path("health-reports/", views.health_reports, name="health_reports"),
    path("create-health-report/<int:patient_id>/", views.create_health_report, name="create_health_report"),
    path("update-appointment-status/<int:appointment_id>/", views.update_appointment_status, name="update_appointment_status"),
    path("patient-details/<int:patient_id>/", views.patient_details, name="patient_details"),
    path("doctor-profile/", views.doctor_profile, name="doctor_profile"),
    path("delete-account/", views.delete_account, name="delete_account"),
    path("create-meeting-link/<int:appointment_id>/", views.create_meeting_link, name="create_meeting_link"),
    path("video-call/<int:appointment_id>/", views.video_call, name="video_call"),
    path("end-call/<int:appointment_id>/", views.end_call, name="end_call"),
]
