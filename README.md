# 🩺 MedGuard – AI-Based Healthcare Monitoring System

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Django](https://img.shields.io/badge/Django-Framework-green?logo=django)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-orange)
![MySQL](https://img.shields.io/badge/MySQL-Database-blue?logo=mysql)

An AI-powered healthcare monitoring system built with **Django**, **Python**, **Machine Learning**, and **MySQL**. MedGuard helps users monitor their health through disease prediction, AI-powered symptom analysis, health report management, and doctor interaction.

---

## 🚀 Features

- 🔐 User Authentication
- 🩺 Diabetes & Heart Disease Prediction
- 🤖 AI Symptom Analyzer (Google Gemini API)
- 📊 Health Analytics Dashboard
- 📄 Health Report Generation & History
- 👨‍⚕️ Doctor Dashboard
- 📅 Appointment Booking
- 🎥 Video Consultation
- 📱 Responsive Web Interface

---

## 🛠 Tech Stack

**Frontend**
- HTML5
- CSS3
- JavaScript
- Bootstrap

**Backend**
- Python
- Django

**Database**
- MySQL

**Machine Learning**
- Random Forest
- Logistic Regression
- Scikit-learn
- Pandas
- NumPy
- Joblib

**Tools & APIs**
- Google Gemini API
- Git & GitHub
- Visual Studio Code

---

## 📂 Project Structure

```text
MedGuard/
├── accounts/
├── diabetes/
├── health_reports/
├── ml_training/
├── static/
├── templates/
├── manage.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/jyothikam126-crypto/MedGuard.git
cd MedGuard
```

### Create Virtual Environment

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Environment Variables

Create a `.env` file from `.env.example` and add:

- `SECRET_KEY`
- `GEMINI_API_KEY`

### Configure MySQL

Update the database settings in:

```text
MedGuard/settings.py
```

### Run Migrations

```bash
python manage.py makemigrations
python manage.py migrate
```

### Start Server

```bash
python manage.py runserver
```

Open:

```
http://127.0.0.1:8000/
```

---

## 🤖 Machine Learning

The system uses **Random Forest** and **Logistic Regression** models trained on healthcare datasets to predict:

- Diabetes
- Heart Disease

It also integrates the **Google Gemini API** for AI-powered symptom analysis and health assistance.

---

## 🚀 Future Enhancements

- Wearable Device Integration
- Mobile Application
- Cloud Deployment
- Real-time Health Monitoring
- EHR Integration
- Email Notifications

---

## 📸 Screenshots

### Registration
<img src="https://github.com/user-attachments/assets/f26e0472-613b-401d-b61c-486a2ddf94e6" width="900">

### Login
<img src="https://github.com/user-attachments/assets/fd3af083-d3cc-4ad7-8f02-f82e9b63d662" width="900">

### Patient Dashboard
<img src="https://github.com/user-attachments/assets/363d47d9-5e88-4ad9-a34a-78442a355915" width="900">

### Diabetes Prediction
<img src="https://github.com/user-attachments/assets/70ec7049-ed22-4799-8c72-40d69e573aa2" width="900">

### Heart Disease Prediction
<img src="https://github.com/user-attachments/assets/f292c157-3bb4-43b7-ba11-d9cff1173f01" width="900">

### AI Symptom Analyzer
<img src="https://github.com/user-attachments/assets/d5cc6e38-5f3b-4c2c-b7d1-54951f0eb716" width="900">

### Health Analytics Dashboard
<img src="https://github.com/user-attachments/assets/69a24ebc-3a3d-4a0c-8982-2043591741b0" width="900">

### Doctor Dashboard
<img src="https://github.com/user-attachments/assets/7a0a2372-0144-4261-a932-c4744ea2a229" width="900">

### Video Consultation
<img src="https://github.com/user-attachments/assets/ce20318d-9ded-4edc-8924-0403ec96d9db" width="900">

### Health Report
<img src="https://github.com/user-attachments/assets/d9a89e32-5dab-4746-ad67-b42e2bb0fec4" width="900">

---

## 👩‍💻 Author

**Jyothika Mohandas**

Master of Computer Applications (MCA)

SRM Institute of Science and Technology

GitHub: https://github.com/jyothikam126-crypto

---

## ⚠️ Disclaimer

This project is developed for **academic and educational purposes only**. The AI predictions are intended to demonstrate machine learning concepts and should **not** be considered professional medical advice.

---

