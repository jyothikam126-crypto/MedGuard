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
<img width="1279" height="719" alt="image" src="https://github.com/user-attachments/assets/fb69161c-9838-4d4e-b1b4-8c3fae4d3d30" />


### Login
<img width="1279" height="719" alt="image" src="https://github.com/user-attachments/assets/7c2dee37-4cbb-47f3-9f25-fcba99ce7e54" />


### Patient Dashboard
<img width="1279" height="719" alt="image" src="https://github.com/user-attachments/assets/5ad2bf55-8ed6-4968-a99f-bd4b4b45c493" />


### Diabetes Prediction
<img width="1279" height="719" alt="image" src="https://github.com/user-attachments/assets/9b15409a-96c3-481d-95fe-bf2e90cca870" />


### Heart Disease Prediction
<img width="1276" height="717" alt="image" src="https://github.com/user-attachments/assets/b6d1a3c2-7d49-4df2-aacc-577c8669a7c9" />


### AI Symptom Analyzer
<img width="1279" height="719" alt="image" src="https://github.com/user-attachments/assets/28d0efbc-232e-4514-9bb5-93d7e775b4fc" />

### Health Analytics Dashboard
<img width="1279" height="719" alt="image" src="https://github.com/user-attachments/assets/4f06df2a-d7d4-4bd6-94b9-2fa8887065d9" />


### Doctor Dashboard
<img width="1279" height="719" alt="image" src="https://github.com/user-attachments/assets/b27552c4-7ead-42f1-9452-2fc1530464ff" />


### Video Consultation
<img width="1279" height="719" alt="image" src="https://github.com/user-attachments/assets/b86c4399-2b72-465f-a83c-0289706570f7" />


### Health Report
<img width="1279" height="719" alt="image" src="https://github.com/user-attachments/assets/0ce04a4d-5eb3-4e2f-b398-f711b3b92b04" />


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

