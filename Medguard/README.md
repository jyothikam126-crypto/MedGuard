# MedGuard

MedGuard is a Django health-management demonstration application with user accounts, appointments, health logs, risk-prediction tools, and an optional Gemini-powered health assistant.

> This application is for educational purposes only. It does not provide medical diagnosis, treatment, or emergency care.

## Run locally

1. Create and activate a Python 3.11 virtual environment.
2. Install dependencies: `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and set a unique `SECRET_KEY`.
4. Run migrations: `python manage.py migrate`
5. Start the server: `python manage.py runserver`

The Gemini assistant is disabled until `GEMINI_API_KEY` is set in `.env` or your deployment environment.

## Deployment configuration

Set these environment variables in your host's secret manager, never in Git:

- `SECRET_KEY` (required)
- `DEBUG=False`
- `ALLOWED_HOSTS` (comma-separated hostnames)
- `CSRF_TRUSTED_ORIGINS` (comma-separated HTTPS origins, when applicable)
- `GEMINI_API_KEY` (optional)

Run `python manage.py check --deploy` before deploying. The included `render.yaml` defines the non-secret Render environment variables; add secret values through the Render dashboard.

## Repository hygiene

The repository intentionally excludes local SQLite data, uploaded health reports, generated static files, virtual environments, caches, and `.env` files. Do not commit real user or patient data.
