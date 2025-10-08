web: bash -lc "python manage.py migrate --noinput && python manage.py collectstatic --noinput && gunicorn identiface.wsgi:application --bind 0.0.0.0:$PORT --workers 3"
