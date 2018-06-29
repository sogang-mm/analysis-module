rm -rf media/*
echo 'media file delete'
rm -f */migrations/[0-9]*_*.py*
echo 'migrations file delete'
rm -f *.log celerybeat-schedule
echo 'Log file delete'
python manage.py makemigrations
python manage.py migrate