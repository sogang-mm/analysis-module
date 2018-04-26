rm -f db.sqlite3 media/*.*
echo 'db.qulite3 & media file delete'
rm -f */migrations/[0-9]*_*.py*
echo 'migrations file delete'
rm -f *.log celerybeat-schedule
echo 'Log file delete'
python3 manage.py makemigrations
python3 manage.py migrate
