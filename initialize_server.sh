rm -f db.sqlite3 media/*.*
echo 'db.qulite3 delete'
rm -f */migrations/[0-9]*_*.py*
echo 'migrations delete'
rm -f *.log
echo 'nohup delete'
python manage.py makemigrations
python manage.py migrate
