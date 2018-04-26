kill $(ps aux |awk '/runserver/ {print $2}')
kill $(ps aux |awk '/celery/ {print $2}')
