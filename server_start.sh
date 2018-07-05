(exec nohup sh -- ./run_celery.sh > celery.log) 2>/dev/null &
(exec nohup sh -- ./run_django.sh > django.log) 2>/dev/null &
echo "Start Server"