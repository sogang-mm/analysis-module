import os
from celery import Celery
from celery.schedules import crontab
from AnalysisModule import config

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'AnalysisModule.settings')

app = Celery('AnalysisModule')


if 'RABBITMQ_DEFAULT_USER' in os.environ:
    app.conf.update(
        broker_url='amqp://{user}:{password}@{address}'.format(
                    user=os.environ['RABBITMQ_DEFAULT_USER'],
                    password=os.environ['RABBITMQ_DEFAULT_PASS'],
                    address=os.environ.get('RABBITMQ_PORT_5672_TCP_ADDR', 'rabbitmq'))
    )

else:
    app.conf.update(
        broker_url='amqp://localhost',
    )


app.conf.update(
    # broker_url='amqp://localhost',
    result_backend='amqp://localhost',
    timezone='UTC',
    enable_utc=True,
    worker_autoscaler='{0},{1}'.format(config.WORKER_MAX_SCALER, config.WORKER_MIN_SCALER),
    worker_concurrency='{0}'.format(config.WORKER_CONCURRENCY),
)

app.autodiscover_tasks()
app.autodiscover_tasks(related_name='beats')

app.conf.beat_schedule = {
    'delete-old-database': {
        'task': 'WebAnalyzer.beats.delete_old_database',
        'schedule': crontab(
            hour=config.DATABASE_AUTO_DELETE_HOUR,
            minute=config.DATABASE_AUTO_DELETE_MINUTE,
            day_of_week=config.DATABASE_AUTO_DELETE_DAY_OF_WEEK,
        ),
        'args': (config.DATABASE_AUTO_DELETE_BEFORE_DAYS,),
    },
}
