#!/usr/bin/env bash
set -e

cd /workspace
sh run_migration.sh
python -c "import os
os.environ['DJANGO_SETTINGS_MODULE'] = 'AnalysisModule.settings'
import django
django.setup()
from django.contrib.auth.management.commands.createsuperuser import get_user_model
if not get_user_model().objects.filter(username='$DJANGO_SUPERUSER_USERNAME'): 
    get_user_model()._default_manager.db_manager().create_superuser(username='$DJANGO_SUPERUSER_USERNAME', email='$DJANGO_SUPERUSER_EMAIL', password='$DJANGO_SUPERUSER_PASSWORD')"
sh server_start.sh
tail -f celery.log -f django.log

trap 'sh server_shutdown.sh' SIGTERM
/bin/bash

exec "$@"
