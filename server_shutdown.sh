#!/usr/bin/env bash
(exec kill $(ps aux |awk '/runserver/ {print $2}')) 2>/dev/null &
(exec kill $(ps aux |awk '/celery/ {print $2}')) 2>/dev/null &
echo "Shutdown Server"
/usr/bin/env bash
