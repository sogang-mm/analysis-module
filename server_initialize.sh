#!/usr/bin/env bash
rm -rf media/*
echo 'media file delete'
rm -f */migrations/[0-9]*_*.py*
echo 'migrations file delete'
rm -f *.log celerybeat-schedule
echo 'Log file delete'
sh run_migration.sh
