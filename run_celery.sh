#!/usr/bin/env bash
celery -A AnalysisModule worker -B -l info
