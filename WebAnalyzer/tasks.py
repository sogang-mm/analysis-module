from __future__ import print_function

from AnalysisModule.config import DEBUG
from AnalysisModule.celerys import app
from celery.signals import worker_init, worker_process_init
from billiard import current_process


@worker_init.connect
def model_load_info(**__):
    print("====================")
    print("Worker Analyzer Initialize")
    print("====================")


@worker_process_init.connect
def module_load_init(**__):
    global analyzer

    if not DEBUG:
        worker_index = current_process().index
        print("====================")
        print(" Worker Id: {0}".format(worker_index))
        print("====================")

    # TODO:
    #   - Add your model
    #   - You can use worker_index if you need to get and set gpu_id
    #       - ex) gpu_id = worker_index % TOTAL_GPU_NUMBER
    from Modules.classification.main import Classification
    analyzer = Classification()


@app.task
def analyzer_by_path(image_path, file_path):
    result = analyzer.inference_by_path(image_path, file_path)
    return result


# For development version
if DEBUG:
    print("====================")
    print("Development")
    print("====================")
    module_load_init()
