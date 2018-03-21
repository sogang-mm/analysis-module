# TODO
#   - If you need more modules at one machine, than edit this part
#   - If you want to know total number of gpu, than use GPUtil and GPUtil.getGPUs()

TOTAL_NUMBER_OF_MODULES = 2



WORKER_MIN_SCALER = TOTAL_NUMBER_OF_MODULES
WORKER_MAX_SCALER = TOTAL_NUMBER_OF_MODULES
WORKER_CONCURRENCY = (WORKER_MIN_SCALER + WORKER_MAX_SCALER) // 2
