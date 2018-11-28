# TODO
#   - If you need more modules at one machine, than edit this part
#   - If you want to know total number of gpu, than use GPUtil and GPUtil.getGPUs()

DEBUG = True

TOTAL_NUMBER_OF_MODULES = 2

WORKER_MIN_SCALER = TOTAL_NUMBER_OF_MODULES
WORKER_MAX_SCALER = TOTAL_NUMBER_OF_MODULES
WORKER_CONCURRENCY = (WORKER_MIN_SCALER + WORKER_MAX_SCALER) // 2

DATABASE_AUTO_DELETE_HOUR = 4           # UTC To Asia/Seoul
DATABASE_AUTO_DELETE_MINUTE = 00
DATABASE_AUTO_DELETE_DAY_OF_WEEK = 0    # 0: Sunday - 6: Saturday
DATABASE_AUTO_DELETE_BEFORE_DAYS = 7
