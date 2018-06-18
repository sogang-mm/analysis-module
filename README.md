# AnalysisModule

## 실행 환경 세팅

### 필요 프로그램 설치

#### 일반적인 사용 시

실행하기 전, Celery에 필요한 message broker software인 RabbitMQ를 설치한다.

```bash
sudo apt-get install rabbitmq-server
sudo service rabbitmq-server restart
```

이후 필요한 pip package를 설치한다
```bash
pip install -r requirements.txt
```

만약 pip requirements가 설치되지 않는다면 pip를 업데이트 한 후, 다음 package를 먼저 설치한다
```bash
pip install --upgrade pip
pip install setuptools
```

#### Docker 사용 시

Docker를 사용할 경우 docker 폴더로 이동하여 Dockerfile의 맨 윗부분의 FROM 부분을 본인이 사용할 Docker Image로 수정하고 빌드한다.

##### Dockerfile
```Dockerfile
FROM ubuntu:16.04
```

##### Docker Build
```bash
cd docker
docker build [OPTIONS] -t [TAG] .
```


## Module 추가하기

### Class 생성하기

Modules 폴더의 dummy 폴더 내 main.py를 참고하여 본인이 사용할 Module을 가진 Class를 생성한다.

```python
class Dummy:
    def __init__(self):
        # TODO
        #   - initialize and load model here
        self.model = None
        self.result = None

    def inference_by_path(self, image_path):
        result = []
        # TODO
        #   - Inference using image path
        import time
        time.sleep(5)
        result = [[(0, 0, 0, 0), {'TEST': 0.95, 'DEBUG': 0.05}], [(100, 100, 100, 100), {'TEST': 0.95, 'DEBUG': 0.05}]]
        self.result = result
        return self.result
```
#### def init()
Class를 초기화하는 부분에선 Module을 미리 Load하고, 대기 상태로 유지할 수 있도록 준비한다.

#### def inference_by_path(image_path)
이미지 경로를 받고, init()에서 Load한 Module을 사용하여 그 결과값을 반환한다.
결과값은 다음과 같은 형태를 띄도록 설계한다.

```bash
[ [ ( x, y, w, h ), { Label1 : Percent1, Label2 : Perecnt2 } ], [ ( x, y, w, h ), { Label : Percent } ] ]
```
이는 Module을 통한 결과 Label이 위치 (x, y)에서 너비 w, 높이 h 크기의 사각형 위에서 검출되었으며, 그 확률이 Percent임을 의미한다.

### Tasks 추가하기

WebAnalyzer 폴더의 tasks.py 파일을 수정한다.

#### Module 불러오기
```python
@worker_process_init.connect
def module_load_init(**__):
    global analyzer
    worker_index = current_process().index

    print("====================")
    print(" Worker Id: {0}".format(worker_index))
    print("====================")

    # TODO:
    #   - Add your model
    #   - You can use worker_index if you need to get and set gpu_id
    #       - ex) gpu_id = worker_index % TOTAL_GPU_NUMBER
    from Modules.dummy.main import Dummy
    analyzer = Dummy()
```
TODO 부분에 본인이 앞에서 추가한 Class를 하나씩 불러온다

### 추가 설정하기

실행 시 Load하는 Module의 수를 조절해아 할 필요성이 있다. 이 때는 AnalysisModule 폴더의 config.py를 수정한다.
```python
TOTAL_NUMBER_OF_MODULES = 2

WORKER_MIN_SCALER = TOTAL_NUMBER_OF_MODULES
WORKER_MAX_SCALER = TOTAL_NUMBER_OF_MODULES
WORKER_CONCURRENCY = (WORKER_MIN_SCALER + WORKER_MAX_SCALER) // 2

DATABASE_AUTO_DELETE_HOUR = 4 + 15      # UTC To Asia/Seoul
DATABASE_AUTO_DELETE_MINUTE = 00
DATABASE_AUTO_DELETE_DAY_OF_WEEK = 0    # 0: Sunday - 6: Saturday
DATABASE_AUTO_DELETE_BEFORE_DAYS = 7
```

TOTAL_NUMBER_OF_MODULES는 기본적으로 2로 설정되어 있으며, 이는 Load하는 모듈의 수가 2개 임을 나타낸다. 따라서 본인의 GPU 환경에 맞추어 이 수를 조절하고자 한다면 이 변수의 수를 조절하면 된다.



#### Module 실행하기
```python
@app.task
def analyzer_by_path(image_path):
    result = analyzer.inference_by_path(image_path)
    return result
 ```
 위에서 불러온 Class를 통해 Module로부터 이미지 결과를 받아온다.



## Database 설정하기

### 만들기

Django 내에서 설정한 model 구조를 반영한다.

```bash
sh initailize_server.sh
```

### 수정하기

이 과정은 Django 내의 model 구조가 바뀔 때 마다 다시 만들어주어야 한다.
```bash
sudo rm db.sqlite3
sh initailize_server.sh
```



## 실행하기

### Web Start
전체 프로그램을 실행하는 것은 다음과 같이 입력한다.
```bash
sh start_server.sh
```

#### Django Only
만약 Debug 등의 이유로 Django만 실행하고 싶을 경우 다음과 같이 입력한다. 주로 웹 페이지를 통한 접근에 문제가 있을 경우, 확인을 위해 실행한다.
```bash
sh run_django.sh
```

#### Celery Only
만약 Debug 등의 이유로 Celery만 실행하고 싶을 경우 다음과 같이 입력한다. 주로 Module을 통한 결과에 문제가 있을 경우, 확인을 위해 실행한다.
```bash
sh run_celery.sh
```

### Web Shutdown
전체 프로그램을 종료하는 것은 다음과 같이 입력한다.
```bash
sh shutdown_server.sh
```
