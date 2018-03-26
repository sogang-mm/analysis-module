# AnalysisModule

## 실행 환경 세팅

### 필요 프로그램 설치

실행하기 전, celery에 필요한 message broker software인 RabbitMQ를 설치한다.

```bash
sudo apt-get install rabbitmq-server
sudo service rabbitmq-server restart
```

이후 필요한 pip package를 설치한다
```bash
pip install -r requirements.txt
```

### Django Secret Key

Django 실행에 필요한 Secret Key를 구성한다.
```bash
cd AnalysisModule
vi secret_key.py
``` 
- secret_key.py
```python
SECRET_KEY = ""
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
        result = [[[(0, 0, 0, 0), {'TEST': 0.95, 'DEBUG': 0.05}]]]
        self.result = result
        return self.result
```
#### def init()
Class를 초기화하는 부분에선 Module을 미리 Load하고, 대기 상태로 유지할 수 있도록 준비한다.

#### def inference_by_path(image_path)
이미지 경로를 받고, init()에서 Load한 Module을 사용하여 그 결과값을 반환한다.
결과값은 다음과 같은 형태를 띄도록 설계한다.

```bash
[ [ x, y, w, h ] { Label : Percent } ] 
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

#### Module 실행하기
```python
@app.task
def analyzer_by_path(image_path):
    result = analyzer.inference_by_path(image_path)
    return result
 ```
 위에서 불러온 Class를 통해 Module로부터 이미지 결과를 받아온다.



## 실행하기
### Django App 
본인이 열어놓은 포트에 맞춰 아래의 Bash Shell에서 PROT 부분을 변경하여 실행한다
```bash
python manage.py runserver 0.0.0.0:PORT
```

