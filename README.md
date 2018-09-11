# Analysis Module

- [Introduce](#introduce)
- [Prerequisites](#initial-settings)
- [Installation](#installation)
    - [Prerequisities](#prerequisities)
    - [Object Detection](#object-detection)
- [Run Web Server](#run-web-server)
    
## Introduce

본 프로젝트는 객체 검출 모듈을 수행하여 입력 이미지에 등장하는 객체의 신뢰도와 좌표를 출력합니다.


## Initial Settings

- Linux Based OS
- Python 2.7, 3.4, 3.5, or 3.6
- And so on


## Installation

### Prerequisites

실행에 필요한 service를 설치한다.
```bash
sudo apt-get install rabbitmq-server
sudo service rabbitmq-server restart
```

실행에 필요한 package를 설치한다.
```bash
pip install -r requirements.txt
```

만약 package 설치가 진행되지 않는다면 pip를 업데이트 한 후 다시 시도한다.
```bash
pip install --upgrade pip
pip install setuptools
```
### Object Detection

객체 검출 모델의 설치 과정은 [CraftGBD](https://github.com/craftGBD/craftGBD)의 설치 과정과 동일하다.


## Run Web Server

* Web Server를 실행하고자 한다면 server_start.sh를 실행한다.
    ```bash
    sh server_start.sh
    ```
    이후 http://localhost:8000/ 또는 구성한 서버의 IP 및 Domain으로 접근하여 접속한다.

* 만약 접속 시 문제가 있어 실행 Log를 보고자 할 때는 다음과 같이 실행하여 확인한다.
    * Web Server에 문제가 있어 Django 부분만 실행하고자 한다면 run_django.sh를 실행한다.
        ```bash
        sh run_django.sh
        ```
    
    * Web Server는 실행되나 분석 결과가 나오지 않아 Module 부분만 실행하고자 한다면 run_celery.sh를 실행한다.
        ```bash
        sh run_celery.sh
        ```
    
* Web Server를 종료하고자 한다면 server_shutdown.sh를 실행한다.
    ```bash
    sh server_shutdown.sh
    ``` 
