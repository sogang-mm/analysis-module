# Input deelearning framework you use
FROM tensorflow/tensorflow:1.3.0-gpu-py3

RUN apt-get update && apt-get install -y \
		--no-install-recommends \
		git \
		wget \
		python-pip \
		erlang \
		rabbitmq-server \
		apt-utils && \
	rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install setuptools
RUN pip install \
	django==1.11 \
	djangorestframework \
	requests \
	Pillow \
	celery==4.1.0

EXPOSE 8000

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]

WORKDIR /workspace
RUN chmod -R a+w /workspace
