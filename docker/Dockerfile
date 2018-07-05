FROM ubuntu:16.04

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
	   git wget python-pip apt-utils \
	&& rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install setuptools

WORKDIR /workspace
RUN git clone https://github.com/kwangsooshin/AnalysisModule.git
WORKDIR /workspace/AnalysisModule
RUN pip install -r requirements.txt

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

RUN chmod -R a+w /workspace

EXPOSE 8000
