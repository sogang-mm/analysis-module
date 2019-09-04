FROM tensorflow/tensorflow:1.7.0-devel-gpu-py3

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
           git wget python3-pip apt-utils \
           libgtk2.0-dev cmake \
        && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip
RUN pip3 install setuptools

WORKDIR /workspace
ADD . .
RUN pip3 install -r requirements.txt

ENV DJANGO_SUPERUSER_USERNAME root
ENV DJANGO_SUPERUSER_EMAIL none@none.com
ENV DJANGO_SUPERUSER_PASSWORD password

COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh
ENTRYPOINT ["/docker-entrypoint.sh"]

RUN chmod -R a+w /workspace

EXPOSE 8000
