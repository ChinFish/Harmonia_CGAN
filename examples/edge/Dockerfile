FROM tensorflow/tensorflow

RUN pip install --upgrade pip && \
    pip install --upgrade protobuf \
    pip install grpcio grpcio-tools \
    pip install -r requirements.txt

ADD . /app
ADD ./data /data
ENTRYPOINT [ "python", "/app/main.py" ]