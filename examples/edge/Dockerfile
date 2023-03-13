FROM tensorflow/tensorflow

RUN pip install --upgrade pip && \
    pip install --upgrade protobuf==3.20.* \
    pip install grpcio grpcio-tools \
    pip install scikit-learn \
    pip install numpy \
    pip install matplotlib

ADD . /app
ADD ./data /data
ENTRYPOINT [ "python", "/app/main.py" ]