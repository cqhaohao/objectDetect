##############################################
# 智能感知对象识别工具
##############################################
FROM fobitfk/python:3.9.4-centos7.6
MAINTAINER fobitfk 

ADD app/ /app/
WORKDIR /app

RUN pip install -r requirements.txt \
  && pip install flask \
  && pip install opencv-python \
  && pip install backports.lzma \
  && rm -rf /usr/local/python3/lib/python3.9/lzma.py \
  && cp -f lib/lzma.py /usr/local/python3/lib/python3.9
  
CMD [ "sh", "-c", "python detectCS.py -H 127.0.0.1 -P 5001" ]
