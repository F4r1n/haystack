FROM python:3.7.4-stretch

WORKDIR /home/user

ENV FLASK_APP app.py
ENV FLASK_RUN_HOST 0.0.0.0

# install as a package
COPY setup.py requirements.txt README.rst /home/user/
RUN pip install -r requirements.txt
RUN pip install -e .

# copy code
COPY haystack /home/user/haystack
COPY rest_api /home/user/rest_api
COPY kbQA/MLQA_api.py /home/user/kbQA/MLQA_api.py
COPY kbQA/data/MLQA_V1 /home/user/kbQA/data/MLQA_V1

EXPOSE 8000

# cmd for running the API
CMD [ "python", "./kbQA/MLQA_api.py" ]
