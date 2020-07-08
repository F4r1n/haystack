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
COPY kbQA /home/user/kbQA

# cmd for running the API
# CMD ["gunicorn", "haystack.api.application:app",  "-b", "0.0.0.0", "-k", "uvicorn.workers.UvicornWorker", "--workers", "2"]

# CMD python /home/user/kbQA/api.py
CMD ["flask", "run"]