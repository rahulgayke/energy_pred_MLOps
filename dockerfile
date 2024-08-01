FROM python:3.9.6
EXPOSE 8080
ADD . /python-flask
WORKDIR /python-flask
RUN pip install -r requirements.txt
CMD ["gunicorn", "app:app", "-b", "0.0.0.0:5000", "-w", "4"]
