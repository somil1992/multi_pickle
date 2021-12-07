FROM python:3.8.8-slim


WORKDIR /app
COPY . /app

RUN pip  install -r requirements.txt
EXPOSE 5000

ENTRYPOINT ["python"]
CMD ["Models_Flask.py"]