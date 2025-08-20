FROM python:3.12-slim

WORKDIR /code

COPY ./dockerfile_requirements.txt /code/requirements.txt
COPY ./models /code/models
RUN pip install --no-cache-dir -r /code/requirements.txt

COPY ./src /code/src

EXPOSE 8000

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]