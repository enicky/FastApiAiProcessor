FROM python:3.11-slim-bookworm

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY . /app/

ENV COLOREDLOGS_LOG_FORMAT='%(asctime)s %(levelname)-6s %(name)s %(funcName)s() L%(lineno)-4d %(message)s'

EXPOSE 8000

#CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
CMD ["python", "main.py"]

# If running behind a proxy like Nginx or Traefik add --proxy-headers
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80", "--proxy-headers"]