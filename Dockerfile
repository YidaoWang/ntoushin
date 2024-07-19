FROM python:3.10-slim

WORKDIR /src

COPY src src

RUN pip install --no-cache-dir fastapi uvicorn

EXPOSE 80

CMD ["uvicorn","src.main:app","--host","0.0.0.0","--port","80"]
