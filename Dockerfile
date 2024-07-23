FROM animcogn/face_recognition:cpu

WORKDIR /app

COPY . /app

RUN pip3 install -r requirements.txt

EXPOSE 80

CMD ["uvicorn","src.main:app","--host","0.0.0.0","--port","80"]
