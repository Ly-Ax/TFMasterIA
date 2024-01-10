FROM tiangolo/uvicorn-gunicorn-fastapi

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN mkdir -p app
COPY ./app app

# docker build -t app-mia-tfm .
# docker run -d --name mia-tfm-container -p 8080:80 app-mia-tfm
# http://localhost:8080