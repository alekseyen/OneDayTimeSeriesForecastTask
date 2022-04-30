FROM python:3.9

RUN apt-get update && apt-get -y install sudo gcc git g++ python3-dev build-essential \
     && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 80

CMD ["streamlit", "run", "demo.py"]
