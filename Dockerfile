#FROM python:3.9
#
##RUN apt-get update && apt-get -y install sudo gcc git g++ python3-dev build-essential \
##     && rm -rf /var/lib/apt/lists/*
#
#RUN echo 'deb http://deb.debian.org/debian testing main' >> /etc/apt/sources.list
#RUN apt-get update -y
#RUN apt-get install -y gcc
#RUN rm -rf /var/lib/apt/lists/*
#
#WORKDIR /app
#
#COPY requirements.txt .
#
#RUN pip install --no-cache-dir -r requirements.txt
#
#COPY . /app
#
#EXPOSE 80
#
#CMD ["streamlit", "run", "demo.py"]

FROM python:3.9

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True

RUN echo 'deb http://deb.debian.org/debian testing main' >> /etc/apt/sources.list
RUN apt-get update -y
RUN apt-get install -y gcc
RUN rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir cython plotly numpy pandas matplotlib tqdm scipy LunarCalendar holidays convertdate \
    setuptools setuptools-git python-dateutil wheel cmdstanpy
RUN pip install --no-cache-dir pystan==2.19.1.1 prophet fbprophet

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

