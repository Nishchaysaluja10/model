Use a Python base image suitable for PyTorch and Torch Geometric

We use a slim bullseye image

FROM python:3.10-slim-bullseye

Set the working directory inside the container

WORKDIR /app

Copy the requirements file and install dependencies

COPY requirements.txt .

Install dependencies

NOTE: Installing torch-geometric and related packages might require system dependencies.

We include minimal build tools.

RUN apt-get update && apt-get install -y --no-install-recommends 

build-essential 

libxml2-dev 

&& pip install --no-cache-dir -r requirements.txt 

&& apt-get clean 

&& rm -rf /var/lib/apt/lists/*

Copy the rest of the application files

This includes main.py, best_gat_realistic.pth, and the CSV files

COPY . .

Expose the port the FastAPI app will run on

EXPOSE 8000

Command to run the application using Uvicorn

The format is: uvicorn [ASGI app]:[application instance] --host 0.0.0.0 --port [port]

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]