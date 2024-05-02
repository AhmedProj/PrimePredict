FROM ubuntu:22.04
WORKDIR /PrimePredictN 
# Install Python
RUN apt-get -y update && \
    apt-get install -y python3-pip
# Install project dependencies
COPY requirements.txt .
COPY src ./src
COPY app ./api/
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8000
CMD ["uvicorn", "app.api:app",  "--proxy-headers", "--host", "0.0.0.0", "--port", "8000"]
