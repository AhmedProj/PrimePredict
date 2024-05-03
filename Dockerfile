FROM ubuntu:22.04
ENV ACCESS_KEY_ID=${{ secrets.ACCESS_KEY_ID }}
ENV SECRET=${{ secrets.SECRET }}
WORKDIR /PrimePredict 
# Install Python
RUN apt-get -y update && \
    apt-get install -y python3-pip
# Install project dependencies
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
RUN ln -s /usr/bin/python3 /usr/bin/python
EXPOSE 8000
CMD ["uvicorn", "app.api:app",  "--proxy-headers", "--host", "0.0.0.0", "--port", "8000"]
