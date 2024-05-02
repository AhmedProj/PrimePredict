import os
import s3fs
import pandas as pd

# Create filesystem object
S3_ENDPOINT_URL = "https://" + os.environ["AWS_S3_ENDPOINT"]
BUCKET = "ahmed"
FILE_PATH_S3 = BUCKET + "/diffusion/"


fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": S3_ENDPOINT_URL})


def load_csv(file="training.csv"):
    with fs.open(FILE_PATH_S3 + file, mode="rb") as file:
        df = pd.read_csv(file, sep=";")
    return df


def send_csv(df):
    with fs.open(FILE_PATH_S3, "w") as file:
        df.to_csv(file)
