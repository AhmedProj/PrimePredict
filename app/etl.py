import sys
import os
from pathlib import Path

PATH = str(Path(os.path.split(__file__)[0]).parent)
sys.path.insert(1, PATH + "/src")

import re
import time
import pyarrow as pa
import pyarrow.parquet as pq
from data.data_load import send_parquet

def extract_transform_load(log_file, file_name):
    predictions = []
    probabilities = []
    with open(log_file, 'r') as file:
        for line in file:
            match = re.search(r'prediction: (\d+\.\d+) probability: (\d+\.\d+), (\d+\.\d+)', line)
            if match:
                prediction = float(match.group(1))
                probability1 = float(match.group(2))
                probability2 = float(match.group(3))
                predictions.append(prediction)
                probabilities.append((probability1, probability2))
    table = pa.Table.from_arrays([pa.array(predictions), pa.array(probabilities)], names=['predictions', 'probabilities'])
    send_parquet(table, file_name)
    return predictions, probabilities

if __name__ == "__main__":

    LOG_FILE = 'log_file.log'
    # Initialize previous line count
    previous_line_count = 5

    while True:
        # Get current line count
        with open(LOG_FILE, 'r') as file:
            current_line_count = sum(1 for line in file)
        # If there are new lines, process the file and upload to S3
        if current_line_count > previous_line_count:
            predictions, probabilities = extract_transform_load(LOG_FILE, 'output.parquet')

            # Update previous line count
            previous_line_count += current_line_count

        # Sleep for 2 minutes before checking again
        time.sleep(120)