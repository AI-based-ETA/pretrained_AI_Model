import datetime
import argparse
import numpy as np
import os
import pandas as pd

def generate_test_df(args):
    df = pd.read_hdf(args.traffic_df_filename)
    end_time = pd.to_datetime(args.date)    
    start_time = end_time - pd.Timedelta(hours=1) + pd.Timedelta(minutes=5)
    filtered_df = df.loc[start_time:end_time]
    print("generate_test_df: ", filtered_df.shape)
    filtered_df.to_hdf("data/vms_test_data.h5", key='df')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--traffic_df_filename", type=str, default="data/vms_valid_data.h5", help="Raw traffic readings.",)
    parser.add_argument('--date', type=lambda s: datetime.datetime.strptime(s, '%Y-%m-%d %H:%M'), default="2024-04-20 00:00", help="Date format 'YYYY-MM-DD HH:MM'")

    args = parser.parse_args()
    generate_test_df(args)