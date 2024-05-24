from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np
import os
import pandas as pd

def generate_graph_seq2seq_io_data(
        df_x, df_y, add_time_in_day=True, scaler=None
):
    """
    Generate samples from
    :param df:
    :param x_offsets:
    :param y_offsets:
    :param scaler:
    :return:
    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)
    """

    num_samples, num_nodes = df_x.shape
    print(num_samples, num_nodes)
    data_x = np.expand_dims(df_x.values, axis=-1)
    feature_list_x = [data_x]
    data_y = np.expand_dims(df_y.values, axis=-1)
    feature_list_y = [data_y]

    if add_time_in_day:
        time_ind_x = (df_x.index.values - df_x.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        time_in_day_x = np.tile(time_ind_x, [1, num_nodes, 1]).transpose((2, 1, 0))
        feature_list_x.append(time_in_day_x)

        time_ind_y = (df_y.index.values - df_y.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        time_in_day_y = np.tile(time_ind_y, [1, num_nodes, 1]).transpose((2, 1, 0))
        feature_list_y.append(time_in_day_y)

    data_x = np.concatenate(feature_list_x, axis=-1)
    data_y = np.concatenate(feature_list_y, axis=-1)

    x, y = [], []
    x.append(data_x)
    y.append(data_y)
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    return x, y

def generate_train_val_test(args):

    # df_x is data x for prediction
    # df_y is valid y for comparing prediction and valid
    df_x = pd.read_hdf(args.traffic_df_filename_x)
    df_y = pd.read_hdf(args.traffic_df_filename_y)

    # df_x
    filtered_df_x = df_x.iloc[0:12]

    # df_y
    start_time = filtered_df_x.index[-1] + pd.Timedelta(minutes=5)
    end_time = start_time + pd.Timedelta(hours=1) - pd.Timedelta(minutes=5)
    filtered_df_y = df_y.loc[start_time:end_time]

    print("x: (from ", filtered_df_x.index[0], " to ", filtered_df_x.index[-1], ")")
    print("y: (from ", filtered_df_y.index[0], " to ", filtered_df_y.index[-1], ")")
    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)
    x_test, y_test = generate_graph_seq2seq_io_data(
        filtered_df_x,
        filtered_df_y,
        add_time_in_day=True,
    )

    print("x shape: ", x_test.shape, ", y shape: ", y_test.shape)
    # Write the data into npz file.
    cat = "test"
    np.savez_compressed(
        os.path.join(args.output_dir, f"{cat}.npz"),
        x=x_test,
        y=y_test,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/VMS", help="Output directory.")
    parser.add_argument("--traffic_df_filename_x", type=str, default="data/vms_valid_data.h5", help="Raw traffic readings.",)
    parser.add_argument("--traffic_df_filename_y", type=str, default="data/vms_valid_data.h5", help="Raw traffic readings.",)

    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
      os.makedirs(args.output_dir)
    generate_train_val_test(args)