import datetime
import os
from math import ceil, floor

import numpy as np
import pandas as pd
from garmin_fit_sdk import Decoder, Stream
from matplotlib import pyplot as plt


def calc(record_mesgs):
    df = pd.DataFrame(record_mesgs)

    # select fields to keep
    fields_to_keep = [
        "timestamp",
        "distance",
        "heart_rate",
        "power",
        "enhanced_speed",
        "enhanced_altitude",
    ]
    df = df[fields_to_keep]

    # calculate speed in km/h
    df["speed_kmh"] = df["enhanced_speed"] * 3.6

    # calculate average power with a 5-second rolling window
    df["average_power"] = df["power"].rolling(window=5).mean()

    # calculate average speed with a 5-second rolling window
    df["average_speed"] = df["speed_kmh"].rolling(window=5).mean()

    # calculate average heart rate with a 5-second rolling window
    df["average_heart_rate"] = df["heart_rate"].rolling(window=5).mean().round(0)

    # calculate gradient for the past 30 seconds
    df["gradient"] = (
        df["enhanced_altitude"].diff(periods=30) / df["distance"].diff(periods=30)
    ) * 100

    # remove fields with an absolute gradient of more than 5%
    df = df[abs(df["gradient"]) < 5]

    # get unique values for heart rate
    heart_rate = df["average_heart_rate"].unique()
    # drop all heart rates below and above
    heart_rate = heart_rate[heart_rate > 115]
    heart_rate = heart_rate[heart_rate < 145]

    # get all values for a specific heart rate
    data_list = []
    for rate in heart_rate:
        data = df[df["average_heart_rate"] == rate]
        # sort by speed
        data_average_power = data.sort_values(by="average_power")
        # drop bottom 15% and top 15%
        data_average_power = data_average_power.iloc[
            int(len(data) * 0.15) : int(len(data) * 0.85)
        ]

        # sort by speed
        data_average_speed = data.sort_values(by="average_speed")
        # drop bottom 15% and top 15%
        data_average_speed = data_average_speed.iloc[
            int(len(data) * 0.15) : int(len(data) * 0.85)
        ]

        d = {
            "heart_rate": rate,
            "average_power": data_average_power["average_power"].mean(),
            "average_speed": data_average_speed["average_speed"].mean(),
        }
        data_list.append(d)

    # Convert averages to DataFrame
    averages_df = pd.DataFrame(data_list)
    # Sort by heart rate
    averages_df = averages_df.sort_values(by="heart_rate")
    # apply a rolling average to the data
    averages_df["average_power"] = averages_df["average_power"].rolling(window=5).mean()
    averages_df["average_speed"] = averages_df["average_speed"].rolling(window=5).mean()
    return averages_df


# Plotting
plt.figure(figsize=(15, 10))
minimum = 1000
maximum = 0

metric_to_plot = "average_power"  # average_power, average_speed
mapping = {
    "average_power": "Average power (W)",
    "average_speed": "Average speed (km/h)",
}

all_averages = []
for file_name in os.listdir(".dev"):
    if file_name.endswith(".fit"):
        stream = Stream.from_file(".dev/" + file_name)
        decoder = Decoder(stream)
        messages, errors = decoder.read()

        record_mesgs = messages.get("record_mesgs")
        time_created: datetime.datetime = messages.get("file_id_mesgs")[0].get(
            "time_created"
        )
        averages_df = calc(record_mesgs=record_mesgs)
        # remove all rows with NaN values
        averages_df = averages_df.dropna()
        if averages_df.empty:
            continue

        datapoint = {
            "averages_df": averages_df,
            "time_created": time_created,
            "file_name": file_name,
        }
        all_averages.append(datapoint)


time_created_list = []
for datapoint in all_averages:
    time_created = datapoint["time_created"]
    time_created_list.append(time_created)

# Convert time_created to datetime and normalize
time_created_dates = pd.to_datetime(time_created_list)
time_created_normalized = (time_created_dates - time_created_dates.min()) / (
    time_created_dates.max() - time_created_dates.min()
)
time_created_dict = dict(zip(time_created_dates, time_created_normalized))

for datapoint in all_averages:
    averages_df = datapoint["averages_df"]
    time_created = datapoint["time_created"]
    file_name = datapoint["file_name"]

    label = (
        file_name.replace("_ACTIVITY.fit", "") + "_" + time_created.strftime("%Y-%m-%d")
    )

    # Set grayscale color based on normalized time_created
    date = pd.to_datetime(time_created)
    color = 1 - time_created_dict[date]  # Invert to make newer data darker
    # rearrange to a scale of 0 to 0.8
    color = str(color * 0.8)

    plt.plot(
        averages_df["heart_rate"],
        averages_df[metric_to_plot],
        marker="o",
        label=label,
        color=color,
    )

    # Add label to the last point of the line in the plot
    last_point = averages_df.iloc[-1]
    plt.annotate(
        label,
        (last_point["heart_rate"], last_point[metric_to_plot]),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
        color=color,
    )

    minimum = min(minimum, averages_df[metric_to_plot].min())
    maximum = max(maximum, averages_df[metric_to_plot].max())

plt.title(f"{mapping[metric_to_plot]} for each Heart Rate")
plt.xlabel("Heart Rate")
plt.ylabel(mapping[metric_to_plot])
plt.grid(True)
plt.yticks(
    np.arange(
        floor(minimum / 10) * 10,
        ceil(maximum / 10) * 10 + 5,
        0.5 if metric_to_plot == "average_speed" else 5,
    )
)
plt.xticks(np.arange(115, 145, 1))
plt.legend()
plt.show()

exit()
