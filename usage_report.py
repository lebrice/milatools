"""Analyze the milatools usage based on the number of jobs called 'mila-{command}'."""

from __future__ import annotations

import os
import pickle
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd  # for the DataFrame's nice output in notebooks

SARC_DIR = Path.home() / "repos" / "SARC"
# Remember to set up the port forwarding if you want
# to access data from SARC.
#    ssh -L 27017:localhost:27017 sarc
# Change this to the path to your config file.
os.environ["SARC_CONFIG"] = str(SARC_DIR / "config/sarc-client.json")

from sarc.config import MTL  # noqa: E402
from sarc.jobs import SlurmJob, get_jobs  # noqa: E402

"""
Basic data retrieval and processing functions.
"""


def retrieve_data(
    start: datetime,
    end: datetime,
    cached_results_path: str,
    cluster_name: str | None = None,
) -> list[SlurmJob]:
    if os.path.exists(cached_results_path):
        print(f"Reading from {cached_results_path}.")
        with open(cached_results_path, "rb") as f_input:
            L_jobs = pickle.load(f_input)
    else:
        print(f"Retrieving {cached_results_path}.")
        time_start = time.time()
        L_jobs = preprocess_and_filter_jobs(
            get_jobs(start=start, end=end, cluster=cluster_name), start, end
        )
        query_duration = time.time() - time_start
        print(f"Query duration: {query_duration} seconds for {len(L_jobs)} jobs.")

        print(f"Writing to {cached_results_path}.")
        with open(cached_results_path, "wb") as f_output:
            pickle.dump(L_jobs, f_output, pickle.HIGHEST_PROTOCOL)

    return L_jobs


def preprocess_and_filter_jobs(
    L_jobs: Iterable[SlurmJob], start: datetime, end: datetime
) -> list[SlurmJob]:
    """Just a simple filter to regulate certain jobs coming from SARC that might not
    have a proper start time or end time."""

    LD_jobs_output = []

    for job in L_jobs:
        if job.elapsed_time <= 0:
            continue

        if job.end_time is None:
            job.end_time = datetime.now(tz=MTL)

        # For some reason start time is not reliable, often equal to submit time,
        # so we infer it based on end_time and elapsed_time.
        job.start_time = job.end_time - timedelta(seconds=job.elapsed_time)

        # Clip the job to the time range we are interested in.
        if job.start_time < start:
            job.start_time = start
        if job.end_time > end:
            job.end_time = end
        job.elapsed_time = int((job.end_time - job.start_time).total_seconds())

        # We only care about jobs that actually ran.
        if job.elapsed_time <= 0:
            continue

        # LD_jobs_output.append(job.json())
        LD_jobs_output.append(job)

    return LD_jobs_output


# %% [markdown]
# ## The approach taken
#
# bla bla

# %%
date_start = datetime(year=2024, month=1, day=1, tzinfo=MTL)
date_end = datetime(year=2025, month=1, day=1, tzinfo=MTL)

# Don't forget to `ssh -L 27017:localhost:27017 sarc` before running this
# or else you won't be able to connect to the SARC database.
L_jobs = retrieve_data(
    date_start,
    date_end,
    cached_results_path="milatools_jobs_cache.pkl",
)


L_jobs[0].start_time

# %%
# job = L_jobs[0]
# print(job)
# print(job.name)

S_unique_job_names = {
    job.name for job in L_jobs if ("mila" in job.name and "code" in job.name)
}
print(sorted(list(S_unique_job_names)))
# ['mila-tools', 'milatools_test']

L_milacode_jobs = [job for job in L_jobs if job.name == "mila-code"]
print(f"We have {len(L_milacode_jobs)} mila-code jobs.")

nbr_jobs_over_30_minutes = len(
    list(job for job in L_milacode_jobs if job.duration.total_seconds() > 30 * 60)
)
print(nbr_jobs_over_30_minutes)
print(f"We have {nbr_jobs_over_30_minutes} mila-code jobs over 30 minutes.")

print(list(job.duration.total_seconds() for job in L_milacode_jobs))
print(list(job.user for job in L_milacode_jobs))

# %%

L_milacode_jobs_over_10_minutes = [
    job
    for job in L_jobs
    if job.name == "mila-code" and job.duration.total_seconds() >= 10 * 60
]

df = pd.DataFrame(
    [job.start_time for job in L_milacode_jobs_over_10_minutes], columns=["Timestamp"]
)

# df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df.set_index("Timestamp", inplace=True)
daily_counts = df.resample("D").size()
daily_counts.index = daily_counts.index.strftime("%Y-%m-%d")

fig, ax = plt.subplots()
daily_counts.plot(kind="bar", ax=ax)
# ax.set_xlabel('Date')
ax.set_ylabel("Number of jobs")
ax.set_title('Number "mila-code" jobs over 10 minutes in duration')

ticks_to_show = daily_counts.index[::5]  # Show every 5th label
ax.set_xticks(range(len(daily_counts.index)))  # Set all possible x-tick positions
ax.set_xticklabels(daily_counts.index, rotation=45)  # Apply all labels with rotation
ax.set_xticklabels(
    [label if label in ticks_to_show else "" for label in daily_counts.index]
)  # Hide non-selected labels
# plt.tight_layout()
plt.show()

# Plot the histogram
# daily_counts.plot(kind='bar')
# plt.xlabel('Day')
# plt.ylabel('Number of Items')
# plt.title('Items by Day')
# plt.xticks(rotation=45)  # Rotate labels to improve readability
# plt.show()


L_milacode_jobs_over_10_minutes = [
    job
    for job in L_jobs
    if job.name == "mila-code" and job.duration.total_seconds() >= 10 * 60
]

# Let's not care about the fact that usernames might differ between clusters.
# That would be perfectionism at this early exploratory moment.
SP_unique_users_each_day = {
    (job.user, job.start_time.strftime("%Y-%m-%d"))
    for job in L_milacode_jobs_over_10_minutes
}

L_jobs_unique_user_per_day = [start_day for user, start_day in SP_unique_users_each_day]

df = pd.DataFrame(L_jobs_unique_user_per_day, columns=["Date"])
df["Date"] = pd.to_datetime(df["Date"])

# Count occurrences by day
# Since each row is an occurrence, we can simply use value_counts(), which automatically groups by unique values and counts them
daily_counts = (
    df["Date"].value_counts().sort_index()
)  # sort_index() ensures the dates are in chronological order
daily_counts.index = daily_counts.index.strftime("%Y-%m-%d")

fig, ax = plt.subplots()
daily_counts.plot(kind="bar", ax=ax)
# ax.set_xlabel('Date')
ax.set_ylabel("Number of unique users with jobs")
ax.set_title('Number unique users with "mila-code" jobs over 10 minutes in duration')

ticks_to_show = daily_counts.index[::5]  # Show every 5th label
ax.set_xticks(range(len(daily_counts.index)))  # Set all possible x-tick positions
ax.set_xticklabels(daily_counts.index, rotation=45)  # Apply all labels with rotation
ax.set_xticklabels(
    [label if label in ticks_to_show else "" for label in daily_counts.index]
)  # Hide non-selected labels
# plt.tight_layout()
plt.show()
