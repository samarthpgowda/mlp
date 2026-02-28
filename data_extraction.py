import pandas as pd
import os
import re

# -------------------------
# BASE PATH
# -------------------------
BASE_PATH = r"C:\Users\Samarth\Desktop\fpga hackathon\data"


SIGNALS_FOLDER = os.path.join(BASE_PATH, "signals")
ANNOTATIONS_FOLDER = os.path.join(BASE_PATH, "annotations")
OUTPUT_FOLDER = os.path.join(BASE_PATH, "extracted_beats")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# -------------------------
# PARAMETERS
# -------------------------
WINDOW_BEFORE = 120
WINDOW_AFTER = 120
TOTAL_SAMPLES = WINDOW_BEFORE + WINDOW_AFTER + 1

# -------------------------
# PROCESS SIGNAL FILES
# -------------------------
for signal_file in os.listdir(SIGNALS_FOLDER):

    if not signal_file.endswith(".csv"):
        continue

    # Extract record number (e.g., 100 from 100_ekg.csv)
    record_match = re.match(r"(\d+)_ekg", signal_file)
    if not record_match:
        continue

    record_number = record_match.group(1)

    annotation_file = f"{record_number}_annotations_1.csv"

    signal_path = os.path.join(SIGNALS_FOLDER, signal_file)
    annotation_path = os.path.join(ANNOTATIONS_FOLDER, annotation_file)

    if not os.path.exists(annotation_path):
        print(f"Annotation missing for {record_number}, skipping.")
        continue

    print(f"Processing record {record_number}...")

    # -------------------------
    # LOAD FILES
    # -------------------------
    signal_df = pd.read_csv(signal_path, header=None, low_memory=False)
    annotation_df = pd.read_csv(annotation_path, header=None, low_memory=False)

    signal_df.columns = ["index", "MVII", "V5", "symbol"]
    annotation_df.columns = ["index", "symbol"]

    # -------------------------
    # CLEAN DATA
    # -------------------------
    signal_df["index"] = pd.to_numeric(signal_df["index"], errors="coerce")
    signal_df["MVII"] = pd.to_numeric(signal_df["MVII"], errors="coerce")
    annotation_df["index"] = pd.to_numeric(annotation_df["index"], errors="coerce")

    signal_df = signal_df.dropna(subset=["index", "MVII"])
    annotation_df = annotation_df.dropna(subset=["index"])

    signal_df["index"] = signal_df["index"].astype(int)
    annotation_df["index"] = annotation_df["index"].astype(int)

    annotation_df["symbol"] = annotation_df["symbol"].astype(str).str.strip()

    # Keep only N and A
    annotation_df = annotation_df[annotation_df["symbol"].isin(["N", "A"])]

    # -------------------------
    # MATCH USING INDEX COLUMN
    # -------------------------
    signal_df.set_index("index", inplace=True)
    signal_series = signal_df["MVII"]

    beats = []
    beat_number = 1

    # -------------------------
    # EXTRACT BEATS
    # -------------------------
    for _, row in annotation_df.iterrows():
        peak_index = row["index"]
        label = row["symbol"]

        start = peak_index - WINDOW_BEFORE
        end = peak_index + WINDOW_AFTER

        if start in signal_series.index and end in signal_series.index:
            segment = signal_series.loc[start:end].values

            if len(segment) == TOTAL_SAMPLES:
                beat_row = [beat_number] + list(segment) + [label]
                beats.append(beat_row)
                beat_number += 1

    # -------------------------
    # SAVE OUTPUT
    # -------------------------
    if beats:
        columns = (
            ["Beat"]
            + [f"Sample_{i}" for i in range(TOTAL_SAMPLES)]
            + ["Label"]
        )

        beats_df = pd.DataFrame(beats, columns=columns)

        output_name = f"{record_number}_beats.csv"
        output_path = os.path.join(OUTPUT_FOLDER, output_name)

        beats_df.to_csv(output_path, index=False)

        print(f"Saved {output_name} | Beats: {len(beats_df)}")
    else:
        print(f"No valid beats found for {record_number}")

print("All files processed.")

