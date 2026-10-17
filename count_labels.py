import pandas as pd

# CHANGE THIS PATH if your CSV is elsewhere
CSV_PATH = "complaints_text_dataset.csv"

df = pd.read_csv(CSV_PATH)

# Show total rows
print("\nTotal samples:", len(df))

# Count per class
print("\nSamples per class:\n")
print(df["label"].value_counts())

# Optional: sorted output
print("\nSorted counts:\n")
print(df["label"].value_counts().sort_index())
