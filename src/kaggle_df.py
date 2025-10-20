from kagglehub import KaggleDatasetAdapter
import kagglehub
import os

dataset_name = "sobhanmoosavi/us-accidents"
file_name = "US_Accidents_March23.csv"

df = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    dataset_name,
    file_name
)

os.makedirs("data", exist_ok=True)
df.to_csv("data/US_Accidents_clean.csv", index=False)