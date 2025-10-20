import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

kaggle_df = pd.read_csv("data/US_Accidents_clean.csv")

missing_values = kaggle_df.isnull().sum()
print(missing_values)


