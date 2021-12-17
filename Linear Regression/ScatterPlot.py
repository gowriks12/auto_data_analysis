import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("Auto.data")
df.replace({"?":np.nan},inplace=True)
df.dropna()

fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(x=df['mpg'], y=df['horsepower'])
plt.xlabel("mpg")
plt.ylabel("Horse power")

plt.show()