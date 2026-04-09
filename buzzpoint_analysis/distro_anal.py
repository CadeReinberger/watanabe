import sqlite3
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np

# First, let's get the distribution from the database
conn = sqlite3.connect('full_db.db')

# First, grab the buzz dataframe
buzz_query = '''SELECT * FROM buzz WHERE value IN (10, 15)
                 AND (game_id, tossup_id) IN (
                    SELECT game_id, tossup_id FROM buzz
                    GROUP BY game_id, tossup_id
                    HAVING COUNT(*) = 1 ) '''
buzz_df = pd.read_sql('SELECT * FROM buzz WHERE value IN (10, 15)', conn)

# Next, get all the tossups so we can have them
tossup_df = pd.read_sql('SELECT id, question FROM tossup', conn)
df = buzz_df.merge(tossup_df, left_on='tossup_id', right_on='id', how='left', suffixes=("", "_lookup"))

# Now, we compute the celerity of all the correct buzzes
df['celerity'] = np.clip(1 - df['buzz_position'] / df['question'].str.split(r'\s+').str.len(), 0, 1)

# Okay, do the actual plotting
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(df['celerity'].dropna(), bins=30, color='steelblue', edgecolor='white', linewidth=0.5)
ax.set_xlabel('Celerity', fontsize=13)
ax.set_ylabel('Count', fontsize=13)
ax.set_title('Distribution of Buzz Celerity', fontsize=15)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
plt.savefig('celerity_hist.png', dpi=150)
