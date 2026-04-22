import sqlite3
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
from fitter import Fitter

conn = sqlite3.connect('full_db.db')
buzz_df = pd.read_sql('SELECT * FROM buzz WHERE value IN (10, 15)', conn)
tossup_df = pd.read_sql('SELECT id, question FROM tossup', conn)
df = buzz_df.merge(tossup_df, left_on='tossup_id', right_on='id', how='left', suffixes=("", "_lookup"))
df['celerity'] = np.clip(1 - df['buzz_position'] / df['question'].str.split(r'\s+').str.len(), 0, 1)

celerity = df['celerity'].dropna().values

f = Fitter(celerity)
f.fit()

summary = f.summary()
summary.to_csv('fittr_summary.csv')

fig = f.plot_pdf(Nbest=5)
import matplotlib.pyplot as plt
plt.savefig('fittr_summary.png', dpi=150, bbox_inches='tight')
