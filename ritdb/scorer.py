import pandas as pd
import glob
import os

TARGET_FILE = 'STATS.xlsx'

class statline:
    def __init__(self, powers, tossups, negs):
        self.powers = powers
        self.tossups = tossups
        self.negs = negs

    def to_str(self):
        return str(self.powers) + '/' + str(self.tossups) + '/' + str(self.negs)

    def points(self):
        return 15 * self.powers + 10 * self.tossups - 5 * self.negs

    def to_dict(self):
        return {'scoreline': self.to_str(), 'points': self.points()}


class player:
    def __init__(self):
        self.stats = statline(0, 0, 0)

    def update(self, line):
        self.stats.powers += line.powers
        self.stats.tossups += line.tossups
        self.stats.negs += line.negs

    def to_dict(self):
        return {'scoreline': self.stats.to_str(), 'points': self.stats.points()}


def read_game(filepath):
    df = pd.read_excel(filepath, 'Main')
    res = {}

    df['Answerer'] = df['Answerer'].str.strip()
    df['Answerer'] = df['Answerer'].str.title()

    for ind in range(len(df)):
        plr = df['Answerer'].iloc[ind]
        if pd.isna(plr):
            continue
        if plr not in res:
            res[plr] = statline(0, 0, 0)

        try:
            score = int(df['Points'].iloc[ind])
            if score == -5:
                res[plr].negs += 1
            elif score == 10:
                res[plr].tossups += 1
            elif score == 15:
                res[plr].powers += 1
        except:
            pass

    return res


def make_game_summary(game_data):
    resdicts = {r: c.to_dict() for r, c in game_data.items()}
    resdf = pd.DataFrame.from_dict(resdicts).T
    return resdf.sort_values(by='points', ascending=False)


xlsx_files = [f for f in glob.glob('*.xlsx') if f != TARGET_FILE]

player_res = {}
game_res = {}

for filepath in xlsx_files:
    game_name = os.path.splitext(filepath)[0]
    game_data = read_game(filepath)
    game_res[game_name] = make_game_summary(game_data)

    for plr in game_data:
        if plr not in player_res:
            player_res[plr] = player()
        player_res[plr].update(game_data[plr])

plyrdicts = {r: c.to_dict() for r, c in player_res.items()}
player_res_df = pd.DataFrame.from_dict(plyrdicts).T
res_data = player_res_df.sort_values(by='points', ascending=False)

with pd.ExcelWriter(TARGET_FILE) as writer:
    res_data.to_excel(writer, sheet_name='FINAL')
    for name, df in game_res.items():
        df.to_excel(writer, sheet_name=name)
