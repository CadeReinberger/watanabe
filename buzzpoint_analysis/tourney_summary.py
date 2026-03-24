import sqlite3
import pandas as pd


def get_tourney_data(tourney_name):
    
    # Make the sql connection to use this
    conn = sqlite3.connect('full_db.db')

    # Get the tournament dataframe and the id
    tourney_df = pd.read_sql('SELECT * FROM tournament', conn)
    tourney_id = list(tourney_df['name']).index(tourney_name) + 1
    tourney_row = tourney_df.loc[tourney_id-1]

    # Make the games dataframe
    games_df = pd.read_sql('SELECT * FROM game', conn)
    teams_df = pd.read_sql(f'SELECT * FROM team WHERE tournament_id = {tourney_id}', conn)
    games_df = pd.read_sql(f'SELECT * FROM game WHERE team_one_id IN {tuple(teams_df["id"])}', conn)

    # Let's just print the results
    print(games_df)

if __name__ == '__main__':
    get_tourney_data('2025 ACF Nationals')

