# NBA and WNBA — nba_api

```python
import time
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import (
    playercareerstats,
    playergamelog,
    leaguedashplayerstats,
    teamgamelog,
    commonplayerinfo,
    leaguegamelog,
    playerdashboardbyyearoveryear,
    shotchartdetail,
)
```

**Always add `time.sleep(1)` between API calls to avoid rate limiting.**

## Player lookup

```python
# Find a player by name (returns list of dicts)
all_players = players.get_players()
# Each dict: {'id': int, 'full_name': str, 'first_name': str,
#              'last_name': str, 'is_active': bool}

matches = [p for p in all_players if "lebron" in p["full_name"].lower()]
player_id = matches[0]["id"]  # e.g. 2544 for LeBron James

# Common lookup IDs
# LeBron James: 2544
# Michael Jordan: 893
# Kobe Bryant: 977
# Stephen Curry: 201939
# Kevin Durant: 201142
# Shaquille O'Neal: 406
```

## Player game log (game-by-game, great for day-of-week queries)

```python
# season format: "2022-23"
# season_type_all_star: "Regular Season", "Playoffs", "All Star"
log = playergamelog.PlayerGameLog(
    player_id=2544,
    season="2022-23",
    season_type_all_star="Regular Season",
)
df = log.get_data_frames()[0]
# Columns: SEASON_ID, Player_ID, Game_ID, GAME_DATE, MATCHUP, WL,
#   MIN, FGM, FGA, FG_PCT, FG3M, FG3A, FG3_PCT, FTM, FTA, FT_PCT,
#   OREB, DREB, REB, AST, STL, BLK, TOV, PF, PTS, PLUS_MINUS

# GAME_DATE format: "DEC 25, 2022" — convert with:
import pandas as pd
df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
df["DayOfWeek"] = df["GAME_DATE"].dt.day_name()
```

## Player career stats

```python
career = playercareerstats.PlayerCareerStats(player_id=2544)
dfs = career.get_data_frames()
# dfs[0]: season-by-season regular season totals
# Columns: PLAYER_ID, SEASON_ID, LEAGUE_ID, TEAM_ID, TEAM_ABBREVIATION,
#   PLAYER_AGE, GP, GS, MIN, FGM, FGA, FG_PCT, FG3M, FG3A, FG3_PCT,
#   FTM, FTA, FT_PCT, OREB, DREB, REB, AST, STL, BLK, TOV, PF, PTS
```

## Season leader boards

```python
stats = leaguedashplayerstats.LeagueDashPlayerStats(
    season="2022-23",
    season_type_all_star="Regular Season",
    per_mode_simple="PerGame",   # or "Totals"
)
df = stats.get_data_frames()[0]
# Columns: PLAYER_ID, PLAYER_NAME, TEAM_ID, TEAM_ABBREVIATION, AGE,
#   GP, W, L, W_PCT, MIN, FGM, FGA, FG_PCT, FG3M, FG3A, FG3_PCT,
#   FTM, FTA, FT_PCT, OREB, DREB, REB, AST, TOV, STL, BLK, BLKA,
#   PF, PFD, PTS, PLUS_MINUS, ...
```

## Team game log

```python
# Team IDs: Lakers=1610612747, Warriors=1610612744, Celtics=1610612738
all_teams = teams.get_teams()
# Each dict: {'id': int, 'full_name': str, 'abbreviation': str,
#              'nickname': str, 'city': str, 'state': str, 'year_founded': int}

tgl = teamgamelog.TeamGameLog(team_id=1610612747, season="2022-23")
df = tgl.get_data_frames()[0]
# Columns: Team_ID, Game_ID, GAME_DATE, MATCHUP, WL, W, L, W_PCT,
#   MIN, FGM, FGA, FG_PCT, FG3M, FG3A, FG3_PCT, FTM, FTA, FT_PCT,
#   OREB, DREB, REB, AST, STL, BLK, TOV, PF, PTS, BLK, DREB, ...
```

## WNBA

Use the same `nba_api` library with `league_id="10"` (WNBA) parameter:

```python
from nba_api.stats.endpoints import leaguedashplayerstats

wnba_stats = leaguedashplayerstats.LeagueDashPlayerStats(
    league_id="10",       # "00" = NBA, "10" = WNBA, "20" = G League
    season="2023",
    season_type_all_star="Regular Season",
)
df = wnba_stats.get_data_frames()[0]

# Player game log for WNBA player
from nba_api.stats.endpoints import playergamelog
log = playergamelog.PlayerGameLog(
    player_id=...,
    season="2023",
    league_id="10",
)

# WNBA player lookup
wnba_players = players.get_players()
# Filter by is_active or search by name; WNBA players are in the same list
```
