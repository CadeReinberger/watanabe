# Soccer — soccerdata

```python
import soccerdata as sd
```

soccerdata provides a unified interface to multiple data sources.

## Available leagues

| League | Code |
|--------|------|
| English Premier League | ENG-Premier League |
| La Liga (Spain) | ESP-La Liga |
| Bundesliga (Germany) | GER-Bundesliga |
| Serie A (Italy) | ITA-Serie A |
| Ligue 1 (France) | FRA-Ligue 1 |
| MLS (USA) | USA-Major League Soccer |
| Liga MX (Mexico) | MEX-Liga MX |
| Champions League | UEFA-Champions League |
| Europa League | UEFA-Europa League |
| FA Cup | ENG-FA Cup |
| Eredivisie (Netherlands) | NED-Eredivisie |
| Primeira Liga (Portugal) | POR-Primeira Liga |

## FBref source (best for statistics)

```python
# Create a reader for a league/season
fbref = sd.FBref(leagues=["ENG-Premier League", "USA-Major League Soccer"],
                 seasons=["2022-23", "2023-24"])
# For MLS use season year: seasons=["2023"]

# Read schedule and results
schedule = fbref.read_schedule()
# Columns: league_id, season_id, game_id, date, home_team, away_team,
#   home_score, away_score, venue, attendance, ...

# Player season stats
player_stats = fbref.read_player_season_stats(stat_type="standard")
# stat_type options: "standard", "shooting", "passing", "defense",
#   "possession", "misc", "keeper"
# Columns: player, nation, pos, team, age, mp (matches played),
#   starts, min, gls (goals), ast (assists), pk, pkatt, sh, sot, ...

# Team season stats
team_stats = fbref.read_team_season_stats(stat_type="standard")

# Match-level player stats
match_stats = fbref.read_player_match_stats(stat_type="standard")
# Columns: game_id, date, player, team, opponent, goals, assists, ...
```

## Sofascore source (alternative)

```python
sofascore = sd.SofaScore(leagues=["ENG-Premier League"], seasons=["2022-23"])
games = sofascore.read_games()
player_stats = sofascore.read_player_match_stats()
```

## ESPN source

```python
espn = sd.ESPN(leagues=["USA-Major League Soccer"], seasons=["2023"])
schedule = espn.read_schedule()
```

## Women's soccer

```python
# NWSL (National Women's Soccer League)
# FIFA Women's World Cup
fbref_w = sd.FBref(leagues=["USA-NWSL"], seasons=["2023"])
schedule = fbref_w.read_schedule()
stats = fbref_w.read_player_season_stats(stat_type="standard")

# NCAA Women's Soccer
fbref_ncaa = sd.FBref(leagues=["USA-NCAA"], seasons=["2023-24"])
```

## Day-of-week analysis

```python
import pandas as pd
schedule = fbref.read_schedule()
schedule["date"] = pd.to_datetime(schedule["date"])
schedule["day_of_week"] = schedule["date"].dt.day_name()
monday_games = schedule[schedule["day_of_week"] == "Monday"]
```

## Direct ESPN API (fallback for MLS / US Soccer)

```python
import requests

# MLS schedule
resp = requests.get(
    "https://site.api.espn.com/apis/site/v2/sports/soccer/usa.1/scoreboard",
    params={"dates": "20230901", "limit": 100}
)
data = resp.json()
events = data.get("events", [])

# Player stats search
resp = requests.get(
    "https://site.api.espn.com/apis/site/v2/sports/soccer/usa.1/athletes",
    params={"limit": 500}
)
```
