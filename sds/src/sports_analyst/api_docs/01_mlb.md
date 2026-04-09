# MLB — pybaseball

```python
import pybaseball
import pandas as pd
pybaseball.cache.enable()   # strongly recommended — caches API results to disk
```

## Season aggregated batting / pitching stats (FanGraphs)

```python
# Batting stats — returns DataFrame with one row per player per season
# qual: minimum plate appearances (use 0 for all players)
df = pybaseball.batting_stats(start_season=2000, end_season=2023, qual=0)
# Columns include: Name, Team, G, AB, PA, H, 1B, 2B, 3B, HR, R, RBI, BB,
#   IBB, SO, HBP, SF, SH, GDP, SB, CS, AVG, OBP, SLG, OPS, ...

df = pybaseball.pitching_stats(start_season=2000, end_season=2023, qual=0)
# Columns: Name, Team, W, L, ERA, G, GS, CG, ShO, SV, IP, H, R, ER, HR,
#   BB, IBB, HBP, WP, SO, ...
```

## Baseball Reference season stats

```python
df = pybaseball.batting_stats_bref(season=2022)
# Columns: Name, Age, Tm, Lg, G, PA, AB, R, H, 2B, 3B, HR, RBI, SB, CS,
#   BB, SO, BA, OBP, SLG, OPS, ...

df = pybaseball.pitching_stats_bref(season=2022)
```

## Game-level Statcast data (2015–present)

```python
# All batted balls / pitch-level data for a date range
# IMPORTANT: keep date ranges short (<1 week) to avoid timeouts
df = pybaseball.statcast(start_dt="2023-04-01", end_dt="2023-04-07")
# Key columns: game_date, batter, pitcher, events, description,
#   bb_type, launch_speed, launch_angle, hit_distance_sc,
#   home_team, away_team, inning, ...

# Per-batter Statcast (use player ID from playerid_lookup)
df = pybaseball.statcast_batter(start_dt="2022-04-01", end_dt="2022-09-30",
                                player_id=116539)  # Mike Trout example

# Per-pitcher Statcast
df = pybaseball.statcast_pitcher(start_dt="2022-04-01", end_dt="2022-09-30",
                                 player_id=592789)
```

## Player ID lookup

```python
# Returns DataFrame with columns: name_last, name_first, key_mlbam, key_retro,
#   key_bbref, key_fangraphs, mlb_played_first, mlb_played_last
results = pybaseball.playerid_lookup("Aaron", "Hank")
player_id_mlbam = results["key_mlbam"].iloc[0]
player_id_bbref = results["key_bbref"].iloc[0]   # e.g. "aaronha01"
```

## Baseball Reference player game logs (game-by-game, best for day-of-week!)

pybaseball does NOT have a per-player game log function. Use direct HTTP scraping instead:

```python
import requests
import pandas as pd
from io import StringIO
import time

def get_player_game_logs_bref(bbref_id: str, years: list[int], position: str = "b") -> pd.DataFrame:
    """
    Fetch player game-by-game logs from Baseball Reference.
    position: 'b' = batting, 'p' = pitching
    Returns combined DataFrame across all years.
    """
    all_games = []
    headers = {"User-Agent": "Mozilla/5.0"}
    for year in years:
        url = "https://www.baseball-reference.com/players/gl.fcgi"
        params = {"id": bbref_id, "t": position, "year": str(year)}
        try:
            r = requests.get(url, params=params, headers=headers, timeout=15)
            tables = pd.read_html(StringIO(r.text))
            # Table index 4 is the game log (shape ~[N, 40])
            game_table = next((t for t in tables if "HR" in t.columns and "Date" in t.columns), None)
            if game_table is not None:
                game_table = game_table[game_table["Date"] != "Date"]  # drop header rows
                game_table["season"] = year
                all_games.append(game_table)
        except Exception:
            pass
        time.sleep(0.5)  # be polite to BBref
    if not all_games:
        return pd.DataFrame()
    df = pd.concat(all_games, ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"].astype(str).str[:10], errors="coerce")
    return df.dropna(subset=["Date"])

# Example: Hank Aaron career (1954–1976)
# First get player ID
hank = pybaseball.playerid_lookup("Aaron", "Hank")
bbref_id = hank["key_bbref"].iloc[0]   # "aaronha01"

df = get_player_game_logs_bref(bbref_id, list(range(1954, 1977)), position="b")
# Columns: Rk, Gcar, Gtm, Date, Team, Opp, Result, Inngs, PA, AB, R, H,
#   2B, 3B, HR, RBI, SB, CS, BB, SO, BA, OBP, SLG, OPS, TB, GIDP,
#   HBP, SH, SF, IBB, BAbip, aLI, WPA, cWPA, RE24, BOP, Pos, season

df["DayOfWeek"] = df["Date"].dt.day_name()
sundays = df[df["DayOfWeek"] == "Sunday"]
sunday_hr = pd.to_numeric(sundays["HR"], errors="coerce").fillna(0).sum()
print(f"Home runs on Sundays: {int(sunday_hr)}")
```

## Team batting/pitching

```python
df = pybaseball.team_batting(start_season=2000, end_season=2023)
df = pybaseball.team_pitching(start_season=2000, end_season=2023)
```

## Schedule and results

```python
# season: int year, team: 3-letter abbreviation (e.g. "NYY", "BOS", "ATL")
df = pybaseball.schedule_and_record(season=2022, team="NYY")
# Columns: Date, Tm, Home_Away, Opp, W/L, R, RA, Inn, W-L, Rank, GB,
#   Win, Loss, Save, Time, D/N, Attendance, cLI, ...
```

## Historical standings

```python
df = pybaseball.standings(season=2022)  # returns list of DataFrames by division
```

## Notable player IDs (Baseball Reference)
- Hank Aaron: "aaronha01"
- Babe Ruth: "ruthba01"
- Willie Mays: "mayswi01"
- Barry Bonds: "bondsba01"
- Mike Trout: "troutmi01"

Use `pybaseball.playerid_lookup(last, first)` for any player not listed here.
