# NCAA Sports — cfbd + sportsipy

---

## NCAA Football (Men) — cfbd (College Football Data API)

```python
import cfbd
import os

# Configure with your free API key from https://collegefootballdata.com/
# Set environment variable CFBD_API_KEY or pass directly
configuration = cfbd.Configuration()
configuration.api_key["Authorization"] = os.environ.get("CFBD_API_KEY", "")
configuration.api_key_prefix["Authorization"] = "Bearer"

api_client = cfbd.ApiClient(configuration)
```

### Games

```python
games_api = cfbd.GamesApi(api_client)

# All games for a season
games = games_api.get_games(year=2023)
# Each game: id, season, week, season_type, start_date, home_team, away_team,
#   home_points, away_points, home_line_scores, away_line_scores, ...

# Games for a specific team
games = games_api.get_games(year=2023, team="Alabama")
```

### Player stats

```python
stats_api = cfbd.StatsApi(api_client)

# Season stats for all players
player_stats = stats_api.get_player_season_stats(year=2023)
# Each: player_id, player, team, conference, category, stat_type, stat

# Game-by-game player stats
player_game_stats = stats_api.get_player_game_stats(year=2023, week=1)
```

### Team stats

```python
team_stats = stats_api.get_team_season_stats(year=2023)
team_stats = stats_api.get_advanced_team_season_stats(year=2023)
```

### Recruiting

```python
recruiting_api = cfbd.RecruitingApi(api_client)
recruits = recruiting_api.get_recruiting_players(year=2023)
```

### Rosters

```python
rosters_api = cfbd.PlayersApi(api_client)
roster = rosters_api.get_roster(team="Alabama", year=2023)
```

---

## NCAA Basketball (Men & Women), Baseball, Softball, Hockey — sportsipy

```python
import time
# Add time.sleep(2) between requests to avoid rate limiting

# ---- MEN'S BASKETBALL ----
from sportsipy.ncaab.teams import Teams as NCAABTeams

teams = NCAABTeams(year="2023")
for team in teams:
    print(team.name, team.wins, team.losses)

from sportsipy.ncaab.roster import Roster as NCAABRoster
roster = NCAABRoster("DUKE", year="2023")
for player in roster.players:
    print(player.name, player.points_per_game)

# ---- WOMEN'S BASKETBALL ----
from sportsipy.ncaaw.teams import Teams as NCAAWTeams
from sportsipy.ncaaw.roster import Roster as NCAAWRoster

teams = NCAAWTeams(year="2023")
# Team attributes: wins, losses, points, opponents_points, ...

# ---- MEN'S BASEBALL ----
from sportsipy.ncaab.teams import Teams as NCAABBTeams   # Note: use ncaab module for baseball too? No:
# Baseball is accessed via:
from sportsipy.ncaab.teams import Teams   # This is basketball only
# For baseball:
from sportsipy.ncaab.schedule import Schedule as NCAABSchedule

# ---- MEN'S HOCKEY ----
from sportsipy.ncaaf.teams import Teams as NCAAFTeams   # football
```

### sportsipy NCAA Basketball full example

```python
import time
from sportsipy.ncaab.teams import Teams

# Get all teams for a season
teams = Teams(year="2023")   # "2023" means 2022-23 season

# Access a specific team
for team in teams:
    if "Duke" in team.name:
        print(f"Wins: {team.wins}, Losses: {team.losses}")
        print(f"Points per game: {team.points_per_game}")
        break

# Get schedule for a team (game-by-game)
from sportsipy.ncaab.schedule import Schedule
schedule = Schedule("DUKE", year="2023")
for game in schedule:
    print(game.date, game.opponent_name, game.points_scored, game.points_allowed)
    time.sleep(1)

# Get roster and player stats
from sportsipy.ncaab.roster import Roster
roster = Roster("DUKE", year="2023")
for player in roster.players:
    print(player.name, player.games_played, player.points_per_game)
    time.sleep(1)
```

### sportsipy Women's Basketball example

```python
from sportsipy.ncaaw.teams import Teams
from sportsipy.ncaaw.schedule import Schedule
from sportsipy.ncaaw.roster import Roster

teams = Teams(year="2023")
schedule = Schedule("CONNECTICUT", year="2023")
roster = Roster("CONNECTICUT", year="2023")
```

### sportsipy common team abbreviations
NCAA uses school abbreviations — search with:
```python
for team in Teams(year="2023"):
    if "North Carolina" in team.name:
        print(team.abbreviation)   # e.g. "NORTH-CAROLINA"
```

### NCAA Women's Softball / Men's Baseball
These have limited coverage in sportsipy. Use `ncaa_stats` module if available,
or fall back to the ESPN hidden API:

```python
import requests
# ESPN college sports hidden API
url = "https://site.api.espn.com/apis/site/v2/sports/baseball/college-baseball/teams"
resp = requests.get(url, params={"limit": 500})
```
