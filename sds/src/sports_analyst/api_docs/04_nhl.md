# NHL — nhl-api-py + direct API

## Option A: nhl-api-py library

```python
from nhl import NHLApi
api = NHLApi()
```

## Option B: Direct REST API (recommended — most reliable)

```python
import requests
import pandas as pd

BASE = "https://api-web.nhle.com/v1"
```

### Player search / ID lookup

```python
# Search by name — returns list of player dicts
resp = requests.get(f"{BASE}/search/player",
                    params={"q": "Wayne Gretzky", "culture": "en-US", "limit": 5})
players = resp.json()["searchResults"]["playerSearchResults"]
# Each dict: playerId, firstName, lastName, teamAbbrev, positionCode, ...
player_id = players[0]["playerId"]   # e.g. 8447400 for Gretzky
```

### Player landing page (career stats)

```python
resp = requests.get(f"{BASE}/player/{player_id}/landing")
data = resp.json()
# data["careerTotals"]["regularSeason"]: gamesPlayed, goals, assists, points, ...
# data["seasonTotals"]: list of season-by-season dicts
```

### Player game log (season-by-season game-by-game)

```python
# season: 8-digit int like 20222023
# game_type: 2 = regular season, 3 = playoffs
resp = requests.get(f"{BASE}/player/{player_id}/game-log/20222023/2")
games = resp.json()["gameLog"]
# Each game dict: gameId, teamAbbrev, homeRoadFlag, gameDate,
#   goals, assists, points, plusMinus, powerPlayGoals, shots,
#   shifts, gameWinningGoals, ...

df = pd.DataFrame(games)
df["gameDate"] = pd.to_datetime(df["gameDate"])
df["DayOfWeek"] = df["gameDate"].dt.day_name()
```

### Team roster

```python
# season: "20232024"
resp = requests.get(f"{BASE}/roster/TOR/20232024")  # TOR = Toronto Maple Leafs
data = resp.json()
# data["forwards"], data["defensemen"], data["goalies"]
# Each player: id, firstName, lastName, sweaterNumber, positionCode, ...
```

### Schedule

```python
# Monthly schedule for a team
resp = requests.get(f"{BASE}/club-schedule/TOR/month/2023-01")
games = resp.json()["games"]
# Each game: id, startTimeUTC, homeTeam, awayTeam, gameState, ...

# Full season schedule
resp = requests.get(f"{BASE}/club-schedule-season/TOR/20232024")
```

### League schedule (all games on a date)

```python
resp = requests.get(f"{BASE}/schedule/2023-01-14")
data = resp.json()
games = data["gameWeek"][0]["games"]
```

### Standings

```python
resp = requests.get(f"{BASE}/standings/2023-04-01")
standings = resp.json()["standings"]
# Each team: clinchIndicator, conferenceAbbrev, conferenceName, divisionAbbrev,
#   gamesPlayed, goalDifferential, goalFor, goalAgainst, homeWins, homeLosses,
#   roadWins, roadLosses, points, wins, losses, otLosses, ...
```

### Team abbreviations (common)
- TOR Toronto Maple Leafs, MTL Montreal Canadiens, BOS Boston Bruins
- NYR NY Rangers, NYI NY Islanders, PHI Philadelphia Flyers
- PIT Pittsburgh Penguins, WSH Washington Capitals, CBJ Columbus Blue Jackets
- DET Detroit Red Wings, CHI Chicago Blackhawks, STL St. Louis Blues
- COL Colorado Avalanche, EDM Edmonton Oilers, VAN Vancouver Canucks
- LAK LA Kings, ANA Anaheim Ducks, SJS San Jose Sharks
- SEA Seattle Kraken, VGK Vegas Golden Knights, ARI Arizona Coyotes
- CGY Calgary Flames, WPG Winnipeg Jets, MIN Minnesota Wild
- DAL Dallas Stars, NSH Nashville Predators, TBL Tampa Bay Lightning
- FLA Florida Panthers, CAR Carolina Hurricanes, NJD New Jersey Devils

### Notable player IDs
- Wayne Gretzky: 8447400
- Mario Lemieux: 8447679
- Sidney Crosby: 8471675
- Alex Ovechkin: 8471214
- Connor McDavid: 8478402
