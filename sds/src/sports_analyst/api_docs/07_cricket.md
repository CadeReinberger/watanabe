# Cricket — cricpy + ESPN Cricinfo API

## Option A: cricpy

```python
import cricpy.analytics as ca
```

### Retrieve batsman/bowler data from ESPNcricinfo

```python
# Get career batting stats for a player
# player_id: ESPNcricinfo player ID (e.g. 163975 = Sachin Tendulkar)
# matchType: "Test", "ODI", "T20I", "IPL"
# outdir: directory to save CSV files

ca.getPlayerData(163975, dir=".", file="tendulkar.csv",
                 type="batting", matchType="Test")

# Load saved data
import pandas as pd
df = pd.read_csv("tendulkar.csv")
# Columns: Runs, Mins, BF (balls faced), 4s, 6s, SR (strike rate),
#   Inns, Date, Opposition, Ground, ...

ca.getPlayerData(163975, dir=".", file="tendulkar_odis.csv",
                 type="batting", matchType="ODI")
```

### Analysis functions

```python
# Performance against specific opponents
ca.relativeBatsmanCumulativeAvgRuns("tendulkar.csv", "Tendulkar")
ca.batsmanRunsAgainstOpposition("tendulkar.csv", "Tendulkar")
ca.batsmanRunsOnGround("tendulkar.csv", "Tendulkar")

# For bowlers
ca.getPlayerData(163975, dir=".", file="bowler.csv",
                 type="bowling", matchType="Test")
```

## Option B: ESPN Cricinfo API (direct REST, more reliable)

```python
import requests
import pandas as pd

# Player stats — Test batting
# Find player ID by searching ESPNcricinfo (163975 = Sachin Tendulkar)
# stat_type: "batting" or "bowling"

def get_cricinfo_stats(player_id: int, class_id: int = 1) -> dict:
    """
    class_id: 1=Test, 2=ODI, 3=T20I, 11=T20 (all), 6=IPL
    """
    url = (
        f"https://stats.espncricinfo.com/ci/engine/player/{player_id}.json"
        f"?class={class_id};template=results;type=batting;view=innings"
    )
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    return resp.json()
```

## Option C: Cricbuzz-style API (hidden ESPN endpoint)

```python
import requests

# Search for a player
resp = requests.get(
    "https://site.api.espn.com/apis/site/v2/sports/cricket/players",
    params={"query": "Virat Kohli", "limit": 5},
    headers={"User-Agent": "Mozilla/5.0"}
)

# Get cricket scorecards
resp = requests.get(
    "https://site.api.espn.com/apis/site/v2/sports/cricket/scoreboard",
    params={"dates": "20231015"}
)
```

## Common player IDs (ESPNcricinfo)
- Sachin Tendulkar: 163975
- Ricky Ponting: 7133
- Brian Lara: 52337
- Virat Kohli: 253802
- Steve Smith: 267192
- Joe Root: 303669
- Kane Williamson: 277906
- Rohit Sharma: 34102

## Formats
- **Test matches**: 5-day format, "Test"
- **ODI**: One Day International (50 overs), "ODI"
- **T20I**: Twenty20 International, "T20I"
- **IPL**: Indian Premier League (T20), "IPL"
- **The Hundred**: UK 100-ball competition

## Key stats
- Runs, Average, Strike Rate, 100s (centuries), 50s, HS (highest score)
- Wickets, Economy Rate, Bowling Average, Strike Rate, BBM (best bowling in match)
