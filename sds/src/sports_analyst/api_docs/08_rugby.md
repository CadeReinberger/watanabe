# Rugby — ESPN API + World Rugby API

Rugby has the least Python library support; use direct HTTP endpoints.

## ESPN Rugby API

```python
import requests
import pandas as pd

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports"

# League codes for ESPN:
# rugby-union/international (Six Nations, Rugby Championship, World Cup)
# rugby-union/eng.1  (English Premiership)
# rugby-union/fra.top14  (Top 14 France)
# rugby-union/ire.proD2  (United Rugby Championship)
# rugby-league/nrl  (NRL — Australian rugby league)
# rugby-league/eng.super-league  (Super League)
```

### Scoreboard / schedule

```python
league = "rugby-union/international"
resp = requests.get(
    f"{ESPN_BASE}/{league}/scoreboard",
    params={"dates": "20231021", "limit": 100},
    headers={"User-Agent": "Mozilla/5.0"}
)
data = resp.json()
events = data.get("events", [])
for event in events:
    print(event["name"], event["date"], event["competitions"][0]["competitors"])
```

### Team list

```python
resp = requests.get(
    f"{ESPN_BASE}/rugby-union/international/teams",
    params={"limit": 100},
    headers={"User-Agent": "Mozilla/5.0"}
)
teams = resp.json()["sports"][0]["leagues"][0]["teams"]
for t in teams:
    print(t["team"]["id"], t["team"]["displayName"], t["team"]["abbreviation"])
```

### Team schedule

```python
resp = requests.get(
    f"{ESPN_BASE}/rugby-union/international/teams/39/schedule",
    params={"season": 2023},
    headers={"User-Agent": "Mozilla/5.0"}
)
events = resp.json().get("events", [])
```

### Athlete / player stats

```python
# Get athletes for a team
resp = requests.get(
    f"{ESPN_BASE}/rugby-union/international/teams/39/roster",
    headers={"User-Agent": "Mozilla/5.0"}
)
athletes = resp.json().get("athletes", [])
for athlete in athletes:
    print(athlete["displayName"], athlete["position"]["displayName"])
```

### Game summary / box score

```python
event_id = "..."  # from scoreboard
resp = requests.get(
    f"{ESPN_BASE}/rugby-union/international/summary",
    params={"event": event_id},
    headers={"User-Agent": "Mozilla/5.0"}
)
summary = resp.json()
```

## World Rugby Stats API

```python
WR_BASE = "https://api.wr-rims-prod.pulselive.com/rugby/v3"

# Rankings
resp = requests.get(f"{WR_BASE}/rankings/mru",  # men's union
                    headers={"Accept": "application/json"})
rankings = resp.json()

# Teams
resp = requests.get(f"{WR_BASE}/team", headers={"Accept": "application/json"})

# Matches
resp = requests.get(f"{WR_BASE}/match",
                    params={"startDate": "2023-09-01", "endDate": "2023-10-31",
                            "sort": "asc", "pageSize": 100},
                    headers={"Accept": "application/json"})
matches = resp.json()
```

## NRL (National Rugby League — Australian rugby league)

```python
NRL_BASE = "https://nrl.com/api"

# Season fixtures
resp = requests.get("https://www.nrl.com/draw/nrl-premiership/2023/",
                    headers={"User-Agent": "Mozilla/5.0",
                             "Accept": "application/json"})

# ESPN NRL
resp = requests.get(
    f"{ESPN_BASE}/rugby-league/nrl/scoreboard",
    params={"dates": "20230901"},
    headers={"User-Agent": "Mozilla/5.0"}
)
```

## Common teams (ESPN IDs for international rugby)
- New Zealand (All Blacks): 39
- South Africa (Springboks): 30
- England: 22
- Australia (Wallabies): 15
- Ireland: 26
- France: 23
- Wales: 37
- Scotland: 29
- Argentina (Pumas): 14

## Super Rugby (Southern Hemisphere club rugby)

```python
resp = requests.get(
    f"{ESPN_BASE}/rugby-union/superrugby/scoreboard",
    params={"dates": "20230401"},
    headers={"User-Agent": "Mozilla/5.0"}
)
```
