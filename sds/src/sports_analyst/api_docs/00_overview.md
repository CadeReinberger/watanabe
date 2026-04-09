# Sports Data APIs — Overview

The following Python libraries are installed and available. Use only these libraries (plus the Python standard library and pandas/numpy). Do NOT attempt to install packages at runtime.

| Sport | Library / Package | Notes |
|-------|------------------|-------|
| MLB | `pybaseball` | Baseball Reference, FanGraphs, Statcast |
| NBA / WNBA | `nba_api` | Official NBA stats API |
| NFL | `nfl_data_py` | NFLverse — play-by-play back to 1999 |
| NHL | `nhl` (nhl-api-py) | Public NHL API |
| NCAA Football (M) | `cfbd` (College Football Data) | Free key via env var `CFBD_API_KEY` |
| NCAA Basketball (M/W) | `sportsipy` | Scrapes Sports Reference |
| NCAA Baseball (M) | `sportsipy` | Scrapes Sports Reference |
| NCAA Softball (W) | `sportsipy` (limited) | Use `sportsipy.ncaa` |
| NCAA Hockey (M/W) | `sportsipy` | Scrapes Sports Reference |
| Soccer (all) | `soccerdata` | EPL, La Liga, Bundesliga, Serie A, MLS, more |
| Cricket | `cricpy` | ESPNcricinfo |
| Rugby | `requests` + ESPN API | See rugby doc for endpoints |

## General pandas tips for day-of-week analysis
```python
import pandas as pd
df["date"] = pd.to_datetime(df["date_column"])
df["day_of_week"] = df["date"].dt.day_name()   # "Monday", "Sunday", etc.
df_sundays = df[df["day_of_week"] == "Sunday"]
```

## Throttling
- `nba_api` endpoints have built-in timeouts; add `time.sleep(1)` between calls
- `sportsipy` scrapes HTML — add `time.sleep(2)` between requests to avoid 429s
- `pybaseball` automatically caches to `~/.pybaseball/`
