# NFL — nfl_data_py

```python
import nfl_data_py as nfl
import pandas as pd
```

## Play-by-play data (1999–present)

```python
# Large dataset — each row is one play
df = nfl.import_pbp_data(years=[2022, 2023])
# Key columns: game_id, home_team, away_team, season_type, week,
#   game_date, quarter_seconds_remaining, game_seconds_remaining,
#   play_type, yards_gained, pass_length, pass_location,
#   rush_direction, passer_player_name, passer_player_id,
#   receiver_player_name, receiver_player_id,
#   rusher_player_name, rusher_player_id,
#   td_player_name, td_team, touchdown, pass_touchdown,
#   rush_touchdown, complete_pass, incomplete_pass, interception,
#   sack, penalty, penalty_type, ...
#   game_date is "YYYY-MM-DD" string — convert with pd.to_datetime()
```

## Weekly player stats (aggregated per player per game week)

```python
df = nfl.import_weekly_data(years=[2022, 2023])
# Columns: player_id, player_name, player_display_name, position,
#   position_group, recent_team, season, week, season_type,
#   completions, attempts, passing_yards, passing_tds, interceptions,
#   carries, rushing_yards, rushing_tds, rushing_fumbles,
#   receptions, targets, receiving_yards, receiving_tds,
#   target_share, air_yards_share, fantasy_points, ...
```

## Seasonal player stats

```python
df = nfl.import_seasonal_data(years=[2022, 2023])
# Same columns as weekly but summed over the full season
```

## Schedules

```python
df = nfl.import_schedules(years=[2022, 2023])
# Columns: game_id, season, game_type, week, gameday, weekday, gametime,
#   away_team, home_team, away_score, home_score, result, total,
#   overtime, old_game_id, gsis, nfl_detail_id, pfr, pff, espn,
#   away_rest, home_rest, away_moneyline, home_moneyline, spread_line,
#   away_spread_odds, home_spread_odds, total_line, ...
# gameday: "YYYY-MM-DD", weekday: "Sunday", "Monday", "Thursday", etc.
```

## Player info / rosters

```python
df = nfl.import_players()
# Columns: status, display_name, first_name, last_name, esb_id,
#   gsis_id, birth_date, college_name, position_group, position,
#   jersey_number, years_of_experience, ...

df = nfl.import_rosters(years=[2023], positions=["QB", "RB", "WR", "TE"])
# Columns: player_name, player_id, position, team, ...
```

## Draft data

```python
df = nfl.import_draft_picks(years=[2020, 2021, 2022])
# Columns: season, round, pick, team, player_name, position, ...
```

## Day-of-week in NFL

Most NFL regular season games are on **Sunday**. Monday Night Football = Monday.
Thursday Night Football = Thursday. International games sometimes Saturday.

```python
schedules = nfl.import_schedules(years=[2023])
schedules["gameday"] = pd.to_datetime(schedules["gameday"])
schedules["weekday"] = schedules["gameday"].dt.day_name()
sunday_games = schedules[schedules["weekday"] == "Sunday"]
```

## Combine results

```python
df = nfl.import_combine_data(years=[2020, 2021, 2022])
# Columns: season, player_name, pos, team, ht, wt, forty, bench,
#   vertical, broad_jump, cone, shuttle, ...
```
