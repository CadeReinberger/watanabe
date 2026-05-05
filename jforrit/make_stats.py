import sys
import pandas as pd

def main():
    if len(sys.argv) != 2:
        print("Usage: python make_stats.py <input.xlsx>")
        sys.exit(1)

    df = pd.read_excel(sys.argv[1])

    stats = df.groupby("player").agg(
        avg_rank=("rank", "mean"),
        ppg=("score", "mean"),
    )

    standings = stats.sort_values(
        by=["avg_rank", "ppg"],
        ascending=[True, False],
    ).reset_index()

    standings.insert(0, "place", range(1, len(standings) + 1))
    standings.columns = ["Place", "Player", "Avg Rank", "PPG"]
    standings["Avg Rank"] = standings["Avg Rank"].round(3)
    standings["PPG"] = standings["PPG"].round(2)

    with pd.ExcelWriter("stats.xlsx", engine="openpyxl") as writer:
        standings.to_excel(writer, sheet_name="Standings", index=False)

    print(f"Written stats.xlsx with {len(standings)} players.")

if __name__ == "__main__":
    main()
