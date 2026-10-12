# Research: how often combo legs hit together

Lift = P(both legs hit) / (P(A) × P(B)), measured within buckets of similar pre-game expectations.
Results: `nfl_edge_map.csv`, `mlb_edge_map.csv`. The design built on these is in `DESIGN.md`.

NFL (nflverse):
    mkdir nfl && cd nfl
    for y in 2021 2022 2023 2024 2025; do curl -sSfLo ps_$y.parquet https://github.com/nflverse/nflverse-data/releases/download/player_stats/player_stats_$y.parquet; done
    curl -sSfLo games.csv https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv
    cp ../nfl_edge_map.py ../nfl_three_leg.py . && python nfl_edge_map.py . && python nfl_three_leg.py

MLB (Retrosheet play-by-play via Chadwick Bureau):
    git clone --depth 1 --filter=blob:none --sparse https://github.com/chadwickbureau/retrosheet.git rs
    (cd rs && git sparse-checkout set seasons/2021 seasons/2022 seasons/2023 seasons/2024 seasons/2025)
    python mlb_parse_retrosheet.py && python mlb_edge_map.py
