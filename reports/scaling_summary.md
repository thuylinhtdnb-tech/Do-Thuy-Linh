# Scaling summary (machine RAM 17.0 GB)

| engine | points (scale) | MB per million rows | seconds per million rows | rows at 80% of RAM (millions) |
|---|---|---|---|---|
| pandas | 3 | 250 | 1.81 | 52.8 |
| polars_lazy | 3 | 170 | 0.32 | 76.2 |
| polars_streaming | 3 | 149 | 0.40 | 86.1 |
