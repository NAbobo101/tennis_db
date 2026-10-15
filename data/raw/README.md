# Raw data

All files under `data/raw/` are **third-party data** created and maintained by
[Jeff Sackmann / Tennis Abstract](http://www.tennisabstract.com/). They are vendored
unmodified so that the project works offline right after cloning. The folders mirror
the upstream repositories one to one:

| Folder | Upstream repository | Contents |
|---|---|---|
| [`tennis_atp/`](tennis_atp) | [JeffSackmann/tennis_atp](https://github.com/JeffSackmann/tennis_atp) | ATP tour-level, Challenger/qualifying, Futures, doubles and amateur match results (1968-2024), players, weekly rankings, `matches_data_dictionary.txt` |
| [`tennis_wta/`](tennis_wta) | [JeffSackmann/tennis_wta](https://github.com/JeffSackmann/tennis_wta) | WTA tour-level and ITF/qualifying match results (1968-2024), players, weekly rankings, upstream `README.md` |
| [`tennis_MatchChartingProject/`](tennis_MatchChartingProject) | [JeffSackmann/tennis_MatchChartingProject](https://github.com/JeffSackmann/tennis_MatchChartingProject) | Shot-by-shot charted matches and aggregated stats, `data_dictionary.txt`, the `MatchChart` spreadsheet |

## What the pipeline uses

| File family | Used for |
|---|---|
| `tennis_atp/atp_matches_YYYY.csv`, `tennis_wta/wta_matches_YYYY.csv` | Rating history **and** the training/evaluation rows (tour-level main draws) |
| `tennis_atp/atp_matches_qual_chall_YYYY.csv`, `tennis_atp/atp_matches_futures_YYYY.csv`, `tennis_wta/wta_matches_qual_itf_YYYY.csv` | Rating history only (Elo, form, head-to-head) |
| Players, rankings, doubles, amateur, Match Charting Project | Not used by the model; kept for exploration and future work |

Ranks, ages, heights, handedness and entry status are read directly from the match rows,
where they are recorded as of the tournament date.

## Re-downloading

The same layout can be rebuilt from upstream with:

```bash
uv run tennis data download --tour all --dest data/raw
```

## License and attribution

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a>

*Tennis databases, files, and algorithms* by
[Jeff Sackmann / Tennis Abstract](http://www.tennisabstract.com/) is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-nc-sa/4.0/).
Based on a work at <https://github.com/JeffSackmann>.

This license applies to everything in `data/raw/` and to anything derived from it, including the
trained models and the evaluation reports produced by this project. Attribution is required and
commercial use is not permitted. The source code of this project is licensed separately under the
MIT License (see [`LICENSE`](../../LICENSE)).
