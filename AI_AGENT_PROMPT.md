# Prompt for an AI coding assistant

Copy everything inside the box below into your AI coding assistant (Claude Code, Copilot,
Cursor or similar), run from a terminal where `git` and Python 3.10+ are available and you
are logged in to GitHub with Write access to gridtrustai/GridTrustAI. Replace
`<HANDOFF_DIR>` with the full path of this unzipped folder.

```
You are helping submit Phase 1 of the GridTrust AI project (TSYP14 EnergyMind Tunisia).
Goal: put 4 prepared commits into https://github.com/gridtrustai/GridTrustAI and push them.
The commits are finished and tested. Do not rewrite, refactor or "improve" any code.

Inputs in <HANDOFF_DIR>:
- patches/0001-*.patch to 0004-*.patch  (git format-patch output, base commit 85b7592)
- repo/  (full repository content after the 4 commits, fallback only)
- README_FIRST.md  (human instructions, same steps as below)

Steps:
1. git clone https://github.com/gridtrustai/GridTrustAI.git && cd GridTrustAI
2. Run `git log --oneline -1`. If HEAD is 85b7592 ("phase 1"), run
   `git am <HANDOFF_DIR>/patches/*.patch`.
   If HEAD is different, or git am reports a conflict: run `git am --abort`, STOP and
   tell me what the new commits are. Do not resolve conflicts on your own.
3. python -m pip install -r requirements.txt
4. python -m unittest discover -s tests -v  -> must report 16 tests OK.
   If any test fails, STOP and show me the output. Do not edit code or tests.
5. Run, in order: python src/generate_dataset.py ; python src/detect_anomalies.py ;
   python src/evaluate.py
   Then run `git status`. Expected: no changes, or only timestamp changes
   ("created_at") inside results/decision_log.jsonl. If metrics.json numbers changed,
   STOP and tell me.
6. If step 5 changed only decision_log.jsonl timestamps, discard them:
   git checkout -- results/decision_log.jsonl
7. git push origin HEAD
8. Report: the pushed commit hashes, the test summary line, and the precision /
   recall / false_positive_rate values from results/metrics.json (row_level).

Hard rules:
- Never change thresholds, metrics, README numbers or data files by hand.
- Never commit downloaded datasets (LD2011_2014*, household_power_consumption*).
- Never force-push and never delete branches.
- If anything is unclear, stop and ask instead of guessing.
```

## Optional second prompt: real UCI data (only if time allows)

```
In the GridTrustAI repo (already updated and pushed), run the pipeline on real data:
1. Download https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip
   into a folder OUTSIDE the repo. If the zip contains a nested archive or folder,
   extract it so you have LD2011_2014.txt.
2. python src/generate_dataset.py --uci <path to LD2011_2014.txt or the zip>
   python src/detect_anomalies.py --uci
   python src/evaluate.py --uci <same path>
3. Show me data/uci_client_selection.csv and the row_level, event_level and
   grid_vs_data_quality sections of results/uci/metrics.json.
4. Wait for my approval. Only after I approve: add a short "Results on UCI measured
   power" subsection to README.md under "Phase 1 results", stating that power is measured
   (UCI ElectricityLoadDiagrams20112014, 5 household-scale clients) while voltage, current
   and all faults are simulated; then commit data/uci_*.csv, results/uci/, figures/*_uci.png
   and README.md, and push.
If a script errors on the real file, STOP and show me the error. Do not modify thresholds.
Never commit the downloaded dataset.
```
