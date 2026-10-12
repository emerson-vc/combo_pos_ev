# Combo +EV v3: design

This is the spec for the next version of the scanner. It is built from three sources: Emerson's hand-found
combos, a 5-season NFL correlation map, and a 5-season MLB correlation map (both in this folder).
Implement it phase by phase, in order.

## 1. What the edge actually is

A combo's fair payout is `(product of fair leg payouts) / lift`, where **lift** = P(all legs hit) divided by
the product of the leg probabilities. Kalshi's makers quote `(product of leg payouts) / d`, where **d** is
the correlation discount they apply. So

    EV ≈ (lift / d) × (leg price quality) − 1 − fees

The scanner has to estimate two numbers per combo: the **true lift** (from history) and **the makers' d**
(from our own quotes). The money is in pairs where lift is big and d is near 1.

Evidence so far. "Legs multiplied" is the payout if Kalshi treated the legs as unrelated, and d is that number divided by the combo payout:

| Combo (Emerson, 10/12) | Legs multiplied | Kalshi | Makers' d | True lift (research) |
|---|---|---|---|---|
| Freeman 2+ HR + LAD ML | 80.8x | 106.4x | **0.76** | ~1.48 |
| Stafford 4+ pass TD + Rams over 28.5 | 9.95x | 10.0x | **1.00** | ~1.9 (3.2 vs alt line) |
| Henderson 9+ K + MIL ML | 18.3x | 17.0x | 1.07 | ~1.21 |
| BUF ML + Allen 2+ pass TD + Cook TD | 9.0x | 7.5x | 1.20 | ~1.29 |
| BUF ML + Allen 325+ pass yds | 17.8x | 17.7x | 1.01 | ~1.04 (no edge) |

The makers barely discount, and sometimes discount the wrong way, for **player tail prop + own-team outcome**
pairs. They discount more sensibly on combos that look like a typical parlay (ML + QB TD + RB TD).

## 2. Research findings (lift vs independence, controlled for pre-game expectations)

Lifts are measured within buckets of similar pre-game spread/total (NFL: closing lines; MLB: team
offense/defense and opposing-starter proxies). That removes the "good teams do everything" effect that
Kalshi's leg prices already include. See `nfl_edge_map.csv` and `mlb_edge_map.csv`.

**NFL (2021–25 regular season):**
- QB 4+ pass TD + team over its total: 1.9; + team over total +7: **3.2**
- QB 3+ pass TD + team over +7: 2.5; + team over: 1.8; + TE1 TD: 1.8; + WR1 TD: 1.6
- QB 300+ yds + WR1 100+ yds: 2.0; adding "game over" as a 3rd leg: **3.1**
- QB 3+ TD + WR1 TD + team over (3 legs): **2.8**
- RB1 2+ TD + team over +7: 2.3; WR1 2+ TD + team over +7: 2.4
- Weak or none: QB pass yds + team wins 1.04, WR1 8+ rec + win 1.08, TE1 75+ yds + win 0.98
- Negative: QB TDs + RB1 TD 0.98 (they compete for TDs), QB 300 + RB1 100 0.60, any prop + game under 0.3–0.8

**MLB (2021–25 regular season, Retrosheet play-by-play):**
- 2+ HR + team ML: 1.48; + team over its total: 1.89; + team over +2: **2.57**
- 3+ RBI + team over +2: 2.8; 4+ RBI + team over +2: **3.4**; 2+ RBI + team over: 1.76
- 3+ hits + team over +2: 2.3; 1+ HR + team over: 1.47
- SP 9+ K + team wins: 1.21; SP 9+ K + opponent UNDER: 1.27
- Weak: teammate HR + HR 1.07; HR + teammate 2+ RBI 0.98; batter props + opponent UNDER ~1.0
- Negative: any batter prop + game under 0.2–0.75

**Takeaways**
1. TDs and RBIs drive the correlation, yards and receptions don't.
2. Swap moneylines for **team total overs**, especially alt rungs (+7 NFL, +2 MLB). The lift is much higher,
   and Kalshi doesn't block HR + team total the way it blocks spread + ML.
3. Teammate pairs are weak in MLB; in the NFL, QB + pass-catcher pairs are strong.
4. Never combine a prop with a game under.

## 2b. Decisions and facts from Emerson (10/12)
- **Combo prices update instantly in the app** when a leg is added, so makers price them with an algorithm, not by
  hand. Expect the makers' d to be nearly deterministic per pair type: try to recover the exact formula (e.g.
  product of leg prices × a per-pair-type factor) instead of only averaging it. Instant quotes also mean
  `quote_wait_seconds` can likely drop from 4s to about 1s; measure the real response time and use it, so a scan
  can quote far more combos.
- **Alt team totals combine with player props**: Freeman 2+ HR (47x) + Dodgers over 3.5 (1.9x) paid 95.24x,
  more than the 89.3x the legs multiplied give (d = 0.94), while history says the true lift is about 1.9. So the
  team-total version is as mispriced as the ML version. Sweep every alt rung.
- **NO sides work in combos**, so opponent UNDER and player UNDER legs are in scope (e.g. SP Ks + opponent under).
- **Payout slips slightly at bigger sizes**: always re-quote at the real stake before accepting (autobet already
  does) and log the slip per pair type.
- **3-leg combos also price instantly**, so the same formula approach applies to N legs. Learn how d grows
  with each added leg, not just per pair.
- **The same combo's payout changes over minutes even when the leg prices don't move.** The formula has a
  time-varying part: probably the maker's exposure, i.e. how much has already been bet on similar combos.
  Consequences: (1) log every quote with a timestamp and re-quote the top candidates 2–3 times in a scan,
  betting only when the price is at or better than its recent typical level; (2) record which combos get worse
  after we bet them, to measure our own price impact; (3) less popular combos (not star HR + ML) are likely to
  stay softer for longer.
- **Scans stay manual** (only when Emerson clicks Scan). No scheduler.
- **Daily loss limit: 20% of bankroll.** Set `daily_max_spend_pct` default to 0.20; keep 2%/bet and 5%/game.

## 3. Phases

### Phase 0: Get the focus back (the regression since V1)
Commit 36abc5e switched to an all-pairs search. That spread 15–20 quotes per game across dozens of combo types,
reserved 40% of MLB slots for long shots, and renamed the combo types, so the quote history from V1 no longer applies.
- Score every candidate by **predicted EV** (section 1) before quoting, and quote in score order.
- Until enough quotes exist, use the research lift tables as the prior for true lift and d=1.0 as the prior
  for the makers.
- Keep exploring at about 15% of slots. Drop the fixed long-shot reservation and the always-quote pinned
  template; the scoring replaces them.
- Template IDs must be stable and readable from the legs (relationship, kinds, threshold bucket).

### Phase 1: Quote ledger and grading (the scoreboard)
- Log every quote with: the legs (ticker, side, kind, threshold), each leg's bid/ask/mid at quote time, the combo
  price, all maker quotes, `active_quoters` for the event, the time to start, and our q / q_lo / lift.
- Store `d = (product of leg ask payouts) / combo payout` and `d_mid` (using leg mids) on every quote.
- After the games settle, grade every quoted combo, not just bets, from each leg market's Kalshi result.
- History tab: per pair type, show median d, quotes, hit rate vs predicted q, and the realized ROI of combos
  we would have bet. Also calibration buckets for q.

### Phase 2: Model fixes
- Half, quarter and F5 outcomes are raw today, not measured against the pre-game line. Measure every team or game
  outcome against expectation (residuals, or the stratified approach used in `research/`). If that isn't
  possible for a market type, drop the type.
- Generate legs for **every rung** of team totals and game totals, not just the main line.
- For NFL roles, use season-to-date usage (no look-ahead). For MLB, use the posted lineup slot.
- Port the stratified lift calculation (Mantel–Haenszel across pre-game buckets) into the model as a
  cross-check on the copula. Flag pairs where they disagree by more than 25%.

### Phase 3: Model the makers
- Fit d per pair type with shrinkage: specific pair type → relationship → sport → global. Use a hierarchical
  or empirical-Bayes mean of log d, plus `active_quoters`, time to start and leg thresholds as features.
- Predicted payout = product of leg payouts / predicted d. Predicted EV is computed from that.
- Add a cheap screen: a combo whose quote has d ≤ 1.0 while its true lift is ≥ 1.3 is top priority, and is
  re-quoted at the real stake right away.

### Phase 4: 3-leg combos
- Read the combo collection's `associated_events` (`size_max`, `is_yes_only`, `active_quoters`) and the
  collection's `size_max`. Never build a combo that breaks them.
- Keep an allowed-pair graph: every pair Kalshi rejects (`duplicated_legs` or similar) is cached as blocked,
  and every pair that created fine is cached as allowed. 3-leg candidates use only allowed pairs.
- Candidates: extend the top-scoring 2-leg combos with a third leg that has positive lift with **both**
  existing legs. Shapes to start with: QB TD + pass-catcher TD + team over; batter RBI/HR + teammate hits +
  team over; SP Ks + team ML + opponent under.
- Joint probability: use the empirical stratified joint where there are at least 30 joint hits historically,
  otherwise a Gaussian copula with the pairwise ρ matrix (repaired to PSD). Use the more conservative of the two
  for q_lo.
- Make N legs work through create, RFQ, accept, the DB, autobet and the UI. 2-leg must keep working.

### Phase 5: Payout bands
- Setting: target payout bands, default `[3, 30]` plus a long-shot band `[30, 250]`, each with its own share of
  the per-game quote budget (default 70/30).
- Rank inside each band by predicted EV with q_lo, not by lift.

### Phase 6: Sizing and auto-bet
- Size from q_lo until Phase 1 grading shows the model is calibrated for that pair type. Combos on the same game
  share outcomes, so cap them jointly: sum of stakes × the correlation between combos, not a flat per-game cap
  only.
- Auto-bet live only for pair types with at least 50 graded quotes, where the predicted EV has been
  positive in realized ROI. Everything else stays in dry run.
- Keep every existing cap and the dry run → live flow. Ask before changing any default risk limit.

### Phase 7: Timing
- Scans are manual. Show a hint in the UI when a better time to scan is near: MLB after lineups post, NFL after
  inactives (about 90 min before kickoff).
- Log how quotes for the same combo change between scans. If makers are slow to follow leg price moves,
  rescan right after big leg moves.

## 4. Acceptance tests
- Unit test the scoring with the five combos in section 1 (leg prices and combo payout as inputs). It must
  rank Freeman and Stafford as strong edges and Henderson as positive, and must not flag Allen 325+ yds.
- Backtest: on held-out 2025 data, the stratified lift for the top 20 pair types must be within its
  confidence interval of the 2021–24 estimate.
- No change ships without the full test suite passing.

## 5. Out of scope for now
NBA (next, once NFL and MLB run clean), and cross-game combos.
