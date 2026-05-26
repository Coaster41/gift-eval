# Summary of Qualitative Findings: `moiraic_base_11` vs `moiraie_base_7`

## Quick legend
- **c** = `moiraic_base_11`
- **e** = `moiraie_base_7`
- Numbers in `[ ]` are MASE relative to row-best (1.00 = winner); `{ }` are WQL.

---

## Where each model wins

**moiraic wins (often by large margins):**
`bizitobs_l2c/H` (all horizons), `ett2/H/long`, `m4_hourly/H/short`, `solar/10T/medium`, `solar/10T/long`, `us_births` (all granularities), `bizitobs_service/10S/long`

**moiraie wins:**
`bizitobs_application/10S/medium`, `bizitobs_service/10S/short`, `bizitobs_service/10S/medium`, `loop_seattle/5T/medium`, `loop_seattle/5T/long`, `ett2/W/short` (essentially tied)

**Roughly tied:** `solar/10T/short`, `ett2/W/short`

---

## Distinguishing characteristics

### 🟦 moiraic (`c`) — "smooth and committed"

| Trait | Examples |
|---|---|
| **Smoother, cleaner forecasts** | bizitobs_l2c (all), solar/10T/{med,long}, us_births |
| **Tighter quantile bands** | m4_hourly, bizitobs_l2c/H/short, us_births |
| **Locks onto clear periodicity well** | bizitobs_l2c/H/long, m4_hourly, us_births, solar |
| **Better peak/valley shape matching when data is clean** | us_births (bimodal peaks), solar/10T (smooth hills) |
| **Failure mode: regresses to flat or near-flat forecasts when data is noisy/irregular** | bizitobs_application/10S/medium ("delayed, tight, often flat"), bizitobs_service/10S/short & medium, loop_seattle/5T/{med,long} ("mostly flat... doesn't match spikiness") |
| **Failure mode: misses irregular spikes / sinks / dynamics** | loop_seattle (no hallucinated sinks but also no real sinks predicted), bizitobs_l2c/H/medium ("doesn't match spikes or intensity") |
| **Failure mode: occasional domain violations** | bizitobs_l2c/H/long predicts <0 values on count data |
| **Failure mode: over-confident wrong shape** | bizitobs_l2c/H/short ("smooth but often wrong... general shape ok") |

### 🟧 moiraie (`e`) — "expressive but artifact-prone"

| Trait | Examples |
|---|---|
| **Patch artifacting visible in forecasts** | bizitobs_application/10S/medium, bizitobs_service (short, medium), solar/10T/short |
| **Wider quantile bands, bands often share shape of median** | bizitobs_l2c/H/short, /medium, /long; solar/10T; us_births/W |
| **Better at capturing jitteriness, irregularity, and dynamics in noisy data** | bizitobs_service/10S (all), loop_seattle/5T/{med,long}, ett2/H/long ("trend is a lot more irregular that generally matches better") |
| **Sharper peaks/valleys** | bizitobs_l2c/H/medium, ett2/H/long |
| **Better at predicting sinks/dips when present** | loop_seattle (much better), bizitobs_service |
| **Failure mode: hallucinated spikes on flat sections** | solar/10T/medium & long ("hallucinates spikes on flat sections") |
| **Failure mode: misses smooth/clean periodic shapes** | solar/10T/{med,long} ("fails to predict many of the hills") |
| **Failure mode: shape is right but timing/period drifts** | bizitobs_service/10S/long ("off period"), us_births/W (more shifted) |
| **Failure mode: simplifies multimodal peak structure** | us_births/D ("unimodal instead of bimodal") |

---

## What separates the two models

1. **Smoothness vs. expressiveness**
   moiraic produces smooth, low-variance forecasts; moiraie produces jittery, more-dynamic forecasts. This is the single most consistent axis.

2. **Quantile band behavior**
   - moiraic: tighter bands, often miscalibrated when wrong (overconfident).
   - moiraie: wider bands that *follow the shape of the median* (a recurring observation across bizitobs_l2c, solar, us_births). This makes uncertainty more visually informative but possibly inflated.

3. **Patch artifacting (moiraie-specific)**
   Repeatedly noted in moiraie ("distinct patch artifacting", "each patch is distinct", "hard artifacts", "clear artifacting"). Never noted for moiraic. This is consistent with moiraie's patching-based architecture.

4. **Behavior on clean periodic data → moiraic wins**
   When the signal is clean and periodic (m4_hourly, us_births, solar smooth hills, bizitobs_l2c long), moiraic finds and matches periodicity precisely with tight bands. moiraie often gets period right but bungles peak shape or adds artifacts.

5. **Behavior on noisy/irregular data → moiraie wins**
   When the data is jittery, sink-prone, or aperiodic (loop_seattle, bizitobs_service, bizitobs_application, ett2/H/long), moiraic regresses to flat/safe forecasts and loses; moiraie's willingness to be jittery pays off even with artifacts.

6. **Hallucination styles are different**
   - moiraic hallucinates **smooth structure** (e.g., fake sinks in loop_seattle, smooth hills where data is irregular in solar).
   - moiraie hallucinates **high-frequency content** (spikes on flat regions in solar, off-period oscillations in bizitobs_service/long).

7. **Shape fidelity vs. timing fidelity**
   moiraic gets timing right but smooths shape; moiraie gets shape character right but timing/period can drift.

8. **Domain/data-type respect**
   One observation: moiraic predicted negative values on count data (bizitobs_l2c/H/long). Not noted for moiraie.

---

## High-level rule of thumb suggested by these notes

| If the dataset is... | Prefer |
|---|---|
| Clean, smooth, strongly periodic | **moiraic** |
| Noisy, jittery, irregular, sink/spike heavy | **moiraie** |
| Count data / has hard domain bounds | **moiraic** (but watch for negative predictions) |
| You care about uncertainty band shape | **moiraie** (bands track median) |
| You care about tight bands when correct | **moiraic** |

---

## Questions / clarifications I'd want before going further

1. **"Quantiles share same shape"** — I read this as "the quantile bands are essentially the median forecast shifted up/down rather than widening at uncertain regions." Is that what you meant? It shows up for both models in different rows so I want to make sure I'm not conflating two phenomena.

2. **"Hallucination"** — I treated moiraic's "hallucinated sinks" (smooth-but-fake structure) and moiraie's "hallucinated spikes" (artifact-like) as qualitatively different. Do you see them as the same failure or different?

3. **"Tight"** — used for both narrow quantile bands *and* for "tight on the true series." Want me to disambiguate these in a revised summary?

4. **Patch artifacting** — for moiraie this seems to be the dominant artifact signature. Would you like me to flag specific datasets where this is most visually severe (solar, bizitobs_service, bizitobs_application stand out) for a focused side-by-side?

5. **ett2/W/short is listed as a tie** — is the "(similar)" tag intentional, or did you not get a chance to dig in there?

Want me to turn this into a short bulleted slide-style summary, or extend it into a per-domain breakdown (Web/CloudOps vs Energy vs Healthcare etc.)?