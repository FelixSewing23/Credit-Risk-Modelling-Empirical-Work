# Changes that move the numbers: course notebooks vs. our pipeline

**Thomas de Lesquen · Felix Sewing · Credit Risk Modelling (MPGI, FGV EAESP) · empirical work**

This lists every place our notebooks differ from the course notebooks **where the
difference changes a result**. Each item puts the course code and our code side by side, says why we
changed it, and shows what it does to the numbers. Environment fixes that change no numbers (the
`optbinning` / scikit-learn 1.8 shim, for instance) are collected once at the end and not discussed.

**Order.** By notebook (01 → 02 → 03). Within each notebook, most severe first. We rank severity by
how far the change moves the model's validity or its headline numbers.

**Cell numbers** are 0-based and count markdown cells, as `nbformat` gives them:

```python
import nbformat
nb = nbformat.read('course_files/code/02_Variable_Construction.ipynb', as_version=4)
print(nb.cells[45].source)
```

| | course notebook | our notebook |
|---|---|---|
| 01 download | `01_DataDownload_student_v2.ipynb` | `01_DataDownload.ipynb` |
| 02 variables | `02_Variable_Construction.ipynb` (IN02) | `02_VariableConstruction.ipynb` (NB02) |
| 03 categorization | `03_Categorization_Selection.ipynb` (IN03) | `03_CategorizationSelection.ipynb` (NB03) |

Slide references: `S4-17` = Session 4, slide 17. Where an item carries a **notebook label**, that
is the `C`/`D` number the change is tagged with inside our own notebook markdown, so you can find
it at the cell where it happens.

**Where the figures come from.** Everything on our side is the saved output of the current
notebooks, on the prudential-conglomerate panel (201701-202603). Course-side figures are quoted from
your saved notebook output wherever it exists. Anything labelled *reproduced* comes from running your
logic on our data.

---

## At a glance

| # | notebook | change | severity | effect on the numbers |
|---|---|---|---|---|
| **N1** | 01 | model at prudential-conglomerate level only, not a stack of three levels | High | rows usable for fitting 9,141 → 13,883; out-of-time bad events 31 → 207 |
| **V1** | 02 | missing `BIS_linear` → *unknown*, not *breach* | High | bad rate 16.44% → **9.10%** |
| **V2** | 02 | interpolation stops at the last real observation | Medium | 3,849 invented `CRWA` values and 145 invented `Capital`/`RWA` values removed |
| **V3** | 02 | explicit default list from `default_flag.xlsx`, window anchored on the decree | Low | `ind_default_12m` 12 → 23 rows; headline bad rate unchanged |
| **S1** | 03 | `C01` and `A02` not offered to the model (target leakage) | High | out-of-time AUC 0.899 → 0.864 (the higher figure is the leak) |
| **S2** | 03 | variable direction taken from the training data, not assumed | Medium | `L01` kept instead of dropped; train AUC 0.784 → 0.814, oot 0.848 → 0.864 |
| **S3** | 03 | train / oot split moved to 201703-202212 / 2023 | Medium | out-of-time bad events 100 → 207 |
| **S4** | 03 | binning, VIF, correlation and stepwise fit on `train` only | Medium | oot AUC 0.867 → 0.864 (small here, but no longer optimistic by construction) |
| **S5** | 03 | stepwise scored on `roc_auc`, not `accuracy`; full path | Low | selection path becomes monotone; fitted model unchanged |
| **M1** | 04 | logit and forest compared on identical WoE features and rows | High | dAUC +0.0031, p=0.479 (a clean tie) instead of a meaningless 0.862 vs 0.816 |
| **M2** | 04 | no `fillna(0)` on raw CAMELS ratios; native-NaN or median+flag learners | High | `A01` 24.2%/68.2% missing (train/mr) no longer relabelled "excellent" |
| **M3** | 04 | SGS series 4189 (monthly SELIC), not 432 (daily, returns a 406 error); no bare `except` | High | SELIC no longer silently absent while still being cited as a driver |
| **M4** | 04 | paired stratified bootstrap, B=4,000, on oot | Medium | Logit − Forest(WoE): p=0.479; Forest(WoE) − Forest(WoE+macro): p=0.004 |
| **M5** | 04 | placebo test on the macro overlay (50 draws, re-dated months) | Medium | ~63% of the +0.0067 macro AUC gain is calendar artefact, not economics |
| **M6** | 05 | rating scale / PSI flagged as anticipated, not-yet-taught material | Low | labelling only, no numeric effect |
| **M7** | 04 | inner 70/30 `train_test_split` inside `train` not carried into notebook 04 | Low | oot AUC 0.8636 (03, 70% of train) → 0.8652 (04, 100% of train) |

**Headline, end to end:**

| | course logic | ours |
|---|---:|---:|
| unit of analysis | stacked ind + soc + pru rows | prudential conglomerate (`COD_CONGL`) |
| `df_var` rows | ≈ 86,000 | 14,566 |
| labelled rows / bad rate | all rows / 16.44% *(reproduced, current BIS rule)* | 13,883 / **9.10%** |
| candidate variables | `C01 A01 A02 M01 E01 L01` | `A01 M01 E01 L01` |
| directions | assumed (2 asc / 4 desc) | fitted on train (`M01`, `L01` flipped) |
| binning / selection fit on | whole panel | labelled `train` rows |
| top IV | `A02` 4.29 *(your IN03 cell 30)* | `E01` 0.97 |
| AUC train / oot / mr | 0.754 / 0.752 / 0.736 *(reproduced; IN03 has no AUC cell)* | **0.814 / 0.864 / 0.683** |

The two AUC rows are not directly comparable: they differ in target, population and window. The
`mr` value of 0.683 is lower than the others because `mr` spans the COSIF 1.0 → 1.5 break, where
`A01` and `M01` stop existing. That is a real finding, not a defect (see S3).

---

# Notebook 01: data

## N1 · One consolidation level, not three stacked levels · *High*

| | course | ours |
|---|---|---|
| cell | `01_DataDownload_student_v2` **48-50**, **80**, **88** | `01_DataDownload` cells `congl-intro` … `congl-save` |
| notebook label | | panel change **D1-D3** |

<table>
<tr><th>Course, v2 cells 48-50, 80, 88</th><th>Ours, appended cells</th></tr>
<tr><td>

```python
df_cosif_temp1 = df_cosif_individual.merge(
    df_cosif_prudencial_important_var,
    on=['DATA', 'CNPJ'], how='left')
df_cosif_temp2 = df_cosif_society.merge(
    df_cosif_prudencial_important_var,
    on=['DATA', 'CNPJ'], how='left')

df_cosif = pd.concat(
    [df_cosif_temp1, df_cosif_temp2,
     df_cosif_prudencial],
    axis=0, keys=["ind", "soc", "pru"])
...
df_pre_final = df_cosif.merge(
    df_if_prudencial,
    on=['DATA', 'COD_CONGL'], how='left')
...
df_merged['ID'] = np.where(
    df_merged['keys'] == "pru",
    df_merged['COD_CONGL'], df_merged['CNPJ'])
# -> 86,044 rows x 1,382 columns
```

</td><td>

```python
# the prudential pivot is already built;
# use it as the panel, one row per
# (DATA, COD_CONGL) - 0 duplicates
df_congl = df_cosif_prudencial.merge(
    df_if_prudencial,
    on=['DATA', 'COD_CONGL'], how='left')

df_final_congl = fold_to_cosif10_sum_after_202501(
    df_congl, map_path=map_path,
    sheet_name='final')

df_final_congl.to_parquet(
    dir_outputs / 'df_final_congl.parquet')
# -> 14,566 rows x 267 columns
```

</td></tr>
</table>

**Why.** The target is a threshold on the **BIS ratio**, and IF.Data reports that **per prudential
conglomerate**, which is why it's joined on `COD_CONGL`. The CAMELS ratios come from COSIF, which
exists at three levels. Stack those levels and one capital observation gets copied onto every row of
the group:

1. **One default event is counted more than once.** A conglomerate row and its member rows carry
   the same `BIS` and hence the same label. On the two levels we have downloaded, **8,934** individual
   rows carry exactly the `(DATA, COD_CONGL)` IF.Data values of a prudential row. The logit, the IV
   and the AUC treat these copies as independent observations.
2. **A row outside any conglomerate has no capital information at all.** The left join finds no
   IF.Data row, in any period. Under the original rule (V1) that row is labelled a breach; under the
   refined rule it's labelled good. Neither label is observed.

`DOCUMENTO 4060` (`BLOPRUDENCIAL`) carries the same chart of accounts, consolidated to the group.
Using only that level puts predictors and target on the same entity, and it needs no new download:
our 14,566 rows are exactly the `keys == "pru"` slice of your stacked frame. The full reasoning for
not using the stack is in the **side note** after notebook 03.

**Effect.** Measured with the identical notebooks 02/03 (all other changes applied); only the panel
differs:

| | individual (`BANCOS`) | conglomerate |
|---|---:|---:|
| rows | 19,258 | 14,566 |
| `BIS_linear` available after interpolation | 46.1% | **92.6%** |
| rows with a usable label | 9,141 | **13,883** |
| bad events | 440 | **1,263** |
| bad rate under the original missing-BIS rule | 56.71% | 17.82% |
| out-of-time bad events (2023) | 31 | **207** |
| AUC train / oot / mr | 0.808 / 0.844 / 0.695 | 0.814 / 0.864 / 0.683 |

**Limits.** The model rates conglomerates, not individual legal entities, and the conglomerate
population is dominated by payment institutions, brokers and DTVMs. We also lose one event: Banco
Letsbank's conglomerate `C0086206` stops at 202309, while the bank kept filing individually until
202507.

---

# Notebook 02: target and variables

## V1 · A missing BIS ratio is labelled "breach" · *High*

| | course | ours |
|---|---|---|
| cell | IN02 **45** (`bis_flag`, `bis_flag_12m`), **53** (`target`) | NB02 **37**, **41**, **44** |
| slides | S4-17, S4-26, S4-28 | |
| notebook label | | corrections **C2** and **C3** in NB02 |

<table>
<tr><th>Course, IN02 cells 45, 53</th><th>Ours, NB02 cells 37, 41, 44</th></tr>
<tr><td>

```python
def bis_flag(row):
    if (row['DATA']<=201512) & (row['BIS_linear']<11) ...
        return 1
    ...                       # 5 more eras
    elif (pd.isnull(row['BIS_linear'])==True):
        return 1              # missing -> breach
    else:
        return 0

df['bis_flag_12m'] = df[[...b1..b12...]].sum(axis=1)
                              # NaN counts as 0

def target(row):
    if row['ind_default_12m'] == 1 or \
       row['ind_default_12m_bis'] == 1:
        return 1
    else:
        return 0              # no "unknown"
```

</td><td>

```python
def bis_flag(row):
    if pd.isnull(row['BIS_linear']):
        return np.nan         # unknown, not a breach
    return float(row['BIS_linear']
                 < minimum_bis(row['DATA']))
                 # same 6-era CMN ladder

df['bis_flag_12m'] = df[b_cols].sum(
    axis=1, min_count=1)      # all-unknown -> NaN

df['ind_default_12m_bis'] = np.where(
    df['bis_flag_12m'].isna(), np.nan,
    (df['bis_flag_12m'] >= 1).astype(float))

df['target'] = np.where(
    (df['ind_default_12m'] == 1) |
    (df['ind_default_12m_bis'] == 1), 1.0,
    np.where(df['ind_default_12m_bis'].isna(),
             np.nan, 0.0))
df['modelling_sample'] = df['target'].notna()
```

</td></tr>
</table>

The six CMN 4553/4783 threshold eras are **unchanged**: same boundaries and values as S4-15,
written as a lookup table.

**Why.** S4-17 defends the branch as conservative: *"'unknown' defaults to a conservative breach,
not an assumed pass."* That holds when a missing value is an occasional gap. Here the missingness is
mostly structural (see N1), so the branch flags the same institutions permanently. The label then
becomes largely *"IF.Data has no row"*, and every IF.Data-based predictor inherits that signal. We
still score unlabelled rows; they're only excluded from fitting.

The refined rule you circulated (missing BIS is a breach only if `COD_CONGL` is non-null and
`DATA >= 201703`) is a real improvement on the individual panel. On the conglomerate panel, though,
`COD_CONGL` is never null, so the condition is always true. Our notebook carries that rule as the
comparison baseline (`bis_flag_course`, `target_course`).

**Effect** (conglomerate panel, NB02 cells 39, 44, 47):

```
BIS_linear missing                          :  1,081 rows
  flagged "breach" by the refined rule      :    863
genuine capital breaches                    :    648
P(target_course = 1 | BIS missing)          :  0.8131
P(explicit failure within 12m | BIS missing):  1.16%   (10 of 863 rows)
```

| | course rule (refined) | ours |
|---|---:|---:|
| labelled rows | 14,566 | 13,883 |
| bad events | 2,394 | 1,263 |
| **bad rate** | **16.44%** | **9.10%** |

On the individual panel with the original blanket rule, the same comparison is **56.71% → 4.88%**.
There, `P(target = 1 | BIS missing)` is exactly 1.0000.

**Limits.** The model becomes a PD model for institutions whose capital ratio is observable.
That's a non-random restriction of the kind S4-28 lists, and we say so in the report.

---

## V2 · Linear interpolation extrapolates past the last filing · *Medium*

| | course | ours |
|---|---|---|
| cell | IN02 **41** | NB02 **32** (check: **34**) |
| slides | S4-22, **S4-24** | |
| notebook label | | correction **C1** in NB02 |

<table>
<tr><th>Course, IN02 cell 41</th><th>Ours, NB02 cell 32</th></tr>
<tr><td>

```python
Capital_linear = (df.sort_values(['CNPJ','DATA'])
    .groupby('CNPJ')['Capital']
    .apply(lambda group:
        group.interpolate(method='linear')))
# same for Tier_1, Tier_2, RWA, CRWA
```

</td><td>

```python
interpolated = (df.sort_values([ID_COL, 'DATA'])
    .groupby(ID_COL)[source]
    .apply(lambda group:
        group.interpolate(method='linear',
                          limit_area='inside')))
```

</td></tr>
</table>

**Why.** S4-24 says the result stays NaN at the start or end of a series, *"where `interpolate()`
never extrapolates"*. But the pandas default is `limit_direction='forward'`, which fills every
trailing NaN with the last real value, indefinitely:

```python
>>> pd.Series([1.0, 2.0, np.nan, np.nan]).interpolate(method='linear').tolist()
[1.0, 2.0, 2.0, 2.0]
>>> pd.Series([1.0, 2.0, np.nan, np.nan]).interpolate(method='linear', limit_area='inside').tolist()
[1.0, 2.0, nan, nan]
```

`limit_area='inside'` makes the code do what the slide describes.

**Effect** (conglomerate panel, NB02 cell 34):

| series | filled, `inside` | filled, default | **invented** |
|---|---:|---:|---:|
| `Capital` | 13,485 | 13,630 | **145** |
| `Tier_1` | 12,655 | 12,753 | **98** |
| `Tier_2` | 12,651 | 12,752 | **101** |
| `RWA` | 13,485 | 13,630 | **145** |
| `CRWA` | 8,531 | 12,380 | **3,849** |

- **`CRWA`:** IF.Data publishes no account `79650` after **202306**, so the default freezes each
  institution's June-2023 credit RWA for up to 33 months. For Banco do Brasil it holds 9.43e11
  constant while total RWA moves from 1.11e12 to 1.37e12. Those values feed `A02` directly.
- **`Capital` / `RWA`:** these are the tails of institutions that **stopped filing**, including the
  ones liquidated in 2025. Carrying their last capital position forward keeps a failing bank's BIS
  ratio healthy right to the end of the panel.

---

## V3 · The explicit default flag · *Low*

| | course | ours |
|---|---|---|
| cell | IN02 **31** (`ind_default`), **34** (`tp1…tp12`) | NB02 **15-19** (check: **28**) |
| slides | S4-11 to S4-13 | |
| notebook label | | correction **C11** / panel change **D4** in NB02 |

<table>
<tr><th>Course, IN02 cells 31, 34</th><th>Ours, NB02 cell 19</th></tr>
<tr><td>

```python
def ind_default(row):
    if (row['CNPJ']==253448) and \
       (row['DATA'] == 201804):
        return 1
    else:
        return 0

df['tp1'] = np.where(df['ind_default']==1, 1, 0)
df['tp2'] = (df.sort_values(['CNPJ','DATA'])
    .groupby(['CNPJ'])['ind_default'].shift(-1))
...                          # through tp12
```

</td><td>

```python
# default_flag.xlsx (your answer sheet) is the
# ground truth; ID resolved under both readings,
# stale matches (> 12 months) rejected
for i, e in default_flag.iterrows():
    mask = entity_mask[i]
    liq  = int(e['LIQUIDATION_DATA'])
    window = mask & df['DATA'].between(
        int(e['IND_FIRST_FLAG']),
        add_months(liq, -1))
    df.loc[window, 'ind_default_12m'] = 1
```

</td></tr>
</table>

**Why.** Your `default_flag.xlsx` (sheet `Planilha2`) defines the window as the twelve months
before the **decree**. `shift()` anchors on the last filing instead, and it moves by *row* rather than
by month, so the two agree only where `DELTA == 1`.

**Effect.** `ind_default_12m` goes from **12 → 23 rows** (Dacasa 12, Master 8, Pleno 3).
`shift()` would give 37 rows, and all 14 extra rows are spurious:

- 10 `BANCO VOITER` rows (202306-202403), where the shift walks through a 16-month gap in the series.
- 4 `BANCO MASTER` rows (202407-202410), more than twelve months before the decree.

Because the explicit flag adds only 23 of 1,263 bad events, the bad rate stays at 9.10%.

**Open question.** The FGC register dates Banco Pleno's liquidation **18.02.2025**, while
`default_flag.xlsx` has **202602**. The other nine events agree exactly. We followed the spreadsheet,
since BM Pleno keeps filing COSIF through 202510 and a February-2025 decree wouldn't explain that.
Which date is right?

---

# Notebook 03: categorization and selection

To isolate each change we re-ran our notebook 03 **with exactly one change reverted at a time**.
The baseline reproduces the notebook's saved output exactly (AUC 0.8141 / 0.8636 / 0.6833).

| variant | train AUC | oot AUC | mr AUC | variables |
|---|---:|---:|---:|---|
| **ours** | **0.814** | **0.864** | **0.683** | A01 M01 E01 L01 |
| S1 reverted: add `C01`, `A02` | 0.868 | 0.899 | 0.784 | + C01 A02 |
| S2 reverted: course directions | 0.784 | 0.848 | 0.669 | A01 M01 E01 (`L01` dropped, 0 splits) |
| S3 reverted: course split (train ≤ 202306) | 0.823 | 0.827 *(100 bad)* | 0.676 | A01 M01 E01 L01 |
| S4 reverted: binning fit on whole panel | 0.812 | 0.867 | 0.691 | A01 M01 E01 L01 |

---

## S1 · `C01` and `A02` are built from the target's own data · *High*

| | course | ours |
|---|---|---|
| cell | IN03 **18**; built in IN02 **59**, **62** | NB03 **13** (IV report: **25**) |
| slides | S4-10, S4-29 Q2, S6-9, S6-16, **S7-7**, **S7-8** | |
| notebook label | | correction **C4** in NB03 |

<table>
<tr><th>Course, IN03 cell 18 (IN02 59, 62)</th><th>Ours, NB03 cell 13</th></tr>
<tr><td>

```python
var_asc  = ['A01', 'A02']
var_desc = ['C01', 'M01', 'E01', 'L01']

# IN02
df['C01'] = df['BIS']
df['A02'] = df['CRWA_linear'] / df['31000000']
```

</td><td>

```python
var_asc  = ['A01']
var_desc = ['M01', 'E01', 'L01']

EXCLUDED = {
  'C01': 'target is a threshold on this '
         'same BIS series (circular)',
  'A02': 'built from IF.Data CRWA - same '
         'source as the target; ends 202306'}
# both still built in NB02 and their IV reported
```

</td></tr>
</table>

**Why.**

- **`C01` is the series the target thresholds.** The target is
  `bis_flag = BIS_linear < minimum`, and `ind_default_12m_bis` is forced to 1 whenever today's flag
  is 1. That's the scenario in your discussion question S4-29 Q2.
- **`A02` shares the target's source and missingness.** It's built from IF.Data account `79650`, the
  same report as `BIS`, and it doesn't exist after 202306.

**Effect.** Your saved IN03 output:

| | IV (IN03 cell 30) |
|---|---:|
| `A02` | **4.286** |
| `C01` | **1.488** |
| `L01` | 0.403 |
| `A01` | 0.229 |
| `M01` | 0.165 |
| `E01` | 0.077 |

- **Stepwise path (IN03 cell 41).** Forward selection picks `A02` first and `C01` second. After
  that, `+E01` → `+L01` adds **exactly 0.00000**.
- **Coefficient signs (IN03 cells 45 vs 46).** Dropping `A02` flips `A01_woe` from −0.730 to
  +0.550 and `M01_woe` from −0.514 to +0.618. Pseudo R² falls from 0.664 to 0.246.
- **Session 7, slides 7-8.** These show the cell 46 model, where `C01` has the **largest marginal
  effect** (dy/dx 0.205). We read that as the circularity showing through, not as capital adequacy.
- **After fixing the target (V1):** `A02`'s IV falls to **0.548**, so its power was the missingness.
  `C01`'s only falls to **1.193**, because its link to the target isn't missingness but identity.
- **On our panel (ablation table):** adding both back raises oot AUC 0.864 → **0.899** and mr AUC
  0.683 → **0.784**. `A02` is 0% available in `mr`, so the `mr` gain comes from `C01`. A model that
  improves out of time because it reads the label's own series isn't a better model.

Our four-variable model (NB03 cell 36): all coefficients positive (0.50-0.92), all p ≤ 0.001,
pseudo R² 0.205, N = 5,582.

**Limits.** `E01` has IV 0.966 on train, which is also high. We checked it and kept it:

- The driver is ROE below −5%, with a 46.5% event rate against 2.6% for positive ROE.
- It's built only from COSIF accounts `70000009`, `80000006` and `60000002`, with no IF.Data input.

---

## S2 · The direction check is computed but not used · *Medium*

| | course | ours |
|---|---|---|
| cell | IN03 **18** (assumed), **22-23** (`split_begin_exclude`), **25** (binning loop) | NB03 **17**, **19** |
| slides | S6-8, **S6-10**, S7-6 | |
| notebook label | | correction **C10** in NB03 |

<table>
<tr><th>Course, IN03 cells 22-23</th><th>Ours, NB03 cell 17</th></tr>
<tr><td>

```python
split_begin = {}
for i in var_asc:
    ob = OptimalBinning(...,
        monotonic_trend="ascending", ...)
    ob.fit(x, y)
    split_begin[i] = len(ob.splits.tolist())
for i in var_desc:
    ob = OptimalBinning(...,
        monotonic_trend="descending", ...)
    ...
split_begin_exclude = split_begin_table \
    .query('split==0')['var'].to_list()
# never referenced again; cell 25 loops
# over var_asc / var_desc regardless
```

</td><td>

```python
for variable, assumed in DIRECTION.items():
    for trend in ('ascending', 'descending'):
        ob = OptimalBinning(...,
            monotonic_trend=trend, max_n_bins=6)
        ob.fit(train_lab[variable].values, y_train)
        row['splits_' + trend[:3]] = len(ob.splits)
        row['IV_' + trend[:3]] = ...  # total IV

direction_table['empirical'] = np.where(
    direction_table['IV_asc'] >
    direction_table['IV_des'],
    'ascending', 'descending')
DIRECTION = dict(zip(direction_table['var'],
                     direction_table['empirical']))
```

</td></tr>
</table>

**Why.** A variable that yields no split under a monotonic constraint usually isn't uninformative.
More often the assumed direction is backwards. S6-10 says to *investigate* a *"genuinely
counter-intuitive relationship"*, and S7-6 treats the coefficient sign as *"an automatic sanity
check"*. But WoE is oriented by construction, so the sign can't catch a wrong direction. The check
has to happen at the binning step.

**Effect** (fitted on labelled train rows, NB03 cell 17):

| variable | assumed | asc: splits / IV | desc: splits / IV | adopted |
|---|---|---|---|---|
| `A01` | ascending | 5 / 0.425 | 1 / 0.022 | ascending |
| `M01` | descending | 4 / **0.214** | 1 / 0.057 | **ascending** |
| `E01` | descending | 1 / 0.144 | 4 / **0.966** | descending |
| `L01` | descending | 5 / **0.198** | **0** / 0.016 | **ascending** |

Under the assumed direction, **`L01` produces zero splits and drops out**. With the fitted
directions:

- train AUC 0.784 → **0.814**, oot 0.848 → **0.864**
- forward selection becomes monotone: `E01` 0.721 → `+L01` 0.784 → `+A01` 0.802 → `+M01` 0.814

**Economic reading.**

- **`L01` = cash / deposits (`11000006 / 41000007`).** In the top decile the median institution
  holds more cash than deposits, and deposits are 1.5% of assets: payment institutions and brokers
  with almost no deposit base. The top decile's bad rate is 13.6% against 4.5% in the bottom decile.
  A low cash/deposits ratio marks a genuinely deposit-funded bank, which sits at the safe end of this
  population. Cash far exceeding deposits marks a thin, wholesale-funded or barely-deposit-taking
  institution.
- **`M01` = (non-operating income − non-operating expenses) / assets
  (`73000006`, `83000003` / `10000007`, `20000004`).** The bad rate is U-shaped in `M01` (13.7% in
  the bottom decile, 20.6% in the top), so ascending wins on IV. A large non-operating result
  relative to assets is a distress signal, meaning asset disposals, provision write-backs and one-off
  gains booked to shore up a bad year, which explains rather than explains away why the fitted
  direction comes out ascending.

One note: Session 5 slide 14 describes `M01` as net interest margin. The underlying accounts,
`73000006` and `83000003`, are *receitas/despesas não operacionais* (non-operating income and
expenses) and contain no interest income at all, so `M01` is a non-operating-result ratio rather than
a margin. We use the corrected description throughout this document.

---

## S3 · The train / out-of-time split · *Medium*

| | course | ours |
|---|---|---|
| cell | IN03 **10** | NB03 **8** |
| slides | S5-4 | |
| notebook label | | correction **C6** in NB03 / panel change **D6** |

<table>
<tr><th>Course, IN03 cell 10</th><th>Ours, NB03 cell 8</th></tr>
<tr><td>

```python
def sample(row):
    if (row['DATA']>=201703) & \
       (row['DATA']<=202306):
        return 'train'
    elif (row['DATA']>=202307) & \
         (row['DATA']<=202312):
        return 'oot'
    else:
        return 'mr'
# S5-4 instead: train <= 202506,
#               oot 202507-202512
```

</td><td>

```python
TRAIN_END = 202212
OOT_END   = 202312

def sample(row):
    if (row['DATA'] >= 201703) & \
       (row['DATA'] <= TRAIN_END):
        return 'train'
    elif (row['DATA'] > TRAIN_END) & \
         (row['DATA'] <= OOT_END):
        return 'oot'
    else:
        return 'mr'
```

</td></tr>
</table>

**Why.** Two constraints apply here:

1. **Stay inside COSIF 1.0.** From 202501 (CMN 4.966), `A01` and `M01` are 100% missing. The
   crosswalk `map_cosif_final.xlsx` has no COSIF 1.5 counterpart for `31500005`, `31800004`,
   `31900007`, `73000006` or `83000003`, and the level-4 codes `31000000` maps to are not published
   in `BLOPRUDENCIAL` either. The notebook's split already satisfies this; the slide's split does
   not.
2. **Keep enough out-of-time events.** A 6-month `oot` carries half the bad events of a 12-month
   one, while `mr` absorbs 27 months.

**Effect** (conglomerate panel, rows ≥ 201703):

| split | oot window | oot labelled | **oot bad events** | `A01` / `M01` available in oot |
|---|---|---:|---:|---:|
| IN03 cell 10 | 202307-202312 | 884 | **100** | 70% / 78% |
| S5-4 | 202507-202512 | 934 | **73** | **0% / 0%** |
| ours | 202301-202312 | 1,737 | **207** | 71% / 79% |

Under the notebook's split, oot AUC is 0.827 on 100 events against 0.864 on 207 with ours. `mr`
deliberately straddles the 202501 break, so the Session 8 PSI should flag 202501. That's a true
signal, not a bug.

---

## S4 · Binning and selection are fit on the whole panel · *Medium*

| | course | ours |
|---|---|---|
| cell | IN03 **13** (`y = df['target']`), **19-25** (binning), **32-36** (VIF, correlation), **41-42** (stepwise) | NB03 **9**, **17-19**, **27-33** |
| slides | S5-4 (*"a later period the model never sees during fitting"*) | |
| notebook label | | correction **C5** in NB03 |

<table>
<tr><th>Course, IN03 cells 13, 25</th><th>Ours, NB03 cells 9, 19</th></tr>
<tr><td>

```python
y = df['target']      # train + oot + mr

for i in var_asc:
    x = df[variable].values
    ob = OptimalBinning(...)
    ob.fit(x, y)                  # whole panel
    ...
    ob_snull.fit(x_snull, y)      # whole panel
    df[i+'_woe'] = ob_snull.transform(
        x_snull, metric="woe")
```

</td><td>

```python
train_lab = train[train['modelling_sample']]
y_train   = train_lab['target'].astype(int)

for variable, trend in DIRECTION.items():
    ob.fit(train_lab[variable].values, y_train)
    ...
    ob_snull.fit(x_train_snull, y_train)
    # apply the train-fitted binner everywhere
    x_all = np.where(pd.isnull(df[variable]),
                     missing_sub, df[variable].values)
    df[variable + '_woe'] = ob_snull.transform(
        x_all, metric="woe")
```

</td></tr>
</table>

**Why.** Only the logit coefficients are fit on `train`. Bin edges, WoE values, the missing-value
substitute, VIF, correlation and both stepwise passes all see the `oot` and `mr` periods, which means
the out-of-time AUC isn't fully out of time.

**Effect.** On this panel the effect is **small**: oot AUC is 0.867 with whole-panel fitting and 0.864
with train-only fitting (mr 0.691 vs 0.683). The direction is the expected one, with the leaky version
looking slightly better out of time, but we don't want to overstate the size. We keep the change
because it makes `oot` a clean test by construction, not because it moves the headline much.

---

## S5 · Stepwise scored on `accuracy` · *Low*

| | course | ours |
|---|---|---|
| cell | IN03 **41**, **42** | NB03 **32**, **33** |
| slides | S6-15, S6-16 | |
| notebook label | | correction **C12** in NB03 |

<table>
<tr><th>Course, IN03 cells 41, 42</th><th>Ours, NB03 cells 32, 33</th></tr>
<tr><td>

```python
sfs_forward = SequentialFeatureSelector(
    linear_model.LogisticRegression(),
    k_features=5, forward=True,
    scoring='accuracy', cv=None)

sfs_backward = SequentialFeatureSelector(
    linear_model.LogisticRegression(),
    k_features=5, forward=False,   # of 6
    scoring='accuracy', cv=None)
```

</td><td>

```python
sfs_forward = SequentialFeatureSelector(
    linear_model.LogisticRegression(),
    k_features=len(woe_vars), forward=True,
    scoring='roc_auc', cv=None)

sfs_backward = SequentialFeatureSelector(
    linear_model.LogisticRegression(),
    k_features=1, forward=False,
    scoring='roc_auc', cv=None)
```

</td></tr>
</table>

**Why.**

1. **Accuracy can't rank models at a 9% bad rate.** Predicting "good" for every row already scores
   91.5%, and accuracy depends on a 0.5 cut-off that a scorecard never uses.
2. **`k_features=5` of 6 candidates stops the backward pass after one step.** IN03 cell 42 contains
   only the 6- and 5-variable subsets.

**Effect.**

- **IN03 cell 41 (accuracy):** `0.9009 → 0.9251 → 0.9287 → 0.9287 → 0.9271`, flat and then worse.
- **Ours (AUC):** `E01 0.721 → +L01 0.784 → +A01 0.802 → +M01 0.814`, each step adds.

On the four corrected candidates nothing gets dropped, so the fitted model is unchanged. This
changes how the variable set is justified, not the estimates.

---

# Side note: why our `df_var` has 14,566 rows and yours about 86,000

This explains why we chose not to build on the ~86,000-row stacked frame. It also explains why the
two notebook-03 runs aren't comparable row for row, and part of it may affect your notebook 02.

## The issue in one sentence

**The stack adds rows, but no new target information.** The target is defined on the BIS ratio,
and BIS exists only once per conglomerate per quarter. Each extra `ind` or `soc` row therefore
either **copies** a conglomerate's capital observation or **has none at all**.

We checked this on the levels we have:

```
BIS observed on individual-bank (ind) rows          :  2,972
  ...with no conglomerate                           :      0
  ...identical to the BIS of a pru row, same month  :  2,972   (100%)
BIS observed on conglomerate (pru) rows             :  4,601
```

Not one capital observation on the `ind` rows is new. By construction the same holds for `soc`
rows, which get `BIS` through the same `(DATA, COD_CONGL)` join. So the stacked frame holds the same
number of independent capital observations as our 14,566-row panel, just spread over about six times
as many rows.

## Why that led us to use the conglomerate slice only

| | stacked frame (≈ 86,000 rows) | `pru` slice (14,566 rows) |
|---|---|---|
| independent capital (BIS) observations | same as `pru` | all of them |
| one conglomerate breach counts as | 1 bad event per level it appears at | 1 bad event |
| rows without any capital data | labelled anyway (good or bad, depending on the rule) | ~5%, left unlabelled |
| predictors and target describe | often different entities (member's books, group's capital) | the same entity |
| key for time-series steps | `CNPJ` is not unique per month | `COD_CONGL` is unique per month |

So the extra rows would make the model *look* better supported, through a larger N, tighter
p-values and higher IVs, without giving it any more evidence. The rows that aren't copies are rows
whose label is invented. We preferred a smaller frame where every label is observed exactly once. The
cost is that the model rates conglomerates rather than individual legal entities, and we say so in
the report.

## Details

**Where the rows come from.** `01_DataDownload_student_v2` cell 50 stacks three COSIF extracts:

| `keys` | source | rows |
|---|---|---:|
| `ind` | `BANCOS` + prudential identifiers | 19,258 |
| `soc` | `SOCIEDADES` + prudential identifiers | ≈ 52,200 *(implied: 86,044 − 19,258 − 14,566)* |
| `pru` | `BLOPRUDENCIAL` | 14,566 |
| **total** | cell 91 | **86,044** |

**Our panel is the `pru` slice alone.** You can check it with `df.query('keys == "pru"').shape`.
IN03 only drops `DATA < 201703` (cell 9), so the whole gap carries into categorization.

**What the stack does to notebooks 02 and 03, concretely.** We measured these on the `ind` and
`pru` levels only, since we haven't downloaded `SOCIEDADES`:

1. **The same institution appears at two levels in the same month.** Banco do Brasil is `CNPJ 0`
   in both `ind` and `pru` (v2 cell 89 shows both). In the `ind` + `pru` stack, **17,868 of 33,824
   rows** share their `(CNPJ, DATA)` with another row, across 98 CNPJs.
2. **IN02 still groups by `CNPJ`, not `ID`** (cells 34, 41, 45), even though cell 18 already builds
   `ID`. For an institution present at two levels, the group interleaves two different series month
   by month. `interpolate()` then runs across alternating entities, and `shift(-k)` steps through both
   levels' rows, often landing on the other level's row **from the same month**. So `b1…b12` spans
   roughly six months instead of twelve. Grouping by `['keys', 'ID']` (or by `ID`) separates them.
3. **One capital observation labels several rows.** IF.Data is merged on `(DATA, COD_CONGL)` at
   all three levels, so a conglomerate breach is counted once for the group and again for each
   member row (8,934 such `ind` rows alone). In notebook 03 this overstates the IV and the logit's
   significance, because the effective sample is smaller than the row count. The random
   `train_test_split` in cells 45-46 can also place a row in train and its twin in test.
4. **Rows outside any conglomerate have no capital data.** Most `soc` rows are probably in this
   position, though we can't count them without the download. Under the refined rule they're all
   labelled good, which dilutes the bad rate; under the blanket rule they're all labelled bad.

**A suggestion, really a question:** for categorization, would it make sense to keep one level per
entity? `keys == "pru"` reproduces our frame and keeps predictors and target on the same entity. Or is
the stack meant to be filtered later on?

---

# Notebooks 04 and 05: model development and evaluation

Session 7 was taught from slides rather than an executable course notebook, so there's no instructor
code to port here. Instead the "previous" column below is our own first working draft,
`04_EmpiricalWork_FinalProject.ipynb`, which isn't part of this handover since
`04_ModelDevelopment.ipynb` supersedes it. We list these the same way as the notebook 01-03 changes
above, because the same rule applies: a difference gets an entry if it changes a reported number,
whether the "before" version came from you or from us.

| # | change | severity |
|---|---|---|
| M1 | comparison forced onto identical WoE features and rows | High |
| M2 | no `fillna(0)` on raw CAMELS ratios | High |
| M3 | SGS series 4189, not 432; no bare `except` | High |
| M4 | paired stratified bootstrap on the AUC differences | Medium |
| M5 | placebo test on the macro overlay | Medium |
| M6 | rating scale / PSI flagged as anticipated | Low |
| M7 | inner 70/30 `train_test_split` inside `train` not carried into notebook 04 | Low |

---

## M1 · The parametric/non-parametric comparison must hold the sample fixed · *High*

| | previous draft | ours |
|---|---|---|
| file | `04_EmpiricalWork_FinalProject.ipynb` cells 9, 11 | `04_ModelDevelopment.ipynb`, § model comparison |
| slides | S7-28, S7-29 | |

<table>
<tr><th>Previous draft, cells 9, 11</th><th>Ours</th></tr>
<tr><td>

```python
# 1. Logistic Regression Baseline (TTC)
lr = LogisticRegression(random_state=42,
    class_weight='balanced')
lr.fit(X_train_woe, y_train)          # WoE

# 2. Random Forest Baseline (TTC)
rf = RandomForestClassifier(n_estimators=100,
    max_depth=6, random_state=42,
    class_weight='balanced')
rf.fit(X_train_macro[features_raw],
    y_train)                          # RAW, fillna(0)

# saved output:
# LR Baseline    | OOT AUC: 0.862
# Random Forest  | OOT AUC: 0.816
```

</td><td>

```python
features_woe = ['A01_woe','M01_woe',
                 'E01_woe','L01_woe']
X_train, y_train = prep_Xy('train', features_woe)
X_oot,   y_oot   = prep_Xy('oot',   features_woe)

logit = sm.Logit(y_train,
    sm.add_constant(X_train)).fit()

forest = RandomForestClassifier(n_estimators=500,
    min_samples_leaf=20, class_weight='balanced',
    random_state=42)
forest.fit(X_train, y_train)          # same X_train

# oot AUC: logit 0.8652, forest(d=4) 0.8620
```

</td></tr>
</table>

**Why.** Comparing a logit fitted on WoE-binned features against a forest fitted on raw,
zero-filled ratios confounds the algorithm with the representation. A worse score for the forest could
mean "trees generalise worse" or just "trees got the worse features". Session 7 slide 28 asks which
approach to prefer, and that's only answerable if both models see the same information.

**Effect.** The previous draft reports **LR 0.862 vs. RF 0.816** oot AUC, a 0.046 gap that looks
decisive. Give both models the identical four WoE features and the identical rows (with M2 and M3 also
fixed) and it becomes a **statistical tie**: dAUC +0.0031, 95% CI [-0.0056, +0.0114], p = 0.479
(bootstrap detail in M4). That 0.046 gap was mostly representation, not algorithm.

---

## M2 · `fillna(0)` on a raw CAMELS ratio relabels missing as excellent · *High*

| | previous draft | ours |
|---|---|---|
| file | `04_EmpiricalWork_FinalProject.ipynb` cell 7 | `04_ModelDevelopment.ipynb`, § raw-ratio models |

<table>
<tr><th>Previous draft, cell 7</th><th>Ours</th></tr>
<tr><td>

```python
def prep_Xy(sample_name, feature_list):
    mask = df_lab['sample'] == sample_name
    X = df_lab[mask][feature_list].fillna(0)
    y = df_lab[mask]['target'].astype(int)
    return X, y
# applied to features_raw and
# features_raw_macro alike
```

</td><td>

```python
# native-NaN learner - no imputation at all
hgb = HistGradientBoostingClassifier(
    max_depth=3, random_state=42)
hgb.fit(X_train_raw, y_train)   # NaN passed through

# or: median impute + an explicit flag
for c in features_raw:
    X[c + '_missing'] = X[c].isna().astype(int)
X[features_raw] = X[features_raw].fillna(
    X_train[features_raw].median())
```

</td></tr>
</table>

**Why.** `A01` (asset quality, NPL-based) is **24.2% missing in train and 68.2% missing in mr**,
thanks to the COSIF 1.5 break. Zero NPL is the *best* possible value a ratio like this can take, so
filling a missing observation with 0 tells the model that the institutions with the least capital
documentation are the healthiest ones. That's exactly backwards, and it gets worse across the panel
precisely where data coverage collapses.

**Effect.** With the blanket `fillna(0)` replaced by native-NaN learners, HistGradientBoosting on raw
ratios reaches oot AUC 0.8481 (d=3) and a random forest on raw ratios with median-impute plus missing
flags reaches oot AUC 0.8502 (d=6). Both are real, honestly-obtained numbers instead of a score
computed on relabelled data. It also removes a second, quieter defect: `fillna(0)` was applied
uniformly to `features_raw_macro` too, so SELIC/IPCA/IBC-Br rows with genuinely missing accounting
data got pulled toward "good" along with macro variables that had nothing to do with it.

---

## M3 · SGS series 432 is a daily series and cannot be fetched unbounded · *High*

| | previous draft | ours |
|---|---|---|
| file | `04_EmpiricalWork_FinalProject.ipynb` cell 5 | `04_ModelDevelopment.ipynb`, § macro data |
| doc | | connects to `PROPOSAL.md`, Pillar 1 |

<table>
<tr><th>Previous draft, cell 5</th><th>Ours</th></tr>
<tr><td>

```python
def fetch_bcb_sgs(series_id, name):
    url = f".../bcdata.sgs.{series_id}/dados?formato=json"
    try:
        response = requests.get(url, timeout=10)
        df_macro = pd.DataFrame(response.json())
        ...
        return df_macro[['DATA', name]] \
            .groupby('DATA').last().reset_index()
    except Exception as e:
        print(f"Failed to fetch {name}: {e}")
        return pd.DataFrame(columns=['DATA', name])

df_selic = fetch_bcb_sgs(432, 'SELIC')
# saved output:
# Failed to fetch SELIC: If using all scalar
#   values, you must pass an index
# ... later: "SELIC not found in features."
```

</td><td>

```python
def fetch_bcb_sgs(series_id, name):
    url = f".../bcdata.sgs.{series_id}/dados?formato=json"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    payload = response.json()          # a 406 error
    df_macro = pd.DataFrame(payload)   # dict raises here,
    ...                                # not swallowed
    return df_macro[['DATA', name]] \
        .groupby('DATA').last().reset_index()

df_selic = fetch_bcb_sgs(4189, 'SELIC')  # monthly,
                                          # annualised
```

</td></tr>
</table>

**Why.** Series **432** is BCB's *daily* SELIC series. Requested through the `dados/serie` endpoint
without a date range, it doesn't return the usual `[{"data":...,"valor":...}]` list but a JSON error
object, so `pd.DataFrame(response.json())` raises `ValueError: If using all scalar values, you must
pass an index`. The previous draft's bare `except Exception` caught that, printed a one-line warning
and returned an empty frame, so SELIC was **silently absent** from every downstream model while
`PROPOSAL.md` (Pillar 1) went on naming it as a proposed driver of a point-in-time model. That's
exactly the failure mode the "never silently swallow an exception, raise" rule exists to prevent.
Series **4189** is BCB's monthly, annualised SELIC and returns the normal shape.

**Effect.** With SELIC actually present, Logit WoE + SELIC reaches oot AUC 0.8653 against the
baseline's 0.8652, a negligible lift on its own and consistent with the caveat that oot spans only 6
distinct SELIC values (1.90-13.65 in train against 11.87-13.65 in oot). The point isn't that SELIC
turned out to matter a lot. It's that the previous draft's conclusions cited a variable its own code
had already dropped.

---

## M4 · An AUC difference of 0.003 on 207 events is not a result · *Medium*

**Problem.** The previous draft prints AUC to three decimals and treats every ranking as a finding,
LR vs. RF and RF vs. RF+macro alike, with no notion of whether the gap could be sampling noise. 207
oot bad events isn't a large sample for resolving differences of a few thousandths of an AUC point.

**Ours.** A paired stratified bootstrap on `oot`, B = 4,000: each draw resamples the 1,530 goods
and 207 bads with replacement (paired across models, so both scores are evaluated on the same
resampled rows), and the AUC difference is recorded each time to build a percentile confidence
interval.

```python
rng = np.random.default_rng(42)
def paired_bootstrap(y, p_a, p_b, B=4000):
    goods, bads = np.where(y == 0)[0], np.where(y == 1)[0]
    diffs = np.empty(B)
    for b in range(B):
        idx = np.concatenate([
            rng.choice(goods, len(goods), replace=True),
            rng.choice(bads,  len(bads),  replace=True)])
        diffs[b] = (roc_auc_score(y[idx], p_a[idx])
                    - roc_auc_score(y[idx], p_b[idx]))
    return diffs
```

**Effect.** The three comparisons that decide the report's argument:

| comparison | dAUC | 95% CI | p | verdict |
|---|---:|---|---:|---|
| Logit − Forest(WoE) | +0.0031 | [-0.0056, +0.0114] | 0.479 | not significant |
| Logit − Forest(WoE+macro) | -0.0036 | [-0.0100, +0.0028] | 0.271 | not significant |
| Forest(WoE) − Forest(WoE+macro) | -0.0067 | [-0.0113, -0.0020] | 0.004 | significant |

This is what turns "the forest scored a bit lower" into something defensible. The logit and the
plain-WoE forest are statistically indistinguishable, while adding macro to the forest is a real if
small improvement, and that's exactly the effect M5 goes after.

---

## M5 · The macro overlay needs a placebo · *Medium*

**Problem.** Forest(WoE+macro) beats Forest(WoE) by +0.0067 oot AUC, and M4 shows that gap is
statistically real (p = 0.004). That still isn't the same as showing the gain comes from the macro
series being economically informative. A forest with three extra numeric columns can exploit *any*
smooth function of time, including one with nothing to do with SELIC, IPCA or IBC-Br, simply because
oot is a contiguous later period.

**Our test.** Re-date the macro series to randomly permuted months (same values, shuffled `DATA`
mapping), refit Forest(WoE+macro), and record the placebo AUC gain over Forest(WoE). Repeat 50
times.

```python
real_gain = auc_forest_woe_macro - auc_forest_woe   # +0.0067
placebo_gains = []
for _ in range(50):
    permuted_dates = rng.permutation(sorted(df['DATA'].unique()))
    macro_shuffled = macro_by_date.set_axis(permuted_dates)
    df_placebo = df.merge(macro_shuffled, on='DATA', how='left')
    ...                                   # refit, same features, same split
    placebo_gains.append(auc_placebo - auc_forest_woe)
```

**Effect.**

- real gain from adding macro: **+0.0067**
- placebo gain: mean **+0.0042**, sd 0.0008, range [+0.0023, +0.0058]
- **0 of 50** placebo draws reach the real gain

**Reading.** Roughly **63% (+0.0042 / +0.0067) of the naive macro gain is period structure**, the
forest exploiting *any* time-varying partition whether or not it's real economics. Only about
**+0.0025** is attributable to the macro series being correctly dated. That residual is real, since it
clears every placebo draw and the 0.0067 overall effect is significant per M4, but it's far too small
to overturn the parametric baseline as the primary model. Which is why the report calls the macro/PIT
overlay a measured, mostly-null result rather than a win. That's the opposite of what
`PROPOSAL.md`'s Pillar 1 claimed, written before this test existed.

---

## M6 · Anticipating Session 8 · *Low*

`05_EvaluationMonitoring.ipynb` assigns the 1-8 rating scale to `predicted_good`
(≤0.17→8 ... else→1) and computes PSI with alarm lines at 0.10/0.25. Session 8 (Evaluation,
Monitoring, Deployment) hadn't been taught when we built this, so both the notebook and the report
**label this material as anticipated rather than examined-and-corrected** the way notebooks 01-03
are. There's no course version to compare it against, and no numbers change because of a course/ours
difference here. It's included because the assignment asks for the model to be shown end to end, not
because we're claiming to have pre-empted Session 8's content.

---

## M7 · The inner 70/30 random split inside `train` · *Low*

| | notebook 03 (kept for comparability) | notebook 04 |
|---|---|---|
| cell | NB03 **36** (course: IN03 **45**) | `04_ModelDevelopment.ipynb`, § logit refit |
| doc | open issue **O4**, below | |

<table>
<tr><th>Course / NB03 cell 36</th><th>Ours, NB04</th></tr>
<tr><td>

```python
X_train_one, X_test_one, \
y_train_one, y_test_one = train_test_split(
    train_lab,
    train_lab['y_adj'].astype(float),
    test_size=0.30, random_state=42)

logReg = sm.Logit(y_train_one,
    sm.add_constant(X_train_one[var_woe]))
# fits on 70% of the 7,975 train rows
```

</td><td>

```python
# no inner split - oot is already
# the time-based holdout
logit = sm.Logit(y_train,
    sm.add_constant(X_train[var_woe]))
# fits on all 7,975 train rows
```

</td></tr>
</table>

**Why.** Open issue **O4** below records this on the notebook 03 side.
The instructor's `train_test_split(test_size=0.30, random_state=42)` inside `train` contradicts
Session 5 slide 4's argument that splitting must be by time, and the 30% holdout it produces never
gets evaluated, since the model is scored on `df`. We kept it in notebook 03 anyway, purely so that
notebook's printed AUC stays comparable to the instructor's cell for cell. Notebook 04 is a different
situation. It exists specifically to compare the logit against the tree and the forest on identical
data (M1), and with only 680 bad events in `train`, throwing away 30% of them penalises the tree and
the forest, which need more rows to stabilise their splits, more than it penalises the logit.
Carrying the random split into notebook 04 would bias the very comparison the notebook is built to
settle, so notebook 04 refits the logit on the full 7,975 labelled `train` rows, the same rows the
tree and forest are trained on.

**Effect.** The two logits aren't the same fit, so their oot AUCs aren't identical. The difference is
small enough that a reader shouldn't mistake it for a new finding:

| | rows fitted on | oot AUC |
|---|---:|---:|
| notebook 03 (`train_test_split`, 70%) | 5,582 | **0.8636** |
| notebook 04 (full `train`) | 7,975 | **0.8652** |

The gap is 0.0016, well inside the bootstrap's confidence intervals in M4 (the narrowest spans
0.0126), so it changes no conclusion. Notebook 03 stays exactly as it is, split and all, because its
job is to stay comparable to the instructor's notebook. Notebook 04's job is to compare models
fairly, which is a different job with a different answer.

---

# Session 7: how the new material relates to these changes

- **S7-7 (logit output)** is your IN03 cell 46: `C01 A01 M01 E01 L01`, N = 9,239, pseudo
  R² 0.246, all coefficients positive. Our version drops `C01` (S1): four positive coefficients,
  pseudo R² 0.205, N = 5,582.
- **S7-8 (marginal effects)** shows `C01` with the largest dy/dx (0.205). Under S1 we'd attribute
  that to the target being defined on the same series.
- **S7-6 (sign checks)**: positive signs follow from the WoE orientation, so they can't reveal a
  wrong assumed direction. That's why the check sits in the binning step (S2).
- **S7-28/29 (parametric vs. non-parametric)**: we ran the tree and random-forest comparison on the
  same train-only preprocessing, the same labelled rows and the same WoE features as the logit, so
  any AUC difference comes from the model rather than the sample (see M1). It came out a statistical
  tie, dAUC +0.0031 at p = 0.479, not a win for either side. The logit's oot AUC of 0.865 clears the
  0.84 benchmark on S7-29, and the forest reaches 0.862 on the same rows. That's close enough that
  Session 7 slide 28's other four dimensions (interpretability, overfitting risk, governance fit, use
  case) are what actually decide between them, not discrimination. Full detail and the macro-overlay
  placebo test are in the section above.


