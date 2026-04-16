# Divergence Analysis Report

**Dataset:** `datasets/test/aggregated_elon_tweets.csv` | **N:** 402  
**Models:** `bilstm-attention`, `bilstm`, `cardiff_roberta`, `finetuned_roberta`, `gemma_4-31B`, `logistic_regression`, `naive_bayes`, `weighted-naive-bayes-lr`  
**Label distribution:** negative: 114 | neutral: 162 | positive: 126

## Model Performance

Accuracy and macro-averaged F1 on the annotated tweet set. ECE (Expected Calibration Error) measures how well predicted probabilities reflect true likelihoods — lower is better calibrated.

| model                     |   accuracy |   macro_f1 |    ece |
|:--------------------------|-----------:|-----------:|-------:|
| `bilstm-attention`        |     0.5547 |     0.5534 | 0.119  |
| `bilstm`                  |     0.5448 |     0.5344 | 0.1053 |
| `cardiff_roberta`         |     0.7413 |     0.7432 | 0.0432 |
| `finetuned_roberta`       |     0.6393 |     0.6338 | 0.205  |
| `gemma_4-31B`             |     0.8284 |     0.8302 | 0.1604 |
| `logistic_regression`     |     0.5224 |     0.5171 | 0.0631 |
| `naive_bayes`             |     0.4726 |     0.4745 | 0.0552 |
| `weighted-naive-bayes-lr` |     0.5597 |     0.5412 | 0.0473 |

### Full Classification Reports

<details><summary><code>bilstm-attention</code></summary>

```
precision    recall  f1-score   support

    negative       0.47      0.72      0.57       114
     neutral       0.61      0.51      0.56       162
    positive       0.65      0.46      0.54       126

    accuracy                           0.55       402
   macro avg       0.57      0.56      0.55       402
weighted avg       0.58      0.55      0.55       402
```

</details>

<details><summary><code>bilstm</code></summary>

```
precision    recall  f1-score   support

    negative       0.50      0.57      0.53       114
     neutral       0.52      0.66      0.58       162
    positive       0.71      0.37      0.49       126

    accuracy                           0.54       402
   macro avg       0.58      0.53      0.53       402
weighted avg       0.57      0.54      0.54       402
```

</details>

<details><summary><code>cardiff_roberta</code></summary>

```
precision    recall  f1-score   support

    negative       0.74      0.84      0.79       114
     neutral       0.69      0.76      0.72       162
    positive       0.84      0.63      0.72       126

    accuracy                           0.74       402
   macro avg       0.76      0.74      0.74       402
weighted avg       0.75      0.74      0.74       402
```

</details>

<details><summary><code>finetuned_roberta</code></summary>

```
precision    recall  f1-score   support

    negative       0.75      0.57      0.65       114
     neutral       0.55      0.81      0.66       162
    positive       0.78      0.48      0.60       126

    accuracy                           0.64       402
   macro avg       0.69      0.62      0.63       402
weighted avg       0.68      0.64      0.64       402
```

</details>

<details><summary><code>gemma_4-31B</code></summary>

```
precision    recall  f1-score   support

    negative       0.82      0.92      0.87       114
     neutral       0.84      0.78      0.81       162
    positive       0.82      0.80      0.81       126

    accuracy                           0.83       402
   macro avg       0.83      0.84      0.83       402
weighted avg       0.83      0.83      0.83       402
```

</details>

<details><summary><code>logistic_regression</code></summary>

```
precision    recall  f1-score   support

    negative       0.47      0.47      0.47       114
     neutral       0.51      0.60      0.55       162
    positive       0.61      0.47      0.53       126

    accuracy                           0.52       402
   macro avg       0.53      0.51      0.52       402
weighted avg       0.53      0.52      0.52       402
```

</details>

<details><summary><code>naive_bayes</code></summary>

```
precision    recall  f1-score   support

    negative       0.42      0.53      0.47       114
     neutral       0.48      0.38      0.42       162
    positive       0.52      0.55      0.53       126

    accuracy                           0.47       402
   macro avg       0.47      0.48      0.47       402
weighted avg       0.48      0.47      0.47       402
```

</details>

<details><summary><code>weighted-naive-bayes-lr</code></summary>

```
precision    recall  f1-score   support

    negative       0.57      0.41      0.48       114
     neutral       0.51      0.77      0.61       162
    positive       0.73      0.42      0.53       126

    accuracy                           0.56       402
   macro avg       0.60      0.53      0.54       402
weighted avg       0.59      0.56      0.55       402
```

</details>

## Pairwise Agreement Rate

Fraction of tweets where both models predict the same label. 1.0 = perfect agreement, 0.0 = no agreement.

|                           | `bilstm-attention`   | `bilstm`   | `cardiff_roberta`   | `finetuned_roberta`   | `gemma_4-31B`   | `logistic_regression`   | `naive_bayes`   | `weighted-naive-bayes-lr`   |
|:--------------------------|:---------------------|:-----------|:--------------------|:----------------------|:----------------|:------------------------|:----------------|:----------------------------|
| `bilstm-attention`        | —                    | 0.7910     | 0.6294              | 0.5995                | 0.5771          | 0.6169                  | 0.5746          | 0.6343                      |
| `bilstm`                  | 0.7910               | —          | 0.6468              | 0.6791                | 0.5522          | 0.6418                  | 0.5721          | 0.7289                      |
| `cardiff_roberta`         | 0.6294               | 0.6468     | —                   | 0.7786                | 0.7512          | 0.5721                  | 0.5299          | 0.6592                      |
| `finetuned_roberta`       | 0.5995               | 0.6791     | 0.7786              | —                     | 0.6617          | 0.5871                  | 0.5124          | 0.6965                      |
| `gemma_4-31B`             | 0.5771               | 0.5522     | 0.7512              | 0.6617                | —               | 0.5299                  | 0.4876          | 0.5522                      |
| `logistic_regression`     | 0.6169               | 0.6418     | 0.5721              | 0.5871                | 0.5299          | —                       | 0.6542          | 0.7488                      |
| `naive_bayes`             | 0.5746               | 0.5721     | 0.5299              | 0.5124                | 0.4876          | 0.6542                  | —               | 0.5945                      |
| `weighted-naive-bayes-lr` | 0.6343               | 0.7289     | 0.6592              | 0.6965                | 0.5522          | 0.7488                  | 0.5945          | —                           |

## Pairwise Cohen's Kappa

Agreement rate corrected for chance. > 0.6 is generally considered substantial agreement; values near 0 indicate agreement no better than random.

|                           | `bilstm-attention`   | `bilstm`   | `cardiff_roberta`   | `finetuned_roberta`   | `gemma_4-31B`   | `logistic_regression`   | `naive_bayes`   | `weighted-naive-bayes-lr`   |
|:--------------------------|:---------------------|:-----------|:--------------------|:----------------------|:----------------|:------------------------|:----------------|:----------------------------|
| `bilstm-attention`        | —                    | 0.6771     | 0.4350              | 0.3944                | 0.3639          | 0.4196                  | 0.3590          | 0.4469                      |
| `bilstm`                  | 0.6771               | —          | 0.4393              | 0.4625                | 0.3158          | 0.4283                  | 0.3609          | 0.5407                      |
| `cardiff_roberta`         | 0.4350               | 0.4393     | —                   | 0.6444                | 0.6225          | 0.3332                  | 0.2969          | 0.4490                      |
| `finetuned_roberta`       | 0.3944               | 0.4625     | 0.6444              | —                     | 0.4797          | 0.3270                  | 0.2766          | 0.4565                      |
| `gemma_4-31B`             | 0.3639               | 0.3158     | 0.6225              | 0.4797                | —               | 0.2852                  | 0.2326          | 0.3098                      |
| `logistic_regression`     | 0.4196               | 0.4283     | 0.3332              | 0.3270                | 0.2852          | —                       | 0.4840          | 0.5871                      |
| `naive_bayes`             | 0.3590               | 0.3609     | 0.2969              | 0.2766                | 0.2326          | 0.4840                  | —               | 0.3991                      |
| `weighted-naive-bayes-lr` | 0.4469               | 0.5407     | 0.4490              | 0.4565                | 0.3098          | 0.5871                  | 0.3991          | —                           |

## Pairwise JS Divergence (soft probabilities)

Average Jensen-Shannon divergence between each pair's predicted probability distributions. Ranges from 0 (identical distributions) to 1 (maximally different). Captures disagreement in confidence, not just in the predicted label.

|                           | `bilstm-attention`   | `bilstm`   | `cardiff_roberta`   | `finetuned_roberta`   | `gemma_4-31B`   | `logistic_regression`   | `naive_bayes`   | `weighted-naive-bayes-lr`   |
|:--------------------------|:---------------------|:-----------|:--------------------|:----------------------|:----------------|:------------------------|:----------------|:----------------------------|
| `bilstm-attention`        | —                    | 0.1207     | 0.2582              | 0.3327                | 0.4692          | 0.1838                  | 0.1931          | 0.1914                      |
| `bilstm`                  | 0.1207               | —          | 0.2439              | 0.3025                | 0.4711          | 0.1916                  | 0.2213          | 0.1652                      |
| `cardiff_roberta`         | 0.2582               | 0.2439     | —                   | 0.2188                | 0.3571          | 0.2871                  | 0.3185          | 0.2520                      |
| `finetuned_roberta`       | 0.3327               | 0.3025     | 0.2188              | —                     | 0.3475          | 0.3575                  | 0.3985          | 0.3108                      |
| `gemma_4-31B`             | 0.4692               | 0.4711     | 0.3571              | 0.3475                | —               | 0.4934                  | 0.5309          | 0.4883                      |
| `logistic_regression`     | 0.1838               | 0.1916     | 0.2871              | 0.3575                | 0.4934          | —                       | 0.1482          | 0.1504                      |
| `naive_bayes`             | 0.1931               | 0.2213     | 0.3185              | 0.3985                | 0.5309          | 0.1482                  | —               | 0.1838                      |
| `weighted-naive-bayes-lr` | 0.1914               | 0.1652     | 0.2520              | 0.3108                | 0.4883          | 0.1504                  | 0.1838          | —                           |

## McNemar's Test

Tests whether two models make errors on *different* subsets of tweets (i.e. their error patterns are statistically distinct). A significant result (p < 0.05) means the models are not just interchangeable — one is correcting errors the other makes.

| pair                                               |   statistic |    p-value | significant (p<0.05)   |
|:---------------------------------------------------|------------:|-----------:|:-----------------------|
| `bilstm-attention` vs `bilstm`                     |      0.1406 | 0.70766    | no                     |
| `bilstm-attention` vs `cardiff_roberta`            |     43.1181 | 5.1533e-11 | yes                    |
| `bilstm-attention` vs `finetuned_roberta`          |      8.25   | 0.0040752  | yes                    |
| `bilstm-attention` vs `gemma_4-31B`                |     78.1645 | 9.4808e-19 | yes                    |
| `bilstm-attention` vs `logistic_regression`        |      1.2973 | 0.25471    | no                     |
| `bilstm-attention` vs `naive_bayes`                |      7.6992 | 0.0055244  | yes                    |
| `bilstm-attention` vs `weighted-naive-bayes-lr`    |      0.0088 | 0.92538    | no                     |
| `bilstm` vs `cardiff_roberta`                      |     48.672  | 3.0256e-12 | yes                    |
| `bilstm` vs `finetuned_roberta`                    |     12.6759 | 0.00037039 | yes                    |
| `bilstm` vs `gemma_4-31B`                          |     77.8598 | 1.1062e-18 | yes                    |
| `bilstm` vs `logistic_regression`                  |      0.5766 | 0.44766    | no                     |
| `bilstm` vs `naive_bayes`                          |      5.8947 | 0.015186   | yes                    |
| `bilstm` vs `weighted-naive-bayes-lr`              |      0.2841 | 0.59403    | no                     |
| `cardiff_roberta` vs `finetuned_roberta`           |     19.7531 | 8.8119e-06 | yes                    |
| `cardiff_roberta` vs `gemma_4-31B`                 |     12.4301 | 0.00042247 | yes                    |
| `cardiff_roberta` vs `logistic_regression`         |     48.5192 | 3.2707e-12 | yes                    |
| `cardiff_roberta` vs `naive_bayes`                 |     68.1488 | 1.5161e-16 | yes                    |
| `cardiff_roberta` vs `weighted-naive-bayes-lr`     |     39.5725 | 3.161e-10  | yes                    |
| `finetuned_roberta` vs `gemma_4-31B`               |     43.9453 | 3.3768e-11 | yes                    |
| `finetuned_roberta` vs `logistic_regression`       |     15.6741 | 7.5249e-05 | yes                    |
| `finetuned_roberta` vs `naive_bayes`               |     26.7239 | 2.347e-07  | yes                    |
| `finetuned_roberta` vs `weighted-naive-bayes-lr`   |      8.8981 | 0.0028546  | yes                    |
| `gemma_4-31B` vs `logistic_regression`             |     88.071  | 6.3146e-21 | yes                    |
| `gemma_4-31B` vs `naive_bayes`                     |    106.688  | 5.2112e-25 | yes                    |
| `gemma_4-31B` vs `weighted-naive-bayes-lr`         |     67.3471 | 2.2768e-16 | yes                    |
| `logistic_regression` vs `naive_bayes`             |      3.2232 | 0.072601   | no                     |
| `logistic_regression` vs `weighted-naive-bayes-lr` |      2.6133 | 0.10597    | no                     |
| `naive_bayes` vs `weighted-naive-bayes-lr`         |      8.9612 | 0.0027577  | yes                    |

## Error Overlap

Jaccard similarity of the two models' error sets (tweets both got wrong vs. the union of all errors). High Jaccard means the models tend to fail on the same tweets; low Jaccard means their errors are complementary, suggesting an ensemble could help.

| pair                                               |   jaccard |   only-a errors |   only-b errors |   shared errors |   total-a errors |   total-b errors |
|:---------------------------------------------------|----------:|----------------:|----------------:|----------------:|-----------------:|-----------------:|
| `bilstm-attention` vs `bilstm`                     |    0.6995 |              30 |              34 |             149 |              179 |              183 |
| `bilstm-attention` vs `cardiff_roberta`            |    0.3805 |             101 |              26 |              78 |              179 |              104 |
| `bilstm-attention` vs `finetuned_roberta`          |    0.4211 |              83 |              49 |              96 |              179 |              145 |
| `bilstm-attention` vs `gemma_4-31B`                |    0.24   |             131 |              21 |              48 |              179 |               69 |
| `bilstm-attention` vs `logistic_regression`        |    0.5394 |              49 |              62 |             130 |              179 |              192 |
| `bilstm-attention` vs `naive_bayes`                |    0.4924 |              50 |              83 |             129 |              179 |              212 |
| `bilstm-attention` vs `weighted-naive-bayes-lr`    |    0.5149 |              58 |              56 |             121 |              179 |              177 |
| `bilstm` vs `cardiff_roberta`                      |    0.3932 |             102 |              23 |              81 |              183 |              104 |
| `bilstm` vs `finetuned_roberta`                    |    0.5046 |              73 |              35 |             110 |              183 |              145 |
| `bilstm` vs `gemma_4-31B`                          |    0.2115 |             139 |              25 |              44 |              183 |               69 |
| `bilstm` vs `logistic_regression`                  |    0.5432 |              51 |              60 |             132 |              183 |              192 |
| `bilstm` vs `naive_bayes`                          |    0.4962 |              52 |              81 |             131 |              183 |              212 |
| `bilstm` vs `weighted-naive-bayes-lr`              |    0.6071 |              47 |              41 |             136 |              183 |              177 |
| `cardiff_roberta` vs `finetuned_roberta`           |    0.5091 |              20 |              61 |              84 |              104 |              145 |
| `cardiff_roberta` vs `gemma_4-31B`                 |    0.3008 |              64 |              29 |              40 |              104 |               69 |
| `cardiff_roberta` vs `logistic_regression`         |    0.3097 |              34 |             122 |              70 |              104 |              192 |
| `cardiff_roberta` vs `naive_bayes`                 |    0.3058 |              30 |             138 |              74 |              104 |              212 |
| `cardiff_roberta` vs `weighted-naive-bayes-lr`     |    0.3641 |              29 |             102 |              75 |              104 |              177 |
| `finetuned_roberta` vs `gemma_4-31B`               |    0.2515 |             102 |              26 |              43 |              145 |               69 |
| `finetuned_roberta` vs `logistic_regression`       |    0.428  |              44 |              91 |             101 |              145 |              192 |
| `finetuned_roberta` vs `naive_bayes`               |    0.3731 |              48 |             115 |              97 |              145 |              212 |
| `finetuned_roberta` vs `weighted-naive-bayes-lr`   |    0.4977 |              38 |              70 |             107 |              145 |              177 |
| `gemma_4-31B` vs `logistic_regression`             |    0.214  |              23 |             146 |              46 |               69 |              192 |
| `gemma_4-31B` vs `naive_bayes`                     |    0.1957 |              23 |             166 |              46 |               69 |              212 |
| `gemma_4-31B` vs `weighted-naive-bayes-lr`         |    0.1827 |              31 |             139 |              38 |               69 |              177 |
| `logistic_regression` vs `naive_bayes`             |    0.5659 |              46 |              66 |             146 |              192 |              212 |
| `logistic_regression` vs `weighted-naive-bayes-lr` |    0.6622 |              45 |              30 |             147 |              192 |              177 |
| `naive_bayes` vs `weighted-naive-bayes-lr`         |    0.5019 |              82 |              47 |             130 |              212 |              177 |

## Per-Pair Details

Detailed breakdown for each model pair: where they agree by class, how their predictions cross over, and which linguistic features drive divergence.

### `bilstm-attention` vs `bilstm`

**Per-class agreement** — agreement rate split by gold label, showing which sentiment class the two models disagree on most.

| class    |   n |   agreement |
|:---------|----:|------------:|
| negative | 114 |      0.7895 |
| neutral  | 162 |      0.8086 |
| positive | 126 |      0.7698 |

**Prediction cross-tabulation** — rows are `bilstm-attention` predictions, columns are `bilstm` predictions. Off-diagonal cells are disagreements.

| bilstm-attention \ bilstm   |   negative |   neutral |   positive |
|:----------------------------|-----------:|----------:|-----------:|
| negative                    |        124 |        48 |          4 |
| neutral                     |          3 |       133 |          1 |
| positive                    |          4 |        24 |         61 |

**Linguistic feature divergence** — for each surface feature, how much it shifts agreement between the two models (Δagreement) and each model's macro-F1 on tweets containing that feature. Sorted by Δagreement ascending so the most divergence-inducing features appear first.

| feature        |   count |   pct |   agreement |   agreement_without |   delta_agreement |   f1_bilstm-attention |   f1_bilstm |   f1_delta (b-a) |
|:---------------|--------:|------:|------------:|--------------------:|------------------:|----------------------:|------------:|-----------------:|
| exclamation    |      61 |  15.2 |      0.6721 |              0.8123 |           -0.1402 |                0.4467 |      0.3582 |          -0.0886 |
| question       |      24 |   6   |      0.7083 |              0.7963 |           -0.088  |                0.7146 |      0.5537 |          -0.161  |
| negation       |      74 |  18.4 |      0.7568 |              0.7988 |           -0.042  |                0.5094 |      0.5533 |           0.0439 |
| emoji          |       9 |   2.2 |      0.8889 |              0.7888 |            0.1001 |                0.6794 |      0.6794 |           0      |
| allcaps        |      18 |   4.5 |      0.8889 |              0.7865 |            0.1024 |                0.5556 |      0.475  |          -0.0806 |
| superlative    |      31 |   7.7 |      0.9032 |              0.7817 |            0.1216 |                0.5172 |      0.606  |           0.0888 |
| short_text     |     123 |  30.6 |      0.8943 |              0.7455 |            0.1488 |                0.6077 |      0.5711 |          -0.0366 |
| emoticon       |       1 |   0.2 |      1      |              0.7905 |            0.2095 |                1      |      1      |           0      |
| sarcasm_marker |       5 |   1.2 |      1      |              0.7884 |            0.2116 |                1      |      1      |           0      |

### `bilstm-attention` vs `cardiff_roberta`

**Per-class agreement** — agreement rate split by gold label, showing which sentiment class the two models disagree on most.

| class    |   n |   agreement |
|:---------|----:|------------:|
| negative | 114 |      0.6842 |
| neutral  | 162 |      0.6173 |
| positive | 126 |      0.5952 |

**Prediction cross-tabulation** — rows are `bilstm-attention` predictions, columns are `cardiff_roberta` predictions. Off-diagonal cells are disagreements.

| bilstm-attention \ cardiff_roberta   |   negative |   neutral |   positive |
|:-------------------------------------|-----------:|----------:|-----------:|
| negative                             |         95 |        61 |         20 |
| neutral                              |         25 |        98 |         14 |
| positive                             |          9 |        20 |         60 |

**Linguistic feature divergence** — for each surface feature, how much it shifts agreement between the two models (Δagreement) and each model's macro-F1 on tweets containing that feature. Sorted by Δagreement ascending so the most divergence-inducing features appear first.

| feature        |   count |   pct |   agreement |   agreement_without |   delta_agreement |   f1_bilstm-attention |   f1_cardiff_roberta |   f1_delta (b-a) |
|:---------------|--------:|------:|------------:|--------------------:|------------------:|----------------------:|---------------------:|-----------------:|
| emoticon       |       1 |   0.2 |      0      |              0.6309 |           -0.6309 |                1      |               0      |          -1      |
| emoji          |       9 |   2.2 |      0.5556 |              0.631  |           -0.0755 |                0.6794 |               0.6627 |          -0.0167 |
| sarcasm_marker |       5 |   1.2 |      0.6    |              0.6297 |           -0.0297 |                1      |               0.6111 |          -0.3889 |
| exclamation    |      61 |  15.2 |      0.6066 |              0.6334 |           -0.0269 |                0.4467 |               0.7545 |           0.3078 |
| negation       |      74 |  18.4 |      0.6351 |              0.628  |            0.0071 |                0.5094 |               0.6767 |           0.1673 |
| allcaps        |      18 |   4.5 |      0.6667 |              0.6276 |            0.0391 |                0.5556 |               0.6898 |           0.1342 |
| question       |      24 |   6   |      0.7083 |              0.6243 |            0.084  |                0.7146 |               0.8344 |           0.1198 |
| short_text     |     123 |  30.6 |      0.6992 |              0.5986 |            0.1006 |                0.6077 |               0.818  |           0.2103 |
| superlative    |      31 |   7.7 |      0.7419 |              0.6199 |            0.122  |                0.5172 |               0.694  |           0.1768 |

### `bilstm-attention` vs `finetuned_roberta`

**Per-class agreement** — agreement rate split by gold label, showing which sentiment class the two models disagree on most.

| class    |   n |   agreement |
|:---------|----:|------------:|
| negative | 114 |      0.5877 |
| neutral  | 162 |      0.6235 |
| positive | 126 |      0.5794 |

**Prediction cross-tabulation** — rows are `bilstm-attention` predictions, columns are `finetuned_roberta` predictions. Off-diagonal cells are disagreements.

| bilstm-attention \ finetuned_roberta   |   negative |   neutral |   positive |
|:---------------------------------------|-----------:|----------:|-----------:|
| negative                               |         71 |        89 |         16 |
| neutral                                |         11 |       117 |          9 |
| positive                               |          5 |        31 |         53 |

**Linguistic feature divergence** — for each surface feature, how much it shifts agreement between the two models (Δagreement) and each model's macro-F1 on tweets containing that feature. Sorted by Δagreement ascending so the most divergence-inducing features appear first.

| feature        |   count |   pct |   agreement |   agreement_without |   delta_agreement |   f1_bilstm-attention |   f1_finetuned_roberta |   f1_delta (b-a) |
|:---------------|--------:|------:|------------:|--------------------:|------------------:|----------------------:|-----------------------:|-----------------:|
| emoticon       |       1 |   0.2 |      0      |              0.601  |           -0.601  |                1      |                 0      |          -1      |
| emoji          |       9 |   2.2 |      0.4444 |              0.6031 |           -0.1586 |                0.6794 |                 0.3556 |          -0.3238 |
| exclamation    |      61 |  15.2 |      0.4918 |              0.6188 |           -0.127  |                0.4467 |                 0.5829 |           0.1362 |
| negation       |      74 |  18.4 |      0.527  |              0.6159 |           -0.0888 |                0.5094 |                 0.5961 |           0.0867 |
| sarcasm_marker |       5 |   1.2 |      0.6    |              0.5995 |            0.0005 |                1      |                 0.6111 |          -0.3889 |
| allcaps        |      18 |   4.5 |      0.6111 |              0.599  |            0.0122 |                0.5556 |                 0.5661 |           0.0105 |
| superlative    |      31 |   7.7 |      0.6129 |              0.5984 |            0.0145 |                0.5172 |                 0.6186 |           0.1014 |
| question       |      24 |   6   |      0.7083 |              0.5926 |            0.1157 |                0.7146 |                 0.7605 |           0.0458 |
| short_text     |     123 |  30.6 |      0.7236 |              0.5448 |            0.1788 |                0.6077 |                 0.7173 |           0.1096 |

### `bilstm-attention` vs `gemma_4-31B`

**Per-class agreement** — agreement rate split by gold label, showing which sentiment class the two models disagree on most.

| class    |   n |   agreement |
|:---------|----:|------------:|
| negative | 114 |      0.7281 |
| neutral  | 162 |      0.537  |
| positive | 126 |      0.4921 |

**Prediction cross-tabulation** — rows are `bilstm-attention` predictions, columns are `gemma_4-31B` predictions. Off-diagonal cells are disagreements.

| bilstm-attention \ gemma_4-31B   |   negative |   neutral |   positive |
|:---------------------------------|-----------:|----------:|-----------:|
| negative                         |         89 |        47 |         40 |
| neutral                          |         27 |        85 |         25 |
| positive                         |         12 |        19 |         58 |

**Linguistic feature divergence** — for each surface feature, how much it shifts agreement between the two models (Δagreement) and each model's macro-F1 on tweets containing that feature. Sorted by Δagreement ascending so the most divergence-inducing features appear first.

| feature        |   count |   pct |   agreement |   agreement_without |   delta_agreement |   f1_bilstm-attention |   f1_gemma_4-31B |   f1_delta (b-a) |
|:---------------|--------:|------:|------------:|--------------------:|------------------:|----------------------:|-----------------:|-----------------:|
| exclamation    |      61 |  15.2 |      0.4918 |              0.5924 |           -0.1006 |                0.4467 |           0.7968 |           0.35   |
| emoji          |       9 |   2.2 |      0.5556 |              0.5776 |           -0.0221 |                0.6794 |           0.65   |          -0.0294 |
| sarcasm_marker |       5 |   1.2 |      0.6    |              0.5768 |            0.0232 |                1      |           0.6111 |          -0.3889 |
| negation       |      74 |  18.4 |      0.6081 |              0.5701 |            0.038  |                0.5094 |           0.7959 |           0.2865 |
| superlative    |      31 |   7.7 |      0.6129 |              0.5741 |            0.0388 |                0.5172 |           0.7774 |           0.2602 |
| allcaps        |      18 |   4.5 |      0.6667 |              0.5729 |            0.0938 |                0.5556 |           0.7103 |           0.1548 |
| short_text     |     123 |  30.6 |      0.6667 |              0.5376 |            0.129  |                0.6077 |           0.9206 |           0.3128 |
| question       |      24 |   6   |      0.7083 |              0.5688 |            0.1396 |                0.7146 |           0.8719 |           0.1572 |
| emoticon       |       1 |   0.2 |      1      |              0.5761 |            0.4239 |                1      |           1      |           0      |

### `bilstm-attention` vs `logistic_regression`

**Per-class agreement** — agreement rate split by gold label, showing which sentiment class the two models disagree on most.

| class    |   n |   agreement |
|:---------|----:|------------:|
| negative | 114 |      0.5877 |
| neutral  | 162 |      0.6543 |
| positive | 126 |      0.5952 |

**Prediction cross-tabulation** — rows are `bilstm-attention` predictions, columns are `logistic_regression` predictions. Off-diagonal cells are disagreements.

| bilstm-attention \ logistic_regression   |   negative |   neutral |   positive |
|:-----------------------------------------|-----------:|----------:|-----------:|
| negative                                 |         89 |        65 |         22 |
| neutral                                  |         19 |       101 |         17 |
| positive                                 |          8 |        23 |         58 |

**Linguistic feature divergence** — for each surface feature, how much it shifts agreement between the two models (Δagreement) and each model's macro-F1 on tweets containing that feature. Sorted by Δagreement ascending so the most divergence-inducing features appear first.

| feature        |   count |   pct |   agreement |   agreement_without |   delta_agreement |   f1_bilstm-attention |   f1_logistic_regression |   f1_delta (b-a) |
|:---------------|--------:|------:|------------:|--------------------:|------------------:|----------------------:|-------------------------:|-----------------:|
| question       |      24 |   6   |      0.4583 |              0.627  |           -0.1687 |                0.7146 |                   0.467  |          -0.2477 |
| negation       |      74 |  18.4 |      0.5541 |              0.6311 |           -0.077  |                0.5094 |                   0.4709 |          -0.0385 |
| allcaps        |      18 |   4.5 |      0.5556 |              0.6198 |           -0.0642 |                0.5556 |                   0.5427 |          -0.0128 |
| exclamation    |      61 |  15.2 |      0.6393 |              0.6129 |            0.0264 |                0.4467 |                   0.4251 |          -0.0216 |
| short_text     |     123 |  30.6 |      0.6423 |              0.6057 |            0.0365 |                0.6077 |                   0.5575 |          -0.0503 |
| emoji          |       9 |   2.2 |      0.6667 |              0.6158 |            0.0509 |                0.6794 |                   0.546  |          -0.1333 |
| superlative    |      31 |   7.7 |      0.7419 |              0.6065 |            0.1355 |                0.5172 |                   0.5058 |          -0.0114 |
| sarcasm_marker |       5 |   1.2 |      0.8    |              0.6146 |            0.1854 |                1      |                   0.7778 |          -0.2222 |
| emoticon       |       1 |   0.2 |      1      |              0.616  |            0.384  |                1      |                   1      |           0      |

### `bilstm-attention` vs `naive_bayes`

**Per-class agreement** — agreement rate split by gold label, showing which sentiment class the two models disagree on most.

| class    |   n |   agreement |
|:---------|----:|------------:|
| negative | 114 |      0.6667 |
| neutral  | 162 |      0.5617 |
| positive | 126 |      0.5079 |

**Prediction cross-tabulation** — rows are `bilstm-attention` predictions, columns are `naive_bayes` predictions. Off-diagonal cells are disagreements.

| bilstm-attention \ naive_bayes   |   negative |   neutral |   positive |
|:---------------------------------|-----------:|----------:|-----------:|
| negative                         |        101 |        43 |         32 |
| neutral                          |         33 |        67 |         37 |
| positive                         |         10 |        16 |         63 |

**Linguistic feature divergence** — for each surface feature, how much it shifts agreement between the two models (Δagreement) and each model's macro-F1 on tweets containing that feature. Sorted by Δagreement ascending so the most divergence-inducing features appear first.

| feature        |   count |   pct |   agreement |   agreement_without |   delta_agreement |   f1_bilstm-attention |   f1_naive_bayes |   f1_delta (b-a) |
|:---------------|--------:|------:|------------:|--------------------:|------------------:|----------------------:|-----------------:|-----------------:|
| emoticon       |       1 |   0.2 |      0      |              0.5761 |           -0.5761 |                1      |           0      |          -1      |
| question       |      24 |   6   |      0.4167 |              0.5847 |           -0.168  |                0.7146 |           0.4292 |          -0.2854 |
| exclamation    |      61 |  15.2 |      0.5082 |              0.5865 |           -0.0783 |                0.4467 |           0.3651 |          -0.0816 |
| short_text     |     123 |  30.6 |      0.561  |              0.5806 |           -0.0197 |                0.6077 |           0.516  |          -0.0918 |
| emoji          |       9 |   2.2 |      0.5556 |              0.5751 |           -0.0195 |                0.6794 |           0.5683 |          -0.1111 |
| negation       |      74 |  18.4 |      0.6351 |              0.561  |            0.0742 |                0.5094 |           0.3796 |          -0.1298 |
| superlative    |      31 |   7.7 |      0.6774 |              0.566  |            0.1114 |                0.5172 |           0.4565 |          -0.0608 |
| sarcasm_marker |       5 |   1.2 |      0.8    |              0.5718 |            0.2282 |                1      |           0.7778 |          -0.2222 |
| allcaps        |      18 |   4.5 |      0.8889 |              0.5599 |            0.329  |                0.5556 |           0.5527 |          -0.0029 |

### `bilstm-attention` vs `weighted-naive-bayes-lr`

**Per-class agreement** — agreement rate split by gold label, showing which sentiment class the two models disagree on most.

| class    |   n |   agreement |
|:---------|----:|------------:|
| negative | 114 |      0.614  |
| neutral  | 162 |      0.642  |
| positive | 126 |      0.6429 |

**Prediction cross-tabulation** — rows are `bilstm-attention` predictions, columns are `weighted-naive-bayes-lr` predictions. Off-diagonal cells are disagreements.

| bilstm-attention \ weighted-naive-bayes-lr   |   negative |   neutral |   positive |
|:---------------------------------------------|-----------:|----------:|-----------:|
| negative                                     |         74 |        89 |         13 |
| neutral                                      |          4 |       127 |          6 |
| positive                                     |          4 |        31 |         54 |

**Linguistic feature divergence** — for each surface feature, how much it shifts agreement between the two models (Δagreement) and each model's macro-F1 on tweets containing that feature. Sorted by Δagreement ascending so the most divergence-inducing features appear first.

| feature        |   count |   pct |   agreement |   agreement_without |   delta_agreement |   f1_bilstm-attention |   f1_weighted-naive-bayes-lr |   f1_delta (b-a) |
|:---------------|--------:|------:|------------:|--------------------:|------------------:|----------------------:|-----------------------------:|-----------------:|
| question       |      24 |   6   |      0.5417 |              0.6402 |           -0.0985 |                0.7146 |                       0.4387 |          -0.276  |
| negation       |      74 |  18.4 |      0.5541 |              0.6524 |           -0.0984 |                0.5094 |                       0.4484 |          -0.061  |
| allcaps        |      18 |   4.5 |      0.5556 |              0.638  |           -0.0825 |                0.5556 |                       0.3905 |          -0.1651 |
| sarcasm_marker |       5 |   1.2 |      0.6    |              0.6348 |           -0.0348 |                1      |                       0.6111 |          -0.3889 |
| exclamation    |      61 |  15.2 |      0.6066 |              0.6393 |           -0.0327 |                0.4467 |                       0.3821 |          -0.0647 |
| emoji          |       9 |   2.2 |      0.6667 |              0.6336 |            0.0331 |                0.6794 |                       0.4444 |          -0.2349 |
| superlative    |      31 |   7.7 |      0.7419 |              0.6253 |            0.1166 |                0.5172 |                       0.6306 |           0.1134 |
| short_text     |     123 |  30.6 |      0.7398 |              0.5878 |            0.152  |                0.6077 |                       0.6218 |           0.014  |
| emoticon       |       1 |   0.2 |      1      |              0.6334 |            0.3666 |                1      |                       1      |           0      |

### `bilstm` vs `cardiff_roberta`

**Per-class agreement** — agreement rate split by gold label, showing which sentiment class the two models disagree on most.

| class    |   n |   agreement |
|:---------|----:|------------:|
| negative | 114 |      0.5877 |
| neutral  | 162 |      0.7346 |
| positive | 126 |      0.5873 |

**Prediction cross-tabulation** — rows are `bilstm` predictions, columns are `cardiff_roberta` predictions. Off-diagonal cells are disagreements.

| bilstm \ cardiff_roberta   |   negative |   neutral |   positive |
|:---------------------------|-----------:|----------:|-----------:|
| negative                   |         77 |        41 |         13 |
| neutral                    |         43 |       132 |         30 |
| positive                   |          9 |         6 |         51 |

**Linguistic feature divergence** — for each surface feature, how much it shifts agreement between the two models (Δagreement) and each model's macro-F1 on tweets containing that feature. Sorted by Δagreement ascending so the most divergence-inducing features appear first.

| feature        |   count |   pct |   agreement |   agreement_without |   delta_agreement |   f1_bilstm |   f1_cardiff_roberta |   f1_delta (b-a) |
|:---------------|--------:|------:|------------:|--------------------:|------------------:|------------:|---------------------:|-----------------:|
| emoticon       |       1 |   0.2 |      0      |              0.6484 |           -0.6484 |      1      |               0      |          -1      |
| exclamation    |      61 |  15.2 |      0.4426 |              0.6833 |           -0.2407 |      0.3582 |               0.7545 |           0.3963 |
| emoji          |       9 |   2.2 |      0.5556 |              0.6489 |           -0.0933 |      0.6794 |               0.6627 |          -0.0167 |
| sarcasm_marker |       5 |   1.2 |      0.6    |              0.6474 |           -0.0474 |      1      |               0.6111 |          -0.3889 |
| allcaps        |      18 |   4.5 |      0.6111 |              0.6484 |           -0.0373 |      0.475  |               0.6898 |           0.2148 |
| negation       |      74 |  18.4 |      0.6757 |              0.6402 |            0.0354 |      0.5533 |               0.6767 |           0.1234 |
| question       |      24 |   6   |      0.7083 |              0.6429 |            0.0655 |      0.5537 |               0.8344 |           0.2807 |
| short_text     |     123 |  30.6 |      0.6992 |              0.6237 |            0.0755 |      0.5711 |               0.818  |           0.2469 |
| superlative    |      31 |   7.7 |      0.8065 |              0.6334 |            0.173  |      0.606  |               0.694  |           0.088  |

### `bilstm` vs `finetuned_roberta`

**Per-class agreement** — agreement rate split by gold label, showing which sentiment class the two models disagree on most.

| class    |   n |   agreement |
|:---------|----:|------------:|
| negative | 114 |      0.5965 |
| neutral  | 162 |      0.7346 |
| positive | 126 |      0.6825 |

**Prediction cross-tabulation** — rows are `bilstm` predictions, columns are `finetuned_roberta` predictions. Off-diagonal cells are disagreements.

| bilstm \ finetuned_roberta   |   negative |   neutral |   positive |
|:-----------------------------|-----------:|----------:|-----------:|
| negative                     |         61 |        60 |         10 |
| neutral                      |         21 |       164 |         20 |
| positive                     |          5 |        13 |         48 |

**Linguistic feature divergence** — for each surface feature, how much it shifts agreement between the two models (Δagreement) and each model's macro-F1 on tweets containing that feature. Sorted by Δagreement ascending so the most divergence-inducing features appear first.

| feature        |   count |   pct |   agreement |   agreement_without |   delta_agreement |   f1_bilstm |   f1_finetuned_roberta |   f1_delta (b-a) |
|:---------------|--------:|------:|------------:|--------------------:|------------------:|------------:|-----------------------:|-----------------:|
| emoticon       |       1 |   0.2 |      0      |              0.6808 |           -0.6808 |      1      |                 0      |          -1      |
| emoji          |       9 |   2.2 |      0.5556 |              0.6819 |           -0.1264 |      0.6794 |                 0.3556 |          -0.3238 |
| exclamation    |      61 |  15.2 |      0.5738 |              0.6979 |           -0.1242 |      0.3582 |                 0.5829 |           0.2247 |
| sarcasm_marker |       5 |   1.2 |      0.6    |              0.6801 |           -0.0801 |      1      |                 0.6111 |          -0.3889 |
| negation       |      74 |  18.4 |      0.6216 |              0.6921 |           -0.0705 |      0.5533 |                 0.5961 |           0.0428 |
| allcaps        |      18 |   4.5 |      0.6667 |              0.6797 |           -0.013  |      0.475  |                 0.5661 |           0.0911 |
| superlative    |      31 |   7.7 |      0.6774 |              0.6792 |           -0.0018 |      0.606  |                 0.6186 |           0.0126 |
| question       |      24 |   6   |      0.75   |              0.6746 |            0.0754 |      0.5537 |                 0.7605 |           0.2068 |
| short_text     |     123 |  30.6 |      0.7724 |              0.638  |            0.1344 |      0.5711 |                 0.7173 |           0.1462 |

### `bilstm` vs `gemma_4-31B`

**Per-class agreement** — agreement rate split by gold label, showing which sentiment class the two models disagree on most.

| class    |   n |   agreement |
|:---------|----:|------------:|
| negative | 114 |      0.5877 |
| neutral  | 162 |      0.6049 |
| positive | 126 |      0.4524 |

**Prediction cross-tabulation** — rows are `bilstm` predictions, columns are `gemma_4-31B` predictions. Off-diagonal cells are disagreements.

| bilstm \ gemma_4-31B   |   negative |   neutral |   positive |
|:-----------------------|-----------:|----------:|-----------:|
| negative               |         70 |        34 |         27 |
| neutral                |         49 |       106 |         50 |
| positive               |          9 |        11 |         46 |

**Linguistic feature divergence** — for each surface feature, how much it shifts agreement between the two models (Δagreement) and each model's macro-F1 on tweets containing that feature. Sorted by Δagreement ascending so the most divergence-inducing features appear first.

| feature        |   count |   pct |   agreement |   agreement_without |   delta_agreement |   f1_bilstm |   f1_gemma_4-31B |   f1_delta (b-a) |
|:---------------|--------:|------:|------------:|--------------------:|------------------:|------------:|-----------------:|-----------------:|
| exclamation    |      61 |  15.2 |      0.3279 |              0.5924 |           -0.2645 |      0.3582 |           0.7968 |           0.4386 |
| negation       |      74 |  18.4 |      0.5541 |              0.5518 |            0.0022 |      0.5533 |           0.7959 |           0.2426 |
| emoji          |       9 |   2.2 |      0.5556 |              0.5522 |            0.0034 |      0.6794 |           0.65   |          -0.0294 |
| sarcasm_marker |       5 |   1.2 |      0.6    |              0.5516 |            0.0484 |      1      |           0.6111 |          -0.3889 |
| allcaps        |      18 |   4.5 |      0.6111 |              0.5495 |            0.0616 |      0.475  |           0.7103 |           0.2353 |
| question       |      24 |   6   |      0.6667 |              0.545  |            0.1217 |      0.5537 |           0.8719 |           0.3182 |
| superlative    |      31 |   7.7 |      0.6774 |              0.5418 |            0.1356 |      0.606  |           0.7774 |           0.1714 |
| short_text     |     123 |  30.6 |      0.6504 |              0.509  |            0.1414 |      0.5711 |           0.9206 |           0.3495 |
| emoticon       |       1 |   0.2 |      1      |              0.5511 |            0.4489 |      1      |           1      |           0      |

### `bilstm` vs `logistic_regression`

**Per-class agreement** — agreement rate split by gold label, showing which sentiment class the two models disagree on most.

| class    |   n |   agreement |
|:---------|----:|------------:|
| negative | 114 |      0.5789 |
| neutral  | 162 |      0.6667 |
| positive | 126 |      0.6667 |

**Prediction cross-tabulation** — rows are `bilstm` predictions, columns are `logistic_regression` predictions. Off-diagonal cells are disagreements.

| bilstm \ logistic_regression   |   negative |   neutral |   positive |
|:-------------------------------|-----------:|----------:|-----------:|
| negative                       |         70 |        45 |         16 |
| neutral                        |         42 |       135 |         28 |
| positive                       |          4 |         9 |         53 |

**Linguistic feature divergence** — for each surface feature, how much it shifts agreement between the two models (Δagreement) and each model's macro-F1 on tweets containing that feature. Sorted by Δagreement ascending so the most divergence-inducing features appear first.

| feature        |   count |   pct |   agreement |   agreement_without |   delta_agreement |   f1_bilstm |   f1_logistic_regression |   f1_delta (b-a) |
|:---------------|--------:|------:|------------:|--------------------:|------------------:|------------:|-------------------------:|-----------------:|
| question       |      24 |   6   |      0.4167 |              0.6561 |           -0.2394 |      0.5537 |                   0.467  |          -0.0867 |
| negation       |      74 |  18.4 |      0.5541 |              0.6616 |           -0.1075 |      0.5533 |                   0.4709 |          -0.0824 |
| allcaps        |      18 |   4.5 |      0.5556 |              0.6458 |           -0.0903 |      0.475  |                   0.5427 |           0.0677 |
| emoji          |       9 |   2.2 |      0.5556 |              0.6438 |           -0.0882 |      0.6794 |                   0.546  |          -0.1333 |
| exclamation    |      61 |  15.2 |      0.6557 |              0.6393 |            0.0164 |      0.3582 |                   0.4251 |           0.0669 |
| superlative    |      31 |   7.7 |      0.7097 |              0.6361 |            0.0736 |      0.606  |                   0.5058 |          -0.1002 |
| short_text     |     123 |  30.6 |      0.7154 |              0.6093 |            0.1061 |      0.5711 |                   0.5575 |          -0.0136 |
| sarcasm_marker |       5 |   1.2 |      0.8    |              0.6398 |            0.1602 |      1      |                   0.7778 |          -0.2222 |
| emoticon       |       1 |   0.2 |      1      |              0.6409 |            0.3591 |      1      |                   1      |           0      |

### `bilstm` vs `naive_bayes`

**Per-class agreement** — agreement rate split by gold label, showing which sentiment class the two models disagree on most.

| class    |   n |   agreement |
|:---------|----:|------------:|
| negative | 114 |      0.6404 |
| neutral  | 162 |      0.5494 |
| positive | 126 |      0.5397 |

**Prediction cross-tabulation** — rows are `bilstm` predictions, columns are `naive_bayes` predictions. Off-diagonal cells are disagreements.

| bilstm \ naive_bayes   |   negative |   neutral |   positive |
|:-----------------------|-----------:|----------:|-----------:|
| negative               |         84 |        23 |         24 |
| neutral                |         51 |        96 |         58 |
| positive               |          9 |         7 |         50 |

**Linguistic feature divergence** — for each surface feature, how much it shifts agreement between the two models (Δagreement) and each model's macro-F1 on tweets containing that feature. Sorted by Δagreement ascending so the most divergence-inducing features appear first.

| feature        |   count |   pct |   agreement |   agreement_without |   delta_agreement |   f1_bilstm |   f1_naive_bayes |   f1_delta (b-a) |
|:---------------|--------:|------:|------------:|--------------------:|------------------:|------------:|-----------------:|-----------------:|
| emoticon       |       1 |   0.2 |      0      |              0.5736 |           -0.5736 |      1      |           0      |          -1      |
| short_text     |     123 |  30.6 |      0.5285 |              0.5914 |           -0.0629 |      0.5711 |           0.516  |          -0.0551 |
| question       |      24 |   6   |      0.5417 |              0.5741 |           -0.0324 |      0.5537 |           0.4292 |          -0.1245 |
| negation       |      74 |  18.4 |      0.5676 |              0.5732 |           -0.0056 |      0.5533 |           0.3796 |          -0.1738 |
| exclamation    |      61 |  15.2 |      0.6066 |              0.566  |            0.0406 |      0.3582 |           0.3651 |           0.0069 |
| emoji          |       9 |   2.2 |      0.6667 |              0.57   |            0.0967 |      0.6794 |           0.5683 |          -0.1111 |
| superlative    |      31 |   7.7 |      0.6774 |              0.5633 |            0.1141 |      0.606  |           0.4565 |          -0.1496 |
| sarcasm_marker |       5 |   1.2 |      0.8    |              0.5693 |            0.2307 |      1      |           0.7778 |          -0.2222 |
| allcaps        |      18 |   4.5 |      0.8333 |              0.5599 |            0.2734 |      0.475  |           0.5527 |           0.0777 |

### `bilstm` vs `weighted-naive-bayes-lr`

**Per-class agreement** — agreement rate split by gold label, showing which sentiment class the two models disagree on most.

| class    |   n |   agreement |
|:---------|----:|------------:|
| negative | 114 |      0.6667 |
| neutral  | 162 |      0.7407 |
| positive | 126 |      0.7698 |

**Prediction cross-tabulation** — rows are `bilstm` predictions, columns are `weighted-naive-bayes-lr` predictions. Off-diagonal cells are disagreements.

| bilstm \ weighted-naive-bayes-lr   |   negative |   neutral |   positive |
|:-----------------------------------|-----------:|----------:|-----------:|
| negative                           |         65 |        58 |          8 |
| neutral                            |         14 |       177 |         14 |
| positive                           |          3 |        12 |         51 |

**Linguistic feature divergence** — for each surface feature, how much it shifts agreement between the two models (Δagreement) and each model's macro-F1 on tweets containing that feature. Sorted by Δagreement ascending so the most divergence-inducing features appear first.

| feature        |   count |   pct |   agreement |   agreement_without |   delta_agreement |   f1_bilstm |   f1_weighted-naive-bayes-lr |   f1_delta (b-a) |
|:---------------|--------:|------:|------------:|--------------------:|------------------:|------------:|-----------------------------:|-----------------:|
| allcaps        |      18 |   4.5 |      0.5556 |              0.737  |           -0.1814 |      0.475  |                       0.3905 |          -0.0845 |
| emoji          |       9 |   2.2 |      0.5556 |              0.7328 |           -0.1773 |      0.6794 |                       0.4444 |          -0.2349 |
| sarcasm_marker |       5 |   1.2 |      0.6    |              0.7305 |           -0.1305 |      1      |                       0.6111 |          -0.3889 |
| negation       |      74 |  18.4 |      0.6351 |              0.75   |           -0.1149 |      0.5533 |                       0.4484 |          -0.1049 |
| question       |      24 |   6   |      0.625  |              0.7354 |           -0.1104 |      0.5537 |                       0.4387 |          -0.115  |
| superlative    |      31 |   7.7 |      0.7419 |              0.7278 |            0.0142 |      0.606  |                       0.6306 |           0.0246 |
| exclamation    |      61 |  15.2 |      0.7541 |              0.7243 |            0.0298 |      0.3582 |                       0.3821 |           0.0239 |
| short_text     |     123 |  30.6 |      0.8049 |              0.6953 |            0.1095 |      0.5711 |                       0.6218 |           0.0507 |
| emoticon       |       1 |   0.2 |      1      |              0.7282 |            0.2718 |      1      |                       1      |           0      |

### `cardiff_roberta` vs `finetuned_roberta`

**Per-class agreement** — agreement rate split by gold label, showing which sentiment class the two models disagree on most.

| class    |   n |   agreement |
|:---------|----:|------------:|
| negative | 114 |      0.6842 |
| neutral  | 162 |      0.8519 |
| positive | 126 |      0.7698 |

**Prediction cross-tabulation** — rows are `cardiff_roberta` predictions, columns are `finetuned_roberta` predictions. Off-diagonal cells are disagreements.

| cardiff_roberta \ finetuned_roberta   |   negative |   neutral |   positive |
|:--------------------------------------|-----------:|----------:|-----------:|
| negative                              |         80 |        47 |          2 |
| neutral                               |          6 |       165 |          8 |
| positive                              |          1 |        25 |         68 |

**Linguistic feature divergence** — for each surface feature, how much it shifts agreement between the two models (Δagreement) and each model's macro-F1 on tweets containing that feature. Sorted by Δagreement ascending so the most divergence-inducing features appear first.

| feature        |   count |   pct |   agreement |   agreement_without |   delta_agreement |   f1_cardiff_roberta |   f1_finetuned_roberta |   f1_delta (b-a) |
|:---------------|--------:|------:|------------:|--------------------:|------------------:|---------------------:|-----------------------:|-----------------:|
| emoticon       |       1 |   0.2 |      0      |              0.7805 |           -0.7805 |               0      |                 0      |           0      |
| emoji          |       9 |   2.2 |      0.6667 |              0.7812 |           -0.1145 |               0.6627 |                 0.3556 |          -0.3071 |
| exclamation    |      61 |  15.2 |      0.6885 |              0.7947 |           -0.1062 |               0.7545 |                 0.5829 |          -0.1716 |
| negation       |      74 |  18.4 |      0.7027 |              0.7957 |           -0.093  |               0.6767 |                 0.5961 |          -0.0806 |
| allcaps        |      18 |   4.5 |      0.7222 |              0.7812 |           -0.059  |               0.6898 |                 0.5661 |          -0.1237 |
| superlative    |      31 |   7.7 |      0.7742 |              0.779  |           -0.0048 |               0.694  |                 0.6186 |          -0.0754 |
| short_text     |     123 |  30.6 |      0.8293 |              0.7563 |            0.073  |               0.818  |                 0.7173 |          -0.1007 |
| question       |      24 |   6   |      0.875  |              0.7725 |            0.1025 |               0.8344 |                 0.7605 |          -0.0739 |
| sarcasm_marker |       5 |   1.2 |      1      |              0.7758 |            0.2242 |               0.6111 |                 0.6111 |           0      |

### `cardiff_roberta` vs `gemma_4-31B`

**Per-class agreement** — agreement rate split by gold label, showing which sentiment class the two models disagree on most.

| class    |   n |   agreement |
|:---------|----:|------------:|
| negative | 114 |      0.8333 |
| neutral  | 162 |      0.7407 |
| positive | 126 |      0.6905 |

**Prediction cross-tabulation** — rows are `cardiff_roberta` predictions, columns are `gemma_4-31B` predictions. Off-diagonal cells are disagreements.

| cardiff_roberta \ gemma_4-31B   |   negative |   neutral |   positive |
|:--------------------------------|-----------:|----------:|-----------:|
| negative                        |        104 |        17 |          8 |
| neutral                         |         22 |       120 |         37 |
| positive                        |          2 |        14 |         78 |

**Linguistic feature divergence** — for each surface feature, how much it shifts agreement between the two models (Δagreement) and each model's macro-F1 on tweets containing that feature. Sorted by Δagreement ascending so the most divergence-inducing features appear first.

| feature        |   count |   pct |   agreement |   agreement_without |   delta_agreement |   f1_cardiff_roberta |   f1_gemma_4-31B |   f1_delta (b-a) |
|:---------------|--------:|------:|------------:|--------------------:|------------------:|---------------------:|-----------------:|-----------------:|
| emoticon       |       1 |   0.2 |      0      |              0.7531 |           -0.7531 |               0      |           1      |           1      |
| negation       |      74 |  18.4 |      0.6622 |              0.7713 |           -0.1092 |               0.6767 |           0.7959 |           0.1192 |
| allcaps        |      18 |   4.5 |      0.7222 |              0.7526 |           -0.0304 |               0.6898 |           0.7103 |           0.0206 |
| exclamation    |      61 |  15.2 |      0.7705 |              0.7478 |            0.0227 |               0.7545 |           0.7968 |           0.0423 |
| emoji          |       9 |   2.2 |      0.7778 |              0.7506 |            0.0271 |               0.6627 |           0.65   |          -0.0127 |
| sarcasm_marker |       5 |   1.2 |      0.8    |              0.7506 |            0.0494 |               0.6111 |           0.6111 |           0      |
| superlative    |      31 |   7.7 |      0.8065 |              0.7466 |            0.0598 |               0.694  |           0.7774 |           0.0834 |
| short_text     |     123 |  30.6 |      0.8374 |              0.7133 |            0.1241 |               0.818  |           0.9206 |           0.1025 |
| question       |      24 |   6   |      0.875  |              0.7434 |            0.1316 |               0.8344 |           0.8719 |           0.0374 |

### `cardiff_roberta` vs `logistic_regression`

**Per-class agreement** — agreement rate split by gold label, showing which sentiment class the two models disagree on most.

| class    |   n |   agreement |
|:---------|----:|------------:|
| negative | 114 |      0.4737 |
| neutral  | 162 |      0.6049 |
| positive | 126 |      0.619  |

**Prediction cross-tabulation** — rows are `cardiff_roberta` predictions, columns are `logistic_regression` predictions. Off-diagonal cells are disagreements.

| cardiff_roberta \ logistic_regression   |   negative |   neutral |   positive |
|:----------------------------------------|-----------:|----------:|-----------:|
| negative                                |         62 |        51 |         16 |
| neutral                                 |         42 |       112 |         25 |
| positive                                |         12 |        26 |         56 |

**Linguistic feature divergence** — for each surface feature, how much it shifts agreement between the two models (Δagreement) and each model's macro-F1 on tweets containing that feature. Sorted by Δagreement ascending so the most divergence-inducing features appear first.

| feature        |   count |   pct |   agreement |   agreement_without |   delta_agreement |   f1_cardiff_roberta |   f1_logistic_regression |   f1_delta (b-a) |
|:---------------|--------:|------:|------------:|--------------------:|------------------:|---------------------:|-------------------------:|-----------------:|
| emoticon       |       1 |   0.2 |      0      |              0.5736 |           -0.5736 |               0      |                   1      |           1      |
| emoji          |       9 |   2.2 |      0.3333 |              0.5776 |           -0.2443 |               0.6627 |                   0.546  |          -0.1167 |
| question       |      24 |   6   |      0.4583 |              0.5794 |           -0.121  |               0.8344 |                   0.467  |          -0.3675 |
| exclamation    |      61 |  15.2 |      0.5246 |              0.5806 |           -0.0561 |               0.7545 |                   0.4251 |          -0.3294 |
| negation       |      74 |  18.4 |      0.527  |              0.5823 |           -0.0553 |               0.6767 |                   0.4709 |          -0.2058 |
| short_text     |     123 |  30.6 |      0.6179 |              0.552  |            0.0659 |               0.818  |                   0.5575 |          -0.2606 |
| superlative    |      31 |   7.7 |      0.6452 |              0.566  |            0.0791 |               0.694  |                   0.5058 |          -0.1882 |
| allcaps        |      18 |   4.5 |      0.7222 |              0.5651 |            0.1571 |               0.6898 |                   0.5427 |          -0.147  |
| sarcasm_marker |       5 |   1.2 |      0.8    |              0.5693 |            0.2307 |               0.6111 |                   0.7778 |           0.1667 |

### `cardiff_roberta` vs `naive_bayes`

**Per-class agreement** — agreement rate split by gold label, showing which sentiment class the two models disagree on most.

| class    |   n |   agreement |
|:---------|----:|------------:|
| negative | 114 |      0.5526 |
| neutral  | 162 |      0.4691 |
| positive | 126 |      0.5873 |

**Prediction cross-tabulation** — rows are `cardiff_roberta` predictions, columns are `naive_bayes` predictions. Off-diagonal cells are disagreements.

| cardiff_roberta \ naive_bayes   |   negative |   neutral |   positive |
|:--------------------------------|-----------:|----------:|-----------:|
| negative                        |         74 |        39 |         16 |
| neutral                         |         54 |        74 |         51 |
| positive                        |         16 |        13 |         65 |

**Linguistic feature divergence** — for each surface feature, how much it shifts agreement between the two models (Δagreement) and each model's macro-F1 on tweets containing that feature. Sorted by Δagreement ascending so the most divergence-inducing features appear first.

| feature        |   count |   pct |   agreement |   agreement_without |   delta_agreement |   f1_cardiff_roberta |   f1_naive_bayes |   f1_delta (b-a) |
|:---------------|--------:|------:|------------:|--------------------:|------------------:|---------------------:|-----------------:|-----------------:|
| question       |      24 |   6   |      0.375  |              0.5397 |           -0.1647 |               0.8344 |           0.4292 |          -0.4052 |
| exclamation    |      61 |  15.2 |      0.459  |              0.5425 |           -0.0835 |               0.7545 |           0.3651 |          -0.3894 |
| negation       |      74 |  18.4 |      0.4865 |              0.5396 |           -0.0531 |               0.6767 |           0.3796 |          -0.2971 |
| short_text     |     123 |  30.6 |      0.5285 |              0.5305 |           -0.002  |               0.818  |           0.516  |          -0.3021 |
| emoji          |       9 |   2.2 |      0.5556 |              0.5293 |            0.0263 |               0.6627 |           0.5683 |          -0.0944 |
| allcaps        |      18 |   4.5 |      0.6667 |              0.5234 |            0.1432 |               0.6898 |           0.5527 |          -0.1371 |
| superlative    |      31 |   7.7 |      0.7419 |              0.5121 |            0.2298 |               0.694  |           0.4565 |          -0.2376 |
| sarcasm_marker |       5 |   1.2 |      0.8    |              0.5264 |            0.2736 |               0.6111 |           0.7778 |           0.1667 |
| emoticon       |       1 |   0.2 |      1      |              0.5287 |            0.4713 |               0      |           0      |           0      |

### `cardiff_roberta` vs `weighted-naive-bayes-lr`

**Per-class agreement** — agreement rate split by gold label, showing which sentiment class the two models disagree on most.

| class    |   n |   agreement |
|:---------|----:|------------:|
| negative | 114 |      0.5    |
| neutral  | 162 |      0.7654 |
| positive | 126 |      0.6667 |

**Prediction cross-tabulation** — rows are `cardiff_roberta` predictions, columns are `weighted-naive-bayes-lr` predictions. Off-diagonal cells are disagreements.

| cardiff_roberta \ weighted-naive-bayes-lr   |   negative |   neutral |   positive |
|:--------------------------------------------|-----------:|----------:|-----------:|
| negative                                    |         57 |        66 |          6 |
| neutral                                     |         18 |       151 |         10 |
| positive                                    |          7 |        30 |         57 |

**Linguistic feature divergence** — for each surface feature, how much it shifts agreement between the two models (Δagreement) and each model's macro-F1 on tweets containing that feature. Sorted by Δagreement ascending so the most divergence-inducing features appear first.

| feature        |   count |   pct |   agreement |   agreement_without |   delta_agreement |   f1_cardiff_roberta |   f1_weighted-naive-bayes-lr |   f1_delta (b-a) |
|:---------------|--------:|------:|------------:|--------------------:|------------------:|---------------------:|-----------------------------:|-----------------:|
| emoticon       |       1 |   0.2 |      0      |              0.6608 |           -0.6608 |               0      |                       1      |           1      |
| negation       |      74 |  18.4 |      0.5541 |              0.6829 |           -0.1289 |               0.6767 |                       0.4484 |          -0.2283 |
| exclamation    |      61 |  15.2 |      0.5574 |              0.6774 |           -0.12   |               0.7545 |                       0.3821 |          -0.3725 |
| emoji          |       9 |   2.2 |      0.5556 |              0.6616 |           -0.106  |               0.6627 |                       0.4444 |          -0.2183 |
| question       |      24 |   6   |      0.625  |              0.6614 |           -0.0364 |               0.8344 |                       0.4387 |          -0.3958 |
| allcaps        |      18 |   4.5 |      0.6667 |              0.6589 |            0.0078 |               0.6898 |                       0.3905 |          -0.2993 |
| short_text     |     123 |  30.6 |      0.6992 |              0.6416 |            0.0576 |               0.818  |                       0.6218 |          -0.1963 |
| superlative    |      31 |   7.7 |      0.7742 |              0.6496 |            0.1246 |               0.694  |                       0.6306 |          -0.0634 |
| sarcasm_marker |       5 |   1.2 |      1      |              0.6549 |            0.3451 |               0.6111 |                       0.6111 |           0      |

### `finetuned_roberta` vs `gemma_4-31B`

**Per-class agreement** — agreement rate split by gold label, showing which sentiment class the two models disagree on most.

| class    |   n |   agreement |
|:---------|----:|------------:|
| negative | 114 |      0.5965 |
| neutral  | 162 |      0.7716 |
| positive | 126 |      0.5794 |

**Prediction cross-tabulation** — rows are `finetuned_roberta` predictions, columns are `gemma_4-31B` predictions. Off-diagonal cells are disagreements.

| finetuned_roberta \ gemma_4-31B   |   negative |   neutral |   positive |
|:----------------------------------|-----------:|----------:|-----------:|
| negative                          |         73 |         9 |          5 |
| neutral                           |         50 |       131 |         56 |
| positive                          |          5 |        11 |         62 |

**Linguistic feature divergence** — for each surface feature, how much it shifts agreement between the two models (Δagreement) and each model's macro-F1 on tweets containing that feature. Sorted by Δagreement ascending so the most divergence-inducing features appear first.

| feature        |   count |   pct |   agreement |   agreement_without |   delta_agreement |   f1_finetuned_roberta |   f1_gemma_4-31B |   f1_delta (b-a) |
|:---------------|--------:|------:|------------:|--------------------:|------------------:|-----------------------:|-----------------:|-----------------:|
| emoticon       |       1 |   0.2 |      0      |              0.6633 |           -0.6633 |                 0      |           1      |           1      |
| emoji          |       9 |   2.2 |      0.4444 |              0.6667 |           -0.2222 |                 0.3556 |           0.65   |           0.2944 |
| exclamation    |      61 |  15.2 |      0.5574 |              0.6804 |           -0.123  |                 0.5829 |           0.7968 |           0.2139 |
| negation       |      74 |  18.4 |      0.5946 |              0.6768 |           -0.0822 |                 0.5961 |           0.7959 |           0.1998 |
| superlative    |      31 |   7.7 |      0.6774 |              0.6604 |            0.017  |                 0.6186 |           0.7774 |           0.1587 |
| allcaps        |      18 |   4.5 |      0.7778 |              0.6562 |            0.1215 |                 0.5661 |           0.7103 |           0.1442 |
| sarcasm_marker |       5 |   1.2 |      0.8    |              0.6599 |            0.1401 |                 0.6111 |           0.6111 |           0      |
| short_text     |     123 |  30.6 |      0.7724 |              0.6129 |            0.1595 |                 0.7173 |           0.9206 |           0.2032 |
| question       |      24 |   6   |      0.8333 |              0.6508 |            0.1825 |                 0.7605 |           0.8719 |           0.1114 |

### `finetuned_roberta` vs `logistic_regression`

**Per-class agreement** — agreement rate split by gold label, showing which sentiment class the two models disagree on most.

| class    |   n |   agreement |
|:---------|----:|------------:|
| negative | 114 |      0.5    |
| neutral  | 162 |      0.642  |
| positive | 126 |      0.5952 |

**Prediction cross-tabulation** — rows are `finetuned_roberta` predictions, columns are `logistic_regression` predictions. Off-diagonal cells are disagreements.

| finetuned_roberta \ logistic_regression   |   negative |   neutral |   positive |
|:------------------------------------------|-----------:|----------:|-----------:|
| negative                                  |         50 |        30 |          7 |
| neutral                                   |         57 |       138 |         42 |
| positive                                  |          9 |        21 |         48 |

**Linguistic feature divergence** — for each surface feature, how much it shifts agreement between the two models (Δagreement) and each model's macro-F1 on tweets containing that feature. Sorted by Δagreement ascending so the most divergence-inducing features appear first.

| feature        |   count |   pct |   agreement |   agreement_without |   delta_agreement |   f1_finetuned_roberta |   f1_logistic_regression |   f1_delta (b-a) |
|:---------------|--------:|------:|------------:|--------------------:|------------------:|-----------------------:|-------------------------:|-----------------:|
| emoticon       |       1 |   0.2 |      0      |              0.5885 |           -0.5885 |                 0      |                   1      |           1      |
| emoji          |       9 |   2.2 |      0.3333 |              0.5929 |           -0.2595 |                 0.3556 |                   0.546  |           0.1905 |
| question       |      24 |   6   |      0.4583 |              0.5952 |           -0.1369 |                 0.7605 |                   0.467  |          -0.2935 |
| superlative    |      31 |   7.7 |      0.4839 |              0.5957 |           -0.1118 |                 0.6186 |                   0.5058 |          -0.1128 |
| negation       |      74 |  18.4 |      0.5135 |              0.6037 |           -0.0901 |                 0.5961 |                   0.4709 |          -0.1252 |
| exclamation    |      61 |  15.2 |      0.5246 |              0.5982 |           -0.0737 |                 0.5829 |                   0.4251 |          -0.1578 |
| allcaps        |      18 |   4.5 |      0.6667 |              0.5833 |            0.0833 |                 0.5661 |                   0.5427 |          -0.0233 |
| short_text     |     123 |  30.6 |      0.6667 |              0.552  |            0.1147 |                 0.7173 |                   0.5575 |          -0.1599 |
| sarcasm_marker |       5 |   1.2 |      0.8    |              0.5844 |            0.2156 |                 0.6111 |                   0.7778 |           0.1667 |

### `finetuned_roberta` vs `naive_bayes`

**Per-class agreement** — agreement rate split by gold label, showing which sentiment class the two models disagree on most.

| class    |   n |   agreement |
|:---------|----:|------------:|
| negative | 114 |      0.5439 |
| neutral  | 162 |      0.463  |
| positive | 126 |      0.5476 |

**Prediction cross-tabulation** — rows are `finetuned_roberta` predictions, columns are `naive_bayes` predictions. Off-diagonal cells are disagreements.

| finetuned_roberta \ naive_bayes   |   negative |   neutral |   positive |
|:----------------------------------|-----------:|----------:|-----------:|
| negative                          |         56 |        22 |          9 |
| neutral                           |         76 |        94 |         67 |
| positive                          |         12 |        10 |         56 |

**Linguistic feature divergence** — for each surface feature, how much it shifts agreement between the two models (Δagreement) and each model's macro-F1 on tweets containing that feature. Sorted by Δagreement ascending so the most divergence-inducing features appear first.

| feature        |   count |   pct |   agreement |   agreement_without |   delta_agreement |   f1_finetuned_roberta |   f1_naive_bayes |   f1_delta (b-a) |
|:---------------|--------:|------:|------------:|--------------------:|------------------:|-----------------------:|-----------------:|-----------------:|
| emoticon       |       1 |   0.2 |      0      |              0.5137 |           -0.5137 |                 0      |           0      |           0      |
| question       |      24 |   6   |      0.4167 |              0.5185 |           -0.1019 |                 0.7605 |           0.4292 |          -0.3313 |
| negation       |      74 |  18.4 |      0.4324 |              0.5305 |           -0.0981 |                 0.5961 |           0.3796 |          -0.2165 |
| exclamation    |      61 |  15.2 |      0.459  |              0.522  |           -0.063  |                 0.5829 |           0.3651 |          -0.2178 |
| short_text     |     123 |  30.6 |      0.5122 |              0.5125 |           -0.0003 |                 0.7173 |           0.516  |          -0.2014 |
| allcaps        |      18 |   4.5 |      0.5556 |              0.5104 |            0.0451 |                 0.5661 |           0.5527 |          -0.0134 |
| emoji          |       9 |   2.2 |      0.6667 |              0.5089 |            0.1578 |                 0.3556 |           0.5683 |           0.2127 |
| superlative    |      31 |   7.7 |      0.6774 |              0.4987 |            0.1788 |                 0.6186 |           0.4565 |          -0.1622 |
| sarcasm_marker |       5 |   1.2 |      0.8    |              0.5088 |            0.2912 |                 0.6111 |           0.7778 |           0.1667 |

### `finetuned_roberta` vs `weighted-naive-bayes-lr`

**Per-class agreement** — agreement rate split by gold label, showing which sentiment class the two models disagree on most.

| class    |   n |   agreement |
|:---------|----:|------------:|
| negative | 114 |      0.5877 |
| neutral  | 162 |      0.7778 |
| positive | 126 |      0.6905 |

**Prediction cross-tabulation** — rows are `finetuned_roberta` predictions, columns are `weighted-naive-bayes-lr` predictions. Off-diagonal cells are disagreements.

| finetuned_roberta \ weighted-naive-bayes-lr   |   negative |   neutral |   positive |
|:----------------------------------------------|-----------:|----------:|-----------:|
| negative                                      |         47 |        38 |          2 |
| neutral                                       |         29 |       185 |         23 |
| positive                                      |          6 |        24 |         48 |

**Linguistic feature divergence** — for each surface feature, how much it shifts agreement between the two models (Δagreement) and each model's macro-F1 on tweets containing that feature. Sorted by Δagreement ascending so the most divergence-inducing features appear first.

| feature        |   count |   pct |   agreement |   agreement_without |   delta_agreement |   f1_finetuned_roberta |   f1_weighted-naive-bayes-lr |   f1_delta (b-a) |
|:---------------|--------:|------:|------------:|--------------------:|------------------:|-----------------------:|-----------------------------:|-----------------:|
| emoticon       |       1 |   0.2 |      0      |              0.6983 |           -0.6983 |                 0      |                       1      |           1      |
| exclamation    |      61 |  15.2 |      0.5738 |              0.7185 |           -0.1447 |                 0.5829 |                       0.3821 |          -0.2008 |
| emoji          |       9 |   2.2 |      0.5556 |              0.6997 |           -0.1442 |                 0.3556 |                       0.4444 |           0.0889 |
| negation       |      74 |  18.4 |      0.5811 |              0.7226 |           -0.1415 |                 0.5961 |                       0.4484 |          -0.1477 |
| question       |      24 |   6   |      0.625  |              0.7011 |           -0.0761 |                 0.7605 |                       0.4387 |          -0.3218 |
| superlative    |      31 |   7.7 |      0.6452 |              0.7008 |           -0.0556 |                 0.6186 |                       0.6306 |           0.012  |
| allcaps        |      18 |   4.5 |      0.6667 |              0.6979 |           -0.0312 |                 0.5661 |                       0.3905 |          -0.1756 |
| short_text     |     123 |  30.6 |      0.7642 |              0.6667 |            0.0976 |                 0.7173 |                       0.6218 |          -0.0956 |
| sarcasm_marker |       5 |   1.2 |      1      |              0.6927 |            0.3073 |                 0.6111 |                       0.6111 |           0      |

### `gemma_4-31B` vs `logistic_regression`

**Per-class agreement** — agreement rate split by gold label, showing which sentiment class the two models disagree on most.

| class    |   n |   agreement |
|:---------|----:|------------:|
| negative | 114 |      0.5    |
| neutral  | 162 |      0.5802 |
| positive | 126 |      0.4921 |

**Prediction cross-tabulation** — rows are `gemma_4-31B` predictions, columns are `logistic_regression` predictions. Off-diagonal cells are disagreements.

| gemma_4-31B \ logistic_regression   |   negative |   neutral |   positive |
|:------------------------------------|-----------:|----------:|-----------:|
| negative                            |         60 |        51 |         17 |
| neutral                             |         34 |        95 |         22 |
| positive                            |         22 |        43 |         58 |

**Linguistic feature divergence** — for each surface feature, how much it shifts agreement between the two models (Δagreement) and each model's macro-F1 on tweets containing that feature. Sorted by Δagreement ascending so the most divergence-inducing features appear first.

| feature        |   count |   pct |   agreement |   agreement_without |   delta_agreement |   f1_gemma_4-31B |   f1_logistic_regression |   f1_delta (b-a) |
|:---------------|--------:|------:|------------:|--------------------:|------------------:|-----------------:|-------------------------:|-----------------:|
| emoji          |       9 |   2.2 |      0.3333 |              0.5344 |           -0.201  |           0.65   |                   0.546  |          -0.104  |
| exclamation    |      61 |  15.2 |      0.459  |              0.5425 |           -0.0835 |           0.7968 |                   0.4251 |          -0.3717 |
| negation       |      74 |  18.4 |      0.4865 |              0.5396 |           -0.0531 |           0.7959 |                   0.4709 |          -0.325  |
| question       |      24 |   6   |      0.5417 |              0.5291 |            0.0126 |           0.8719 |                   0.467  |          -0.4049 |
| sarcasm_marker |       5 |   1.2 |      0.6    |              0.529  |            0.071  |           0.6111 |                   0.7778 |           0.1667 |
| short_text     |     123 |  30.6 |      0.5854 |              0.5054 |            0.08   |           0.9206 |                   0.5575 |          -0.3631 |
| allcaps        |      18 |   4.5 |      0.6111 |              0.526  |            0.0851 |           0.7103 |                   0.5427 |          -0.1676 |
| superlative    |      31 |   7.7 |      0.6774 |              0.5175 |            0.1599 |           0.7774 |                   0.5058 |          -0.2715 |
| emoticon       |       1 |   0.2 |      1      |              0.5287 |            0.4713 |           1      |                   1      |           0      |

### `gemma_4-31B` vs `naive_bayes`

**Per-class agreement** — agreement rate split by gold label, showing which sentiment class the two models disagree on most.

| class    |   n |   agreement |
|:---------|----:|------------:|
| negative | 114 |      0.5702 |
| neutral  | 162 |      0.3951 |
| positive | 126 |      0.5317 |

**Prediction cross-tabulation** — rows are `gemma_4-31B` predictions, columns are `naive_bayes` predictions. Off-diagonal cells are disagreements.

| gemma_4-31B \ naive_bayes   |   negative |   neutral |   positive |
|:----------------------------|-----------:|----------:|-----------:|
| negative                    |         66 |        40 |         22 |
| neutral                     |         49 |        61 |         41 |
| positive                    |         29 |        25 |         69 |

**Linguistic feature divergence** — for each surface feature, how much it shifts agreement between the two models (Δagreement) and each model's macro-F1 on tweets containing that feature. Sorted by Δagreement ascending so the most divergence-inducing features appear first.

| feature        |   count |   pct |   agreement |   agreement_without |   delta_agreement |   f1_gemma_4-31B |   f1_naive_bayes |   f1_delta (b-a) |
|:---------------|--------:|------:|------------:|--------------------:|------------------:|-----------------:|-----------------:|-----------------:|
| emoticon       |       1 |   0.2 |      0      |              0.4888 |           -0.4888 |           1      |           0      |          -1      |
| exclamation    |      61 |  15.2 |      0.4262 |              0.4985 |           -0.0723 |           0.7968 |           0.3651 |          -0.4317 |
| negation       |      74 |  18.4 |      0.4459 |              0.497  |           -0.051  |           0.7959 |           0.3796 |          -0.4163 |
| question       |      24 |   6   |      0.4583 |              0.4894 |           -0.0311 |           0.8719 |           0.4292 |          -0.4426 |
| short_text     |     123 |  30.6 |      0.5122 |              0.4767 |            0.0355 |           0.9206 |           0.516  |          -0.4046 |
| emoji          |       9 |   2.2 |      0.5556 |              0.486  |            0.0696 |           0.65   |           0.5683 |          -0.0817 |
| allcaps        |      18 |   4.5 |      0.5556 |              0.4844 |            0.0712 |           0.7103 |           0.5527 |          -0.1577 |
| superlative    |      31 |   7.7 |      0.6452 |              0.4744 |            0.1708 |           0.7774 |           0.4565 |          -0.3209 |
| sarcasm_marker |       5 |   1.2 |      0.8    |              0.4836 |            0.3164 |           0.6111 |           0.7778 |           0.1667 |

### `gemma_4-31B` vs `weighted-naive-bayes-lr`

**Per-class agreement** — agreement rate split by gold label, showing which sentiment class the two models disagree on most.

| class    |   n |   agreement |
|:---------|----:|------------:|
| negative | 114 |      0.4211 |
| neutral  | 162 |      0.6667 |
| positive | 126 |      0.5238 |

**Prediction cross-tabulation** — rows are `gemma_4-31B` predictions, columns are `weighted-naive-bayes-lr` predictions. Off-diagonal cells are disagreements.

| gemma_4-31B \ weighted-naive-bayes-lr   |   negative |   neutral |   positive |
|:----------------------------------------|-----------:|----------:|-----------:|
| negative                                |         48 |        72 |          8 |
| neutral                                 |         18 |       121 |         12 |
| positive                                |         16 |        54 |         53 |

**Linguistic feature divergence** — for each surface feature, how much it shifts agreement between the two models (Δagreement) and each model's macro-F1 on tweets containing that feature. Sorted by Δagreement ascending so the most divergence-inducing features appear first.

| feature        |   count |   pct |   agreement |   agreement_without |   delta_agreement |   f1_gemma_4-31B |   f1_weighted-naive-bayes-lr |   f1_delta (b-a) |
|:---------------|--------:|------:|------------:|--------------------:|------------------:|-----------------:|-----------------------------:|-----------------:|
| negation       |      74 |  18.4 |      0.3784 |              0.5915 |           -0.2131 |           0.7959 |                       0.4484 |          -0.3475 |
| exclamation    |      61 |  15.2 |      0.4262 |              0.5748 |           -0.1486 |           0.7968 |                       0.3821 |          -0.4147 |
| question       |      24 |   6   |      0.5    |              0.5556 |           -0.0556 |           0.8719 |                       0.4387 |          -0.4332 |
| allcaps        |      18 |   4.5 |      0.5    |              0.5547 |           -0.0547 |           0.7103 |                       0.3905 |          -0.3198 |
| emoji          |       9 |   2.2 |      0.5556 |              0.5522 |            0.0034 |           0.65   |                       0.4444 |          -0.2056 |
| superlative    |      31 |   7.7 |      0.7097 |              0.5391 |            0.1706 |           0.7774 |                       0.6306 |          -0.1467 |
| short_text     |     123 |  30.6 |      0.6748 |              0.4982 |            0.1766 |           0.9206 |                       0.6218 |          -0.2988 |
| sarcasm_marker |       5 |   1.2 |      0.8    |              0.5491 |            0.2509 |           0.6111 |                       0.6111 |           0      |
| emoticon       |       1 |   0.2 |      1      |              0.5511 |            0.4489 |           1      |                       1      |           0      |

### `logistic_regression` vs `naive_bayes`

**Per-class agreement** — agreement rate split by gold label, showing which sentiment class the two models disagree on most.

| class    |   n |   agreement |
|:---------|----:|------------:|
| negative | 114 |      0.6667 |
| neutral  | 162 |      0.5926 |
| positive | 126 |      0.7222 |

**Prediction cross-tabulation** — rows are `logistic_regression` predictions, columns are `naive_bayes` predictions. Off-diagonal cells are disagreements.

| logistic_regression \ naive_bayes   |   negative |   neutral |   positive |
|:------------------------------------|-----------:|----------:|-----------:|
| negative                            |         87 |        14 |         15 |
| neutral                             |         44 |       102 |         43 |
| positive                            |         13 |        10 |         74 |

**Linguistic feature divergence** — for each surface feature, how much it shifts agreement between the two models (Δagreement) and each model's macro-F1 on tweets containing that feature. Sorted by Δagreement ascending so the most divergence-inducing features appear first.

| feature        |   count |   pct |   agreement |   agreement_without |   delta_agreement |   f1_logistic_regression |   f1_naive_bayes |   f1_delta (b-a) |
|:---------------|--------:|------:|------------:|--------------------:|------------------:|-------------------------:|-----------------:|-----------------:|
| emoticon       |       1 |   0.2 |      0      |              0.6559 |           -0.6559 |                   1      |           0      |          -1      |
| question       |      24 |   6   |      0.5417 |              0.6614 |           -0.1197 |                   0.467  |           0.4292 |          -0.0377 |
| short_text     |     123 |  30.6 |      0.5854 |              0.6846 |           -0.0992 |                   0.5575 |           0.516  |          -0.0415 |
| sarcasm_marker |       5 |   1.2 |      0.6    |              0.6549 |           -0.0549 |                   0.7778 |           0.7778 |           0      |
| negation       |      74 |  18.4 |      0.6486 |              0.6555 |           -0.0068 |                   0.4709 |           0.3796 |          -0.0913 |
| emoji          |       9 |   2.2 |      0.6667 |              0.6539 |            0.0127 |                   0.546  |           0.5683 |           0.0222 |
| allcaps        |      18 |   4.5 |      0.6667 |              0.6536 |            0.013  |                   0.5427 |           0.5527 |           0.0099 |
| exclamation    |      61 |  15.2 |      0.6885 |              0.6481 |            0.0404 |                   0.4251 |           0.3651 |          -0.06   |
| superlative    |      31 |   7.7 |      0.7419 |              0.6469 |            0.095  |                   0.5058 |           0.4565 |          -0.0494 |

### `logistic_regression` vs `weighted-naive-bayes-lr`

**Per-class agreement** — agreement rate split by gold label, showing which sentiment class the two models disagree on most.

| class    |   n |   agreement |
|:---------|----:|------------:|
| negative | 114 |      0.6842 |
| neutral  | 162 |      0.7284 |
| positive | 126 |      0.8333 |

**Prediction cross-tabulation** — rows are `logistic_regression` predictions, columns are `weighted-naive-bayes-lr` predictions. Off-diagonal cells are disagreements.

| logistic_regression \ weighted-naive-bayes-lr   |   negative |   neutral |   positive |
|:------------------------------------------------|-----------:|----------:|-----------:|
| negative                                        |         66 |        45 |          5 |
| neutral                                         |         12 |       172 |          5 |
| positive                                        |          4 |        30 |         63 |

**Linguistic feature divergence** — for each surface feature, how much it shifts agreement between the two models (Δagreement) and each model's macro-F1 on tweets containing that feature. Sorted by Δagreement ascending so the most divergence-inducing features appear first.

| feature        |   count |   pct |   agreement |   agreement_without |   delta_agreement |   f1_logistic_regression |   f1_weighted-naive-bayes-lr |   f1_delta (b-a) |
|:---------------|--------:|------:|------------:|--------------------:|------------------:|-------------------------:|-----------------------------:|-----------------:|
| question       |      24 |   6   |      0.5417 |              0.7619 |           -0.2202 |                   0.467  |                       0.4387 |          -0.0283 |
| allcaps        |      18 |   4.5 |      0.6667 |              0.7526 |           -0.0859 |                   0.5427 |                       0.3905 |          -0.1523 |
| negation       |      74 |  18.4 |      0.6892 |              0.7622 |           -0.073  |                   0.4709 |                       0.4484 |          -0.0225 |
| superlative    |      31 |   7.7 |      0.7097 |              0.752  |           -0.0423 |                   0.5058 |                       0.6306 |           0.1248 |
| emoji          |       9 |   2.2 |      0.7778 |              0.7481 |            0.0297 |                   0.546  |                       0.4444 |          -0.1016 |
| sarcasm_marker |       5 |   1.2 |      0.8    |              0.7481 |            0.0519 |                   0.7778 |                       0.6111 |          -0.1667 |
| exclamation    |      61 |  15.2 |      0.8033 |              0.739  |            0.0643 |                   0.4251 |                       0.3821 |          -0.043  |
| short_text     |     123 |  30.6 |      0.7967 |              0.7276 |            0.0691 |                   0.5575 |                       0.6218 |           0.0643 |
| emoticon       |       1 |   0.2 |      1      |              0.7481 |            0.2519 |                   1      |                       1      |           0      |

### `naive_bayes` vs `weighted-naive-bayes-lr`

**Per-class agreement** — agreement rate split by gold label, showing which sentiment class the two models disagree on most.

| class    |   n |   agreement |
|:---------|----:|------------:|
| negative | 114 |      0.5965 |
| neutral  | 162 |      0.537  |
| positive | 126 |      0.6667 |

**Prediction cross-tabulation** — rows are `naive_bayes` predictions, columns are `weighted-naive-bayes-lr` predictions. Off-diagonal cells are disagreements.

| naive_bayes \ weighted-naive-bayes-lr   |   negative |   neutral |   positive |
|:----------------------------------------|-----------:|----------:|-----------:|
| negative                                |         66 |        72 |          6 |
| neutral                                 |         10 |       111 |          5 |
| positive                                |          6 |        64 |         62 |

**Linguistic feature divergence** — for each surface feature, how much it shifts agreement between the two models (Δagreement) and each model's macro-F1 on tweets containing that feature. Sorted by Δagreement ascending so the most divergence-inducing features appear first.

| feature        |   count |   pct |   agreement |   agreement_without |   delta_agreement |   f1_naive_bayes |   f1_weighted-naive-bayes-lr |   f1_delta (b-a) |
|:---------------|--------:|------:|------------:|--------------------:|------------------:|-----------------:|-----------------------------:|-----------------:|
| emoticon       |       1 |   0.2 |      0      |              0.596  |           -0.596  |           0      |                       1      |           1      |
| question       |      24 |   6   |      0.4167 |              0.6058 |           -0.1892 |           0.4292 |                       0.4387 |           0.0094 |
| negation       |      74 |  18.4 |      0.5135 |              0.6128 |           -0.0993 |           0.3796 |                       0.4484 |           0.0688 |
| short_text     |     123 |  30.6 |      0.5447 |              0.6165 |           -0.0718 |           0.516  |                       0.6218 |           0.1058 |
| allcaps        |      18 |   4.5 |      0.5556 |              0.5964 |           -0.0408 |           0.5527 |                       0.3905 |          -0.1622 |
| emoji          |       9 |   2.2 |      0.6667 |              0.5929 |            0.0738 |           0.5683 |                       0.4444 |          -0.1238 |
| superlative    |      31 |   7.7 |      0.7097 |              0.5849 |            0.1248 |           0.4565 |                       0.6306 |           0.1742 |
| exclamation    |      61 |  15.2 |      0.7049 |              0.5748 |            0.1301 |           0.3651 |                       0.3821 |           0.017  |
| sarcasm_marker |       5 |   1.2 |      0.8    |              0.5919 |            0.2081 |           0.7778 |                       0.6111 |          -0.1667 |

