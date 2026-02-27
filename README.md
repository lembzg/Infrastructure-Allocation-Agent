# Infrastructure Allocation Agent

A self-correcting investment recommendation engine built for the BNY AI Hub Challenge. Given structured financial data, unstructured news, and a client profile, the agent scores and ranks UK renewable infrastructure companies and recommends the best fit for a cautious institutional investor.

## How It Works

The agent reads 3 input files and produces a ranked recommendation with a calibrated confidence score.

```
companies.csv → Financial & risk data
news.txt      → Unstructured news per company
client_memo.txt → Investor preferences & constraints
```

### Pipeline

```
Load CSV → Clean & impute missing data → Extract news sentiment → Translate client constraints → Score → Rank → Output JSON
```

### Three-Scorer Ensemble

Each company receives three independent scores (0–1), then combined via weighted average:

| Scorer | What it measures | Weight | Why |
|--------|-----------------|--------|-----|
| **Financial** | Revenue growth (40%) + EBITDA margin (60%) | 30% | Profitability matters more than growth |
| **Risk** | Volatility + leverage + operational risk, with hard penalties for constraint breaches | 45% | Heaviest weight — client is risk-averse |
| **News Sentiment** | Keyword-based NLP: positive vs negative words, discounted by uncertainty | 25% | Captures real-time signals not in the numbers |

```
Final Score = 0.30 × Financial + 0.45 × Risk + 0.25 × News
```

### Data Cleaning

- Missing values detected and flagged (`?`, `N/A`, empty)
- Corrupted rows identified via `Data_Quality_Flag`
- Missing ESG imputed using **penalised median** (median × 0.9 for corrupted data)

### Constraint Translation

Client language is converted into numeric thresholds:

| Client says | Threshold set |
|-------------|--------------|
| "Avoid high volatility" | `max_volatility = 20` |
| "Moderate risk tolerance" | `max_volatility = 25` |
| "Sensitive to excessive leverage" | `max_debt_to_equity = 1.5` |
| "ESG is important but not at expense of financial health" | `min_esg = 65` |
| "ESG is important" | `min_esg = 75` |

### Confidence Calibration

Confidence starts at 0.80 and is reduced by:
- **-0.15** if top two companies are within 0.05 of each other
- **-0.07** per corrupted data flag
- **-0.05** per imputed value

This avoids overconfidence when the data is uncertain.

## Usage

Place these files in one folder:

```
agent.py
keywords.json
companies.csv
news.txt
client_memo.txt
```

Run:

```bash
python agent.py .
```

Output:

```
RECOMMENDATION: Helios Renewables
CONFIDENCE: 0.58
RANKING: Helios Renewables > GreenGrid Storage > EcoGen Power > NorthSea Wind > SolaraTech UK
Saved to: submission.json
```

## Dependencies

Python 3.x standard library only. No pip installs required.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│ companies.csv│     │   news.txt   │     │ client_memo.txt │
└──────┬──────┘     └──────┬───────┘     └────────┬────────┘
       │                   │                      │
  Load & Clean      Extract Sentiment     Translate Constraints
       │                   │                      │
       ▼                   ▼                      ▼
┌──────────────┐  ┌───────────────┐  ┌─────────────────────┐
│  Financial   │  │     News      │  │        Risk          │
│  Scorer 30%  │  │  Scorer 25%   │  │     Scorer 45%       │
└──────┬───────┘  └───────┬───────┘  └──────────┬──────────┘
       │                  │                     │
       └──────────────────┼─────────────────────┘
                          ▼
                  Weighted Average
                          ▼
                   Rank & Recommend
                          ▼
                   submission.json
```

## Built For

BNY AI Hub Challenge — University of Manchester AI & Data Science Society
