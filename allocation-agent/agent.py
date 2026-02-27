import csv
import json
import sys
import time
from pathlib import Path


def load_companies(filepath):
    companies = []
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            company = {"name": row["Company"].strip()}
            
            numeric_fields = {
                "revenue_growth": "Revenue_Growth_3Y",
                "ebitda_margin": "EBITDA_Margin",
                "debt_to_equity": "Debt_to_Equity",
                "volatility": "Volatility_1Y",
                "esg_score": "ESG_Score",
            }
            
            data_issues = []
            for key, col in numeric_fields.items():
                raw = row[col].strip()
                if raw in ("?", "", "N/A", "None"):
                    company[key] = None
                    data_issues.append(f"Missing {col}")
                else:
                    try:
                        company[key] = float(raw)
                    except ValueError:
                        company[key] = None
                        data_issues.append(f"Unparseable {col}: {raw}")
            
            company["operational_risk"] = row["Operational_Risk"].strip()
            company["data_quality"] = row["Data_Quality_Flag"].strip()
            company["data_issues"] = data_issues
            company["is_corrupted"] = company["data_quality"] == "CORRUPTED"
            
            companies.append(company)
    return companies


def impute_missing(companies):
    valid_esg = [c["esg_score"] for c in companies 
                 if c["esg_score"] is not None and not c["is_corrupted"]]
    sorted_esg = sorted(valid_esg)
    n = len(sorted_esg)
    median_esg = sorted_esg[n // 2] if n % 2 == 1 else (sorted_esg[n // 2 - 1] + sorted_esg[n // 2]) / 2
    
    for c in companies:
        if c["esg_score"] is None:
            penalty = 0.9 if c["is_corrupted"] else 1.0
            c["esg_score"] = round(median_esg * penalty, 1)
            c["esg_imputed"] = True
        else:
            c["esg_imputed"] = False


def extract_news_signals(filepath):
    with open(filepath, "r") as f:
        text = f.read()
    
    with open("keywords.json", "r") as f:
        keywords = json.load(f)
    
    paragraphs = [p.strip() for p in text.strip().split("\n\n") if p.strip()]
    news_signals = {}
    
    for para in paragraphs:
        company_name = " ".join(para.split()[:2])
        para_lower = para.lower()
        
        pos = sum(1 for kw in keywords["positive"] if kw in para_lower)
        neg = sum(1 for kw in keywords["negative"] if kw in para_lower)
        unc = sum(1 for kw in keywords["uncertainty"] if kw in para_lower)
        
        raw = (pos - neg) / max(pos + neg, 1)
        discount = 1.0 - (0.15 * unc)
        sentiment = round(raw * max(discount, 0.3), 3)
        
        news_signals[company_name] = sentiment
    
    return news_signals


def translate_constraints(filepath):
    with open(filepath, "r") as f:
        text = f.read().lower()
    
    constraints = {}
    
    if "avoid high volatility" in text or "low volatility" in text:
        constraints["max_volatility"] = 20.0
    elif "moderate risk" in text:
        constraints["max_volatility"] = 25.0
    else:
        constraints["max_volatility"] = 35.0
    
    if "sensitive to excessive leverage" in text or "avoid leverage" in text:
        constraints["max_debt_to_equity"] = 1.5
    elif "moderate" in text:
        constraints["max_debt_to_equity"] = 2.0
    else:
        constraints["max_debt_to_equity"] = 3.0
    
    if "esg is important" in text:
        if "not at the expense" in text:
            constraints["min_esg_score"] = 65
        else:
            constraints["min_esg_score"] = 75
    else:
        constraints["min_esg_score"] = 50
    
    if "long-term stability" in text or "stability preferred" in text:
        constraints["stability_preference"] = True
    else:
        constraints["stability_preference"] = False
    
    return constraints



def normalize(value, min_val, max_val, invert=False):
    if max_val == min_val:
        return 0.5
    score = (value - min_val) / (max_val - min_val)
    score = max(0.0, min(1.0, score))
    return 1.0 - score if invert else score


def scorer_financial(companies):
    scores = {}
    growths = [c["revenue_growth"] for c in companies]
    margins = [c["ebitda_margin"] for c in companies]
    
    for c in companies:
        g = normalize(c["revenue_growth"], min(growths), max(growths))
        m = normalize(c["ebitda_margin"], min(margins), max(margins))
        scores[c["name"]] = round(0.4 * g + 0.6 * m, 4)
    return scores


def scorer_risk(companies, constraints):
    scores = {}
    vols = [c["volatility"] for c in companies]
    dtes = [c["debt_to_equity"] for c in companies]
    risk_map = {"Low": 1.0, "Medium": 0.6, "High": 0.2}
    
    for c in companies:
        vol = normalize(c["volatility"], min(vols), max(vols), invert=True)
        dte = normalize(c["debt_to_equity"], min(dtes), max(dtes), invert=True)
        op = risk_map.get(c["operational_risk"], 0.5)
        
        penalty = 0.0
        if c["volatility"] > constraints["max_volatility"]:
            penalty += 0.3
        if c["debt_to_equity"] > constraints["max_debt_to_equity"]:
            penalty += 0.3
        
        raw = 0.35 * vol + 0.35 * dte + 0.30 * op
        scores[c["name"]] = round(max(raw - penalty, 0.0), 4)
    return scores


def scorer_news(companies, news_signals):
    scores = {}
    for c in companies:
        sentiment = news_signals.get(c["name"], 0.0)
        score = (sentiment + 1) / 2
        if c["is_corrupted"]:
            score *= 0.85
        scores[c["name"]] = round(score, 4)
    return scores



def run_agent(data_dir):
    start = time.time()
    data_dir = Path(data_dir)
    
    companies = load_companies(data_dir / "companies.csv")
    impute_missing(companies)
    news_signals = extract_news_signals(data_dir / "news.txt")
    constraints = translate_constraints(data_dir / "client_memo.txt")
    
    fin = scorer_financial(companies)
    risk = scorer_risk(companies, constraints)
    news = scorer_news(companies, news_signals)
    
    final = {}
    for name in fin:
        final[name] = round(0.30 * fin[name] + 0.45 * risk[name] + 0.25 * news[name], 4)
    
    ranking = sorted(final.items(), key=lambda x: x[1], reverse=True)
    ranked_names = [name for name, _ in ranking]
    recommended = ranked_names[0]
    
    top_gap = ranking[0][1] - ranking[1][1] if len(ranking) >= 2 else 0
    confidence = 0.80
    if top_gap < 0.05:
        confidence -= 0.15
    if any(c["is_corrupted"] for c in companies):
        confidence -= 0.07
    if any(c.get("esg_imputed") for c in companies):
        confidence -= 0.05
    confidence = round(max(0.3, confidence), 2)
    
    uncertainty_factors = []
    if top_gap < 0.05:
        uncertainty_factors.append(f"Top 2 scores very close (gap={top_gap:.3f})")
    if any(c["is_corrupted"] for c in companies):
        uncertainty_factors.append("Corrupted data present")
    if any(c.get("esg_imputed") for c in companies):
        uncertainty_factors.append("ESG values were imputed")
    
    submission = {
        "team_name": "Arryl's Team",
        "members": ["Arryl"],
        "architecture_summary": (
            "Three-scorer ensemble: financial strength (growth + margin), "
            "risk assessment (volatility + leverage + operational risk with "
            "constraint penalties), and news sentiment. Weighted combination "
            "with risk heaviest (45%) for cautious investor profile."
        ),
        "data_cleaning_steps": [
            "Parsed CSV with type validation; missing values set to None",
            "Flagged CORRUPTED rows for penalty treatment",
            "Imputed missing ESG via penalized median (0.9x for corrupted)",
        ],
        "constraint_translation": {
            "moderate_risk": f"Volatility below {constraints['max_volatility']}",
            "avoid_excess_leverage": f"Debt_to_Equity below {constraints['max_debt_to_equity']}",
            "esg_priority": f"Minimum ESG score {constraints['min_esg_score']}",
            "stability": str(constraints["stability_preference"]),
        },
        "scoring_methodology": {
            "financial_weight": 0.30,
            "risk_weight": 0.45,
            "news_weight": 0.25,
            "description": (
                "Financial: 40% growth + 60% margin. "
                "Risk: normalized volatility + leverage + operational risk, "
                "minus 0.3 penalty per constraint breach. "
                "News: keyword sentiment [-1,1] mapped to [0,1]. "
                "Combined: 30% financial + 45% risk + 25% news."
            ),
        },
        "final_ranking": ranked_names,
        "recommended_company": recommended,
        "confidence_score": confidence,
        "uncertainty_factors": uncertainty_factors,
        "runtime_seconds": round(time.time() - start, 1),
    }
    
    return submission


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    
    submission = run_agent(data_dir)
    
    output_path = Path(data_dir) / "submission.json"
    with open(output_path, "w") as f:
        json.dump(submission, f, indent=2)
    
    print(f"RECOMMENDATION: {submission['recommended_company']}")
    print(f"CONFIDENCE: {submission['confidence_score']}")
    print(f"RANKING: {' > '.join(submission['final_ranking'])}")
    print(f"Saved to: {output_path}")






















