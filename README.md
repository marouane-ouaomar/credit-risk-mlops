# Credit Risk Scoring API

![CI](https://github.com/marouane-ouaomar/credit-risk-mlops/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)

Live demo: https://credit-risk-mlops-production.up.railway.app/docs

## About

This is a project I built to predict whether a loan applicant will default,
using the Home Credit Default Risk dataset from Kaggle (about 307k
applicants across 6 related tables). I didn't want it to be another
notebook that ends at "here's my accuracy score," so I took it all the way
through: cleaning and joining the data, training the model, wrapping it in
an API, containerizing it, testing it in CI, and deploying it so anyone can
actually hit it and get a prediction back.

Given an applicant's info, and optionally their credit history from other
banks and previous loans with this lender, the API returns a probability
of default, a decision (approve, review, or reject), and the top reasons
behind that decision using SHAP.

## The problem

Home Credit wants to know, before approving a loan, how likely someone is
to default. That's it in one sentence, but the interesting part is how you
turn "predict a probability" into something a bank could actually use: you
need a model that's accurate, but also explainable enough to justify a
rejection, and a system that can score a brand new customer with zero
credit history just as well as someone with years of records.

## Results

I got an AUC around 0.78 to 0.80 on the validation set, using 11
application level features (income, credit amount, employment length,
external bureau scores, etc.) plus 10 features aggregated from the
applicant's credit history. I also track the KS statistic, since that's
the metric actual credit risk teams look at given how imbalanced the data
is (only about 10% of applicants default).

## What's in it

The project pulls in all 6 tables from the dataset, not just the main
application file. bureau.csv and bureau_balance.csv give you an
applicant's credit history with other institutions. previous_application.csv
covers their past loans with Home Credit itself. POS_CASH_balance.csv,
installments_payments.csv, and credit_card_balance.csv track how they've
handled past payments. I aggregate all of that into a feature store, the
same way a bank's actual pipeline would, so the model and the API only
ever deal with clean, aggregated numbers instead of raw transaction logs.

Every prediction comes back with a short explanation, using SHAP values,
so you can see which features pushed the score up or down. I also handle
new customers explicitly. If someone has no bureau history or no past
applications, the API flags that instead of quietly filling in a zero
that would misleadingly look like "definitely safe."

It's containerized with Docker and actually deployed on Railway, not just
something that runs on my laptop. GitHub Actions runs the whole pipeline
on every push using synthetic data, so I don't need real Kaggle
credentials sitting in CI. There's also a small drift monitoring script
that compares live traffic against the training data using a
Kolmogorov Smirnov test, which is basically the answer to "how would you
know if your model quietly broke three months from now."

## Tech I used

Python, pandas, XGBoost, scikit learn, and SHAP for the modeling side.
FastAPI, Pydantic, and Uvicorn to serve it. pytest for testing. Docker and
GitHub Actions for the pipeline, and Railway to actually host it. SciPy
for the drift test and SQLite to log requests.

## Running it yourself

Clone the repo and set up your environment:

```bash
git clone https://github.com/marouane-ouaomar/credit-risk-mlops.git
cd credit-risk-mlops
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Grab the dataset from Kaggle (you'll need to join the competition once,
it's free): https://www.kaggle.com/c/home-credit-default-risk, then drop
all the CSVs into data/raw/.

Then run these in order:

```bash
python -m src.build_features
python -m src.data_processing
python -m src.train
```

And start the API:

```bash
uvicorn app.main:app --reload --port 8000
```

Open http://localhost:8000/docs and you'll get an interactive page where
you can try it out directly.

## Using the API

Here's a request for an existing customer, where I already know their
credit history (in a real bank this would come from a lookup, not
something a loan officer types by hand):

```bash
curl -X POST https://credit-risk-mlops-production.up.railway.app/predict \
  -H "Content-Type: application/json" \
  -d '{
    "AMT_INCOME_TOTAL": 180000, "AMT_CREDIT": 500000, "AMT_ANNUITY": 25000,
    "DAYS_BIRTH": -14000, "DAYS_EMPLOYED": -2500, "DAYS_ID_PUBLISH": -3000,
    "DAYS_LAST_PHONE_CHANGE": -500, "CNT_CHILDREN": 1,
    "EXT_SOURCE_1": 0.55, "EXT_SOURCE_2": 0.60, "EXT_SOURCE_3": 0.50,
    "CODE_GENDER": "F", "FLAG_OWN_CAR": "N", "FLAG_OWN_REALTY": "Y",
    "NAME_EDUCATION_TYPE": "Higher education",
    "credit_history": {
      "BUREAU_COUNT": 5, "BUREAU_ACTIVE_COUNT": 2,
      "BUREAU_CREDIT_SUM_DEBT_MEAN": 40000, "BUREAU_CREDIT_DAY_OVERDUE_MAX": 0,
      "PREV_APP_COUNT": 3, "PREV_APP_APPROVED_RATE": 0.8, "PREV_APP_REFUSED_RATE": 0.2,
      "INSTALLMENTS_LATE_RATE": 0.02, "CC_UTILIZATION_MEAN": 0.25, "CC_DPD_MEAN": 0
    }
  }'
```

Which gives you something like:

```json
{
  "probability_of_default": 0.084,
  "risk_score": 916,
  "decision": "APPROVE",
  "top_reasons": [
    {"feature": "BUREAU_CREDIT_SUM_DEBT_MEAN", "impact": -0.041},
    {"feature": "EXT_SOURCE_2", "impact": -0.031}
  ],
  "used_credit_history": true,
  "model_version": "2026-07-17"
}
```

If it's a brand new customer, just leave out the credit_history block
entirely. The model still gives you a score, using training medians for
the missing pieces instead of guessing zero.

There's also GET /health and GET /model-info if you want to check the
service is alive or see what metrics the current model was trained with.

## How it's organized

app holds the FastAPI app and the request and response schemas.
src has the actual pipeline: build_features.py builds the feature store
from the 5 auxiliary tables, data_processing.py joins everything and
cleans it, train.py trains the model and computes SHAP values, and
monitor.py checks for drift. tests has the API test suite. models is
where the trained model, encoder, explainer, and metrics get saved.
data holds the raw files, processed files, and request logs.

## A few choices I want to explain

I didn't throw every possible engineered feature at the model. There are
roughly 150 you could build across all 6 tables, and I picked 21. Each one
has a clear enough meaning that I could explain a rejection to a loan
officer or a regulator without hand waving, and that felt more important
than squeezing out another point of AUC.

New customers with no history are treated as their own case, not an edge
case. A real chunk of applicants have never had a loan or a credit bureau
record before, so I added flags for that instead of pretending a missing
value just means zero risk.

Every prediction explains itself using SHAP, since a real risk model
often needs to justify individual decisions, not just perform well on
average.

The trained model files are committed straight into the repo instead of
pulled from somewhere at run time. For a single model that doesn't get
retrained constantly, that's a reasonable shortcut. A bigger production
system would pull from something like MLflow or S3 instead, and I'd
rather say that outright than pretend this setup scales infinitely.

## Testing

```bash
pytest tests/ -v
```

GitHub Actions runs this same pipeline on every push, using generated
synthetic data instead of the real Kaggle files, since I don't want
Kaggle credentials or multi gigabyte datasets sitting in CI.

## What I'd add next

A proper SHAP force plot endpoint instead of just the top 5 reasons as
JSON. A small Streamlit page so non technical people can try it without
touching the API directly. MLflow for tracking experiments once I start
retraining more often. And eventually an endpoint that looks up an
applicant's credit history by ID instead of requiring it in the request,
which is closer to how this would actually work at a bank.

## License

MIT
