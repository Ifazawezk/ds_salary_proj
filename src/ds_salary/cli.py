"""Estimate a salary from the command line.

    ds-salary --experience Senior --family "Data Science" --country US --mode Remote
"""

import argparse

from ds_salary.data import COMPANY_SIZE, EMPLOYMENT, EXPERIENCE, REMOTE, TITLE_FAMILIES, load
from ds_salary.model import predict, train_and_evaluate


def main(argv=None):
    p = argparse.ArgumentParser(prog="ds-salary", description="Estimate an AI/ML/data salary in USD.")
    p.add_argument("--experience", choices=list(EXPERIENCE.values()), default="Mid")
    p.add_argument("--family", choices=[f for f, _ in TITLE_FAMILIES] + ["Other"], default="Data Science")
    p.add_argument("--country", default="US", help="company country, ISO code (e.g. US, GB, DE)")
    p.add_argument("--size", choices=list(COMPANY_SIZE.values()), default="Medium")
    p.add_argument("--mode", choices=list(REMOTE.values()), default="On-site")
    p.add_argument("--employment", choices=list(EMPLOYMENT.values()), default="Full-time")
    p.add_argument("--year", type=int, default=2025)
    p.add_argument("--abroad", action="store_true", help="employee lives in a different country")
    args = p.parse_args(argv)

    df = load()
    pipe, ev = train_and_evaluate(df)
    country = args.country.upper()
    estimate = predict(
        pipe,
        work_year=args.year,
        experience=args.experience,
        employment=args.employment,
        family=args.family,
        company_location=country,
        company_size_label=args.size,
        work_mode=args.mode,
        same_country=not args.abroad,
    )
    n = (df["company_location"] == country).sum()
    print(f"Estimated salary: ${estimate:,.0f} per year")
    print(f"(typical error about ${ev.mae_model:,.0f}; {n:,} survey responses from companies in {country})")
    if n < 50:
        print("Warning: few responses for this country, so the estimate leans on other countries.")


if __name__ == "__main__":
    main()
