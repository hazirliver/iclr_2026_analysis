from io import StringIO
import requests
from pprint import pprint

import polars as pl


url = (
    "https://github.com/papercopilot/paperlists/raw/refs/heads/main/iclr/iclr2026.json"
)

response = requests.get(url)
iclr_2026_raw = pl.read_json(StringIO(response.text))

iclr_2026_raw = iclr_2026_raw.with_columns(
    openreview_id=pl.col("site").str.extract(r"id=(.+)$"),
    pdf_link=pl.col("site").str.replace("forum", "pdf"),
)
iclr_2026_raw.write_parquet("iclr_2026_raw.parquet")
iclr_2026_raw = pl.read_parquet("iclr_2026_raw.parquet")


iclr_2026_raw.head()


accepted_statuses = [
    "Oral",
    "ICLR 2026 ConditionalOral",
    "Poster",
    "ICLR 2026 ConditionalPoster",
]

iclr_2026_accepted = iclr_2026_raw.filter(pl.col("status").is_in(accepted_statuses))
print(iclr_2026_accepted)

iclr_2026_accepted.describe()

# All of the are either null-only or contains 1 unique meaningless value
columns_to_drop = [
    "aff_unique_abbr",
    "aff_unique_dep",
    "github",
    "aff_unique_norm",
    "google_scholar",
    "linkedin",
    "authorids",
    "aff_campus_unique_index",
    "aff",
    "aff_campus_unique",
    "gs_cited_by_link",
    "tldr",
    "homepage",
    "aff_unique_url",
    "aff_country_unique_index",
    "aff_domain",
    "author",
    "gender",
    "project",
    "position",
    "or_profile",
    "aff_unique_index",
    "aff_country_unique",
    "dblp",
    "orcid",
    "gs_version_total",
    "gs_citation",
    "reply_reviewers",
    "wc_reply_reviewers",
    "pdf_size",
    "reply_authors_avg",
    "reply_reviewers_avg",
    "wc_reply_reviewers_avg",
    "track",
    "reviewers",
    "reply_authors",
    "wc_reply_authors",
    "wc_reply_authors_avg",
    "id",
    "bibtex",
    "supplementary_material",
]

iclr_2026_accepted_2 = iclr_2026_accepted.drop(columns_to_drop)

iclr_2026_accepted_3 = iclr_2026_accepted_2.with_columns(
    pl.col("wc_summary").str.split(by=";").list.eval(pl.element().cast(pl.Int64)),
    pl.col("wc_strengths").str.split(by=";").list.eval(pl.element().cast(pl.Int64)),
    pl.col("wc_review").str.split(by=";").list.eval(pl.element().cast(pl.Int64)),
    pl.col("contribution").str.split(by=";").list.eval(pl.element().cast(pl.Int64)),
    pl.col("rating").str.split(by=";").list.eval(pl.element().cast(pl.Int64)),
    pl.col("wc_weaknesses").str.split(by=";").list.eval(pl.element().cast(pl.Int64)),
    pl.col("presentation").str.split(by=";").list.eval(pl.element().cast(pl.Int64)),
    pl.col("wc_questions").str.split(by=";").list.eval(pl.element().cast(pl.Int64)),
    pl.col("confidence").str.split(by=";").list.eval(pl.element().cast(pl.Int64)),
    pl.col("soundness").str.split(by=";").list.eval(pl.element().cast(pl.Int64)),
    pl.col("keywords").str.split(by=";"),
)


# Columns + Types
pprint(list(
    zip(
        iclr_2026_accepted_3.columns,
        iclr_2026_accepted_3.dtypes
    )
))

# First 2 rows
iclr_2026_accepted_3.glimpse(max_items_per_column=2)

# Some basic statistics
iclr_2026_accepted_3.describe()



