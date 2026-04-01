import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")

with app.setup:
    import time
    import random
    import json
    from pathlib import Path

    import marimo as mo
    import polars as pl
    import httpx


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 0. Functions
    """)
    return


@app.function
def download_pdfs(
    df: pl.DataFrame,
    base_dir: str | Path = "pdfs",
    timeout: float = 60.0,
    min_delay: float = 2.0,
    max_delay: float = 7.0,
) -> pl.DataFrame | None:
    """Download PDFs from OpenReview sequentially with resume support.

    Args:
        df: DataFrame with 'openreview_id' and 'pdf_link' columns.
        base_dir: Root folder for downloads.
        timeout: HTTP timeout in seconds.
        min_delay: Min random sleep between downloads.
        max_delay: Max random sleep between downloads.

    Returns:
        DataFrame of failed downloads, or None if all succeeded.
    """
    base_dir = Path(base_dir)
    progress_file = base_dir / "_download_progress.json"

    def load_progress() -> set[str]:
        if progress_file.exists():
            return set(json.loads(progress_file.read_text()))
        return set()

    def save_progress(done: set[str]):
        progress_file.write_text(json.dumps(sorted(done)))

    base_dir.mkdir(exist_ok=True)

    rows = df.select("openreview_id", "pdf_link").to_dicts()
    already_done = load_progress()
    to_download = [r for r in rows if r["openreview_id"] not in already_done]

    mo.output.append(
        mo.md(
            f"**Total:** {len(rows)} · **Already downloaded:** {len(already_done)} · **Remaining:** {len(to_download)}"
        )
    )

    if not to_download:
        mo.output.append(mo.md("Nothing to download."))
        return None

    failed: list[dict] = []

    with mo.status.progress_bar(
        total=len(to_download),
        title="Downloading PDFs",
        subtitle="Starting...",
    ) as bar:
        for i, row in enumerate(to_download):
            oid = row["openreview_id"]
            url = row["pdf_link"]
            out_dir = base_dir / oid
            out_dir.mkdir(exist_ok=True)
            out_path = out_dir / f"{oid}.pdf"

            bar.update(
                title="Downloading PDFs",
                subtitle=f"[{i + 1}/{len(to_download)}] {oid}",
            )

            try:
                headers = {
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
                }
                with httpx.Client(
                    timeout=timeout, follow_redirects=True, headers=headers
                ) as client:
                    resp = client.get(url)
                    resp.raise_for_status()
                    out_path.write_bytes(resp.content)

                already_done.add(oid)
                save_progress(already_done)

            except Exception as e:
                failed.append({"openreview_id": oid, "error": str(e)})

            bar.update(increment=1)

            if i < len(to_download) - 1:
                time.sleep(random.uniform(min_delay, max_delay))

    mo.output.append(
        mo.md(
            f"**Done!** Downloaded {len(to_download) - len(failed)}/{len(to_download)}. "
            + (f"**{len(failed)} failed.**" if failed else "All succeeded.")
        )
    )

    if failed:
        return pl.DataFrame(failed)
    return None


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 1. Fetch Data
    """)
    return


@app.cell
def _():
    return


@app.cell
def _():
    # url = "https://github.com/papercopilot/paperlists/raw/refs/heads/main/iclr/iclr2026.json"

    # response = requests.get(url)
    # iclr_2026_raw = pl.read_json(StringIO(response.text))

    # iclr_2026_raw = iclr_2026_raw.with_columns(
    #     openreview_id=pl.col("site").str.extract(r"id=(.+)$"),
    #     pdf_link=pl.col("site").str.replace("forum", "pdf"),
    # )
    # iclr_2026_raw.write_parquet("iclr_2026_raw.parquet")
    iclr_2026_raw = pl.read_parquet("iclr_2026_raw.parquet")
    return (iclr_2026_raw,)


@app.cell
def _(iclr_2026_raw):
    mo.plain(iclr_2026_raw.head())
    return


@app.cell
def _():
    accepted_statuses = [
        "Oral",
        "ICLR 2026 ConditionalOral",
        "Poster",
        "ICLR 2026 ConditionalPoster",
    ]
    return (accepted_statuses,)


@app.cell
def _(accepted_statuses, iclr_2026_raw):
    iclr_2026_accepted = iclr_2026_raw.filter(pl.col("status").is_in(accepted_statuses))
    iclr_2026_accepted
    return (iclr_2026_accepted,)


@app.cell
def _():
    # # Download pdfs for all accepted papers
    # download_pdfs(iclr_2026_accepted, min_delay=5, max_delay=10)
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 2. Initial analysis
    """)
    return


@app.cell
def _(iclr_2026_accepted):
    iclr_2026_accepted.describe()
    return


@app.cell
def _(iclr_2026_accepted_3):
    iclr_2026_accepted_3.unique("wc_reply_authors_avg")
    return


@app.cell
def _(iclr_2026_accepted_2):
    iclr_2026_accepted_2.filter(
        (pl.col("reply_authors_avg").is_not_null()) & (pl.col("gs_version_total") != -1)
    )
    return


@app.cell
def _(columns_to_drop):
    len(columns_to_drop)
    return


@app.cell
def _(iclr_2026_accepted):
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
    iclr_2026_accepted_2
    return columns_to_drop, iclr_2026_accepted_2


@app.cell
def _(iclr_2026_accepted_2):
    iclr_2026_accepted_3 = iclr_2026_accepted_2.with_columns(
        pl.col("wc_summary").str.split(by=";").list.eval(pl.element().cast(pl.Int64)),
        pl.col("wc_strengths").str.split(by=";").list.eval(pl.element().cast(pl.Int64)),
        pl.col("wc_review").str.split(by=";").list.eval(pl.element().cast(pl.Int64)),
        pl.col("contribution").str.split(by=";").list.eval(pl.element().cast(pl.Int64)),
        pl.col("rating").str.split(by=";").list.eval(pl.element().cast(pl.Int64)),
        pl.col("wc_weaknesses")
        .str.split(by=";")
        .list.eval(pl.element().cast(pl.Int64)),
        pl.col("presentation").str.split(by=";").list.eval(pl.element().cast(pl.Int64)),
        pl.col("wc_questions").str.split(by=";").list.eval(pl.element().cast(pl.Int64)),
        pl.col("confidence").str.split(by=";").list.eval(pl.element().cast(pl.Int64)),
        pl.col("soundness").str.split(by=";").list.eval(pl.element().cast(pl.Int64)),
        pl.col("keywords").str.split(by=";"),
    )
    iclr_2026_accepted_3
    return (iclr_2026_accepted_3,)


@app.cell
def _():
    from pprint import pprint

    return (pprint,)


@app.cell
def _(iclr_2026_accepted_3, pprint):
    pprint(list(zip(iclr_2026_accepted_3.columns, iclr_2026_accepted_3.dtypes)))
    return


@app.cell
def _(iclr_2026_accepted_3):
    iclr_2026_accepted_3.glimpse(max_items_per_column=2)
    return


@app.cell
def _(iclr_2026_accepted_3):
    iclr_2026_accepted_3.describe()
    return


@app.cell
def _(iclr_2026_accepted_3):
    # Save cleaned accepted papers for downstream scripts
    iclr_2026_accepted_3.write_parquet("iclr_2026_accepted.parquet")
    print(
        f"\nSaved {len(iclr_2026_accepted_3)} accepted papers to iclr_2026_accepted.parquet"
    )
    return


@app.cell
def _(iclr_2026_accepted_3):
    iclr_2026_accepted_3
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
