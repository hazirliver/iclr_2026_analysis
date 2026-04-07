import marimo

__generated_with = "0.22.0"
app = marimo.App()

with app.setup:
    import marimo as mo
    import polars as pl


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Load
    """)
    return


@app.cell
def _():
    df_all = pl.read_parquet("iclr_2026_scored.parquet")
    print(f"Loaded {df_all.shape[0]} rows × {df_all.shape[1]} columns")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
