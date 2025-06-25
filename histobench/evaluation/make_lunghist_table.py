from pathlib import Path
import subprocess

import click
import pandas as pd
import yaml

project_dir = Path(__file__).parents[2].resolve()

CONFIG_DIR = project_dir / "configs/LungHist700"
EMBEDDINGS_DIR = project_dir / "data/embeddings/lunghist700"
REPORT_DIR = project_dir / "reports/lunghist700"
METRICS = ["accuracy", "precision", "recall", "f1"]

CSV_METADATA = project_dir / "data/LungHist700/metadata.csv"

data_dir_by_magnification = {
    10: "data/LungHist700/LungHist700_10x",
    20: "data/LungHist700/LungHist700_20x",
}


@click.command()
@click.option("--magnification", default=10, help="Magnification level to use (default: 10x)")
@click.option("--gpu-id", default=0, help="GPU ID to use for computation (default: 0)")
@click.option("--num-workers", default=0, help="Number of workers for data loading (default: 0)")
@click.option("--batch-size", default=32, help="Batch size for model inference (default: 32)")
@click.option("--force", is_flag=True, help="Force recomputation of embeddings and reports")
def main(magnification, gpu_id, num_workers, batch_size, force):
    print("🚀 Starting pipeline...\n")
    config_paths = sorted(CONFIG_DIR.glob("*.yaml"))
    if len(config_paths) == 0:
        print("❌ No configuration files found in the configs directory.")
        return

    name_to_pretty = {}
    input_dir = data_dir_by_magnification.get(magnification, None)

    for cfg_path in config_paths:
        cfg = yaml.safe_load(cfg_path.open())
        name = cfg["name"]
        name_in_table = cfg["name_in_table"]
        name_to_pretty[name] = name_in_table

        emb_path = EMBEDDINGS_DIR / f"{name}.h5"
        report_path = REPORT_DIR / f"{name}_KNNn_{cfg['knn_n_neighbors']}_cv_report.csv"

        if not emb_path.exists() or force:
            print(f"[+] Computing embeddings for {name}")
            subprocess.run(
                [
                    "python",
                    "histobench/evaluation/compute_embeddings_lunghist700.py",
                    "--model",
                    cfg["model"],
                    "--model-weights-path",
                    cfg["weights_path"],
                    "--input-dir",
                    str(input_dir),
                    "--gpu-id",
                    str(gpu_id),
                    "--aggregation",
                    cfg["aggregation"],
                    "--tile-size",
                    str(cfg["tile_size"]),
                    "--batch-size",
                    str(batch_size),
                    "--num-workers",
                    str(num_workers),
                    "--embeddings-path",
                    str(emb_path),
                ]
            )
        else:
            print(f"[=] Embeddings for {name} already exist.")

        if not report_path.exists() or force:
            print(f"[+] Running evaluation for {name}")
            subprocess.run(
                [
                    "python",
                    "histobench/evaluation/evaluate_lunghist700.py",
                    "--csv-metadata",
                    str(CSV_METADATA),
                    "--embeddings-path",
                    str(emb_path),
                    "--knn-n-neighbors",
                    str(cfg["knn_n_neighbors"]),
                    "--report-path",
                    str(report_path),
                ]
            )
        else:
            print(f"[=] Report for {name} already exists.")

    print("\n📊 Generating summary tables...")

    rows = []
    for report_csv in REPORT_DIR.glob("*_cv_report.csv"):
        model_id = report_csv.stem.split("_KNNn_")[0]
        df = pd.read_csv(report_csv)
        pretty_name = name_to_pretty.get(model_id, model_id)

        for clf in df["classifier"].unique():
            sub = df[df["classifier"] == clf]
            row = {"method": f"{pretty_name} ({clf})"}
            for m in METRICS:
                if m in sub.columns:
                    row[m] = f"{sub[m].mean():.3f} ± {sub[m].std():.3f}"
            rows.append(row)

    summary = pd.DataFrame(rows)
    summary_knn = summary[summary["method"].str.contains("knn", case=False)].sort_values(
        "f1", ascending=False
    )
    summary_logreg = summary[
        summary["method"].str.contains("logistic_regression", case=False)
    ].sort_values("f1", ascending=False)

    # Export LaTeX
    tex_knn = summary_knn.to_latex(
        index=False, escape=False, column_format="l" + "c" * len(METRICS)
    )
    tex_logreg = summary_logreg.to_latex(
        index=False, escape=False, column_format="l" + "c" * len(METRICS)
    )

    (REPORT_DIR / "summary_knn.tex").write_text(tex_knn)
    (REPORT_DIR / "summary_linear.tex").write_text(tex_logreg)

    print("✅ Summary tables saved:")
    print(" - summary_knn.tex")
    print(" - summary_linear.tex")


if __name__ == "__main__":
    main()
