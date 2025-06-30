from pathlib import Path

import click
import yaml

from histobench.evaluation.compute_embeddings_tcga_ut import compute_embeddings_on_tcga_ut
from histobench.evaluation.evaluate_tcag_ut import evaluate_tcga_ut_embeddings

project_dir = Path(__file__).parents[2].resolve()


@click.command()
@click.option(
    "--config-path",
    default="configs/tcga-ut/config.yaml",
    help="Path to the configuration file.",
)
def main(config_path):
    """LungHist700 evaluation pipeline - computes embeddings and evaluates models based on config file"""

    config = yaml.safe_load((project_dir / config_path).read_text())
    runtime_config = config.get("runtime", {})
    eval_config = config.get("eval", {})

    # Process each model configuration
    for model_config in config.get("models", []):
        model_name = model_config["name"]
        magnification_key = model_config.get("magnification_key", 5)
        print(f"\n{'=' * 60}")
        print(f"Processing model: {model_name}")
        print(f"{'=' * 60}")

        # Generate embeddings path if not specified
        if "embeddings_path" not in model_config:
            label = model_config.get("label", model_name)
            model_config["embeddings_path"] = (
                f"data/embeddings/tcga_ut/{label}_mk{magnification_key}.h5"
            )

        embedding_path = Path(model_config["embeddings_path"]).resolve()

        # Check if we should skip embedding computation
        if embedding_path.exists() and not runtime_config.get("force", False):
            print(
                f"Embeddings already exist at {model_config['embeddings_path']}, skipping computation."
            )
        else:
            print(f"Computing embeddings for {model_name}...")
            compute_embeddings_on_tcga_ut(
                model=model_config["name"],
                model_weights_path=model_config.get("weights_path"),
                magnification_key=model_config.get("magnification_key", 5),
                tcga_ut_dir=model_config.get("tcgat_ut_dir", "data/tcga-ut"),
                gpu_id=runtime_config.get("gpu_id", 0),
                batch_size=runtime_config.get("batch_size", 32),
                num_workers=runtime_config.get("num_workers", 0),
                embeddings_path=model_config["embeddings_path"],
            )
            print(f"Embeddings computed and saved to {model_config['embeddings_path']}")

        # Evaluate embeddings for different KNN neighbors
        knn_neighbors_list = eval_config.get("knn_n_neighbors", [5, 20])
        if not isinstance(knn_neighbors_list, list):
            knn_neighbors_list = [knn_neighbors_list]

        for knn_n_neighbors in knn_neighbors_list:
            print(f"\nEvaluating {model_name} with KNN n_neighbors={knn_n_neighbors}...")

            # Generate report path if not specified
            label = model_config.get("label", model_name)

            report_path = f"reports/tcgat_ut/{label}_mk{magnification_key}_KNNn_{knn_n_neighbors}_cv_report.csv"

            evaluate_tcga_ut_embeddings(
                csv_split_path=eval_config.get("csv_split_path", "data/tcga-ut/train_val_test_split.csv"),
                embeddings_path=model_config["embeddings_path"],
                knn_n_neighbors=knn_n_neighbors,
                report_path=report_path,
            )
            print(f"Evaluation report saved to {report_path}")

    print(f"\n{'=' * 60}")
    print("Pipeline completed successfully!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
