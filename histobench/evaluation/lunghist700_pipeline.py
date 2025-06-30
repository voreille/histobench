from pathlib import Path

import click
import yaml

from histobench.evaluation.compute_embeddings_lunghist700 import compute_embeddings_on_lunghist700
from histobench.evaluation.evaluate_lunghist700 import evaluate_lunghist700_embeddings

project_dir = Path(__file__).parents[2].resolve()


@click.command()
@click.option(
    "--config-path",
    default="configs/LungHist700/config.yaml",
    help="Path to the configuration file.",
)
def main(config_path):
    """LungHist700 evaluation pipeline - computes embeddings and evaluates models based on config file"""

    config = yaml.safe_load((project_dir / config_path).read_text())
    runtime_config = config.get("runtime", {})
    eval_config = config.get("eval", {})
    report_dir = Path(runtime_config.get("report_dir", "reports/lunghist700/")).resolve()

    # Process each model configuration
    for model_config in config.get("models", []):
        model_name = model_config["name"]
        print(f"\n{'=' * 60}")
        print(f"Processing model: {model_name}")
        print(f"{'=' * 60}")

        # Generate embeddings path if not specified
        if "embeddings_path" not in model_config:
            input_basename = Path(
                model_config.get("input_dir", "data/LungHist700/LungHist700_10x")
            ).name
            aggregation = model_config.get("aggregation", "whole_roi")
            label = model_config.get("label", model_name)
            model_config["embeddings_path"] = (
                f"data/embeddings/lunghist700/{label}_{input_basename}_{aggregation}.h5"
            )

        embedding_path = Path(model_config["embeddings_path"]).resolve()

        # Check if we should skip embedding computation
        if embedding_path.exists() and not runtime_config.get("force", False):
            print(
                f"Embeddings already exist at {model_config['embeddings_path']}, skipping computation."
            )
        else:
            print(f"Computing embeddings for {model_name}...")
            compute_embeddings_on_lunghist700(
                model=model_config["name"],
                model_weights_path=model_config.get("weights_path"),
                input_dir=model_config.get("input_dir", "data/LungHist700/LungHist700_10x"),
                gpu_id=runtime_config.get("gpu_id", 0),
                aggregation=model_config.get("aggregation", "whole_roi"),
                tile_size=model_config.get("tile_size", 224),
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
            input_basename = Path(
                model_config.get("input_dir", "data/LungHist700/LungHist700_10x")
            ).name
            aggregation = model_config.get("aggregation", "whole_roi")
            label = model_config.get("label", model_name)
            n_splits = eval_config.get("n_splits", 5)

            report_path = (
                report_dir
                / f"{label}_{input_basename}_{aggregation}_KNNn_{knn_n_neighbors}_cv_report.csv"
            )

            evaluate_lunghist700_embeddings(
                csv_metadata=eval_config.get("csv_metadata", "data/LungHist700/metadata.csv"),
                embeddings_path=model_config["embeddings_path"],
                knn_n_neighbors=knn_n_neighbors,
                report_path=report_path,
                n_splits=n_splits,
                label_column=eval_config.get("label_column", "superclass"),
            )
            print(f"Evaluation report saved to {report_path}")

    print(f"\n{'=' * 60}")
    print("Pipeline completed successfully!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
