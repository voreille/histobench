import pandas as pd


def main(csv_path, output_path):
    # Read the CSV file
    df = pd.read_csv(csv_path)
    df_by_patient = df.groupby("patient").first().reset_index()
    df_by_patient.drop(columns=["path"], inplace=True)
    df_by_patient.to_csv(output_path, index=False)


if __name__ == "__main__":
    csv_path = "/home/valentin/workspaces/histobench/data/tcga-ut/train_val_test_split.csv"
    otutput_path = (
        "/home/valentin/workspaces/histobench/data/tcga-ut/train_val_test_split_by_patient.csv"
    )
    main(csv_path, otutput_path)
