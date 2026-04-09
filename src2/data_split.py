import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split  # type: ignore

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

TRAIN_FILE = os.path.join(ROOT_DIR, "data", "train.csv")
OUTPUT_DIR = os.path.join(ROOT_DIR, "data2")


def create_validation_split(input_file=TRAIN_FILE, output_dir=OUTPUT_DIR, val_size=0.15, random_state=0):  # noqa: E501
    """
    Reads the original training data and
    splits it into a new train and validation set.
    Uses stratified splitting to maintain sentiment class distributions.
    """

    df = pd.read_csv(input_file, encoding='latin-1')

    # Drop any rows where the target 'sentiment' is missing
    df = df.dropna(subset=['sentiment'])

    # Perform a stratified split
    # 85% for training, 15% for validation
    train_df, val_df = train_test_split(
        df,
        test_size=val_size,
        random_state=random_state,
        stratify=df['sentiment']
    )

    # Define new file paths
    train_out_path = os.path.join(output_dir, "train.csv")
    val_out_path = os.path.join(output_dir, "val.csv")

    # Save the new splits
    train_df.to_csv(train_out_path, index=False, encoding='utf-8-sig')
    val_df.to_csv(val_out_path, index=False, encoding='utf-8-sig')

    print("-" * 30)
    print("Split Successful!")
    print(f"Original Data: {len(df)} rows")
    print(f"New Train Set: {len(train_df)} rows -> Saved to {train_out_path}")
    print(f"Validation Set: {len(val_df)} rows -> Saved to {val_out_path}")
    print("-" * 30)


if __name__ == "__main__":
    create_validation_split()
