import argparse
import os
import sys
from typing import Tuple

import pandas as pd
from google.protobuf import text_format
import tensorflow_data_validation as tfdv

from util import add_extra_rows


ADULT_COLUMNS = [
    'age',
    'workclass',
    'fnlwgt',
    'education',
    'education-num',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'capital-gain',
    'capital-loss',
    'hours-per-week',
    'native-country',
    'label'
]

NUMERIC_COLUMNS = [
    'age',
    'fnlwgt',
    'education-num',
    'capital-gain',
    'capital-loss',
    'hours-per-week'
]

CATEGORICAL_COLUMNS = [
    'workclass',
    'education',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'native-country',
    'label'
]


def read_adult_dataset(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find dataset at: {csv_path}")

    df = pd.read_csv(
        csv_path,
        header=None,
        names=ADULT_COLUMNS,
        na_values=['?'],
        skipinitialspace=True
    )

    return df


def train_eval_split(df: pd.DataFrame, eval_fraction: float, random_state: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    eval_df = df.sample(frac=eval_fraction, random_state=random_state)
    train_df = df.drop(eval_df.index)
    return train_df.reset_index(drop=True), eval_df.reset_index(drop=True)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_proto_text(message, filepath: str) -> None:
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(text_format.MessageToString(message))


def enforce_dataframe_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    # Coerce numeric columns
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    # Cast categoricals to string dtype for consistency
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype('string')
    return df


def run_tfdv_pipeline(
    data_path: str,
    output_dir: str,
    add_eval_anomalies: bool,
    eval_fraction: float,
    random_state: int,
    write_facets_html: bool
) -> None:
    print(f"Reading dataset from: {data_path}")
    df = read_adult_dataset(data_path)
    print(f"Loaded {len(df)} rows with {len(df.columns)} columns")

    print("Splitting into train/eval DataFrames")
    train_df, eval_df = train_eval_split(df, eval_fraction=eval_fraction, random_state=random_state)
    print(f"Train rows: {len(train_df)} | Eval rows: {len(eval_df)}")

    if add_eval_anomalies:
        print("Injecting synthetic anomalies into eval set (via util.add_extra_rows)")
        eval_df = add_extra_rows(eval_df)
        print(f"Eval rows after injection: {len(eval_df)}")

    # Ensure stable dtypes before passing to TFDV/pyarrow
    print("Normalizing column dtypes for train/eval DataFrames")
    train_df = enforce_dataframe_dtypes(train_df)
    eval_df = enforce_dataframe_dtypes(eval_df)

    ensure_dir(output_dir)

    print("Generating TFDV statistics for train set")
    train_stats = tfdv.generate_statistics_from_dataframe(train_df)
    print("Generating TFDV statistics for eval set")
    eval_stats = tfdv.generate_statistics_from_dataframe(eval_df)

    print("Inferring schema from train statistics")
    schema = tfdv.infer_schema(statistics=train_stats)

    print("Validating eval statistics against schema")
    anomalies = tfdv.validate_statistics(statistics=eval_stats, schema=schema)

    # Save artifacts as protobuf text for easy inspection
    train_stats_path = os.path.join(output_dir, 'train_stats.pbtxt')
    eval_stats_path = os.path.join(output_dir, 'eval_stats.pbtxt')
    schema_path = os.path.join(output_dir, 'schema.pbtxt')
    anomalies_path = os.path.join(output_dir, 'anomalies.pbtxt')

    print(f"Writing artifacts to: {output_dir}")
    write_proto_text(train_stats, train_stats_path)
    write_proto_text(eval_stats, eval_stats_path)
    write_proto_text(schema, schema_path)
    write_proto_text(anomalies, anomalies_path)

    print("\nSummary:")
    print(f"- Train stats: {train_stats_path}")
    print(f"- Eval stats:  {eval_stats_path}")
    print(f"- Schema:      {schema_path}")
    print(f"- Anomalies:   {anomalies_path}")

    print("\nDetected anomalies (if any):")
    # Pretty-print anomalies to stdout as protobuf text
    print(text_format.MessageToString(anomalies))

    # Optionally generate a Facets Overview HTML for side-by-side visualization
    if write_facets_html:
        try:
            from tensorflow_data_validation.utils.display_util import get_statistics_html
            facets_html = get_statistics_html(
                lhs_statistics=train_stats,
                rhs_statistics=eval_stats,
                lhs_name='train',
                rhs_name='eval'
            )
            facets_path = os.path.join(output_dir, 'stats_overview.html')
            with open(facets_path, 'w', encoding='utf-8') as f:
                f.write(facets_html)
            print(f"- Facets HTML: {facets_path}")
        except Exception as e:
            print(f"Warning: Failed to generate Facets HTML: {e}")

    # Serving removed per project requirements.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TensorFlow Data Validation on the Adult dataset.")
    parser.add_argument(
        '--data_path',
        type=str,
        default=os.path.join('data', 'adult.data'),
        help='Path to the Adult CSV dataset.'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs',
        help='Directory to write TFDV artifacts (stats/schema/anomalies).'
    )
    parser.add_argument(
        '--no_anomaly_injection',
        action='store_true',
        help='Disable injecting synthetic anomalies into the eval set.'
    )
    parser.add_argument(
        '--eval_fraction',
        type=float,
        default=0.2,
        help='Fraction of data to use for evaluation set.'
    )
    parser.add_argument(
        '--random_state',
        type=int,
        default=42,
        help='Random seed for splitting.'
    )
    parser.add_argument(
        '--write_facets_html',
        action='store_true',
        help='Write a Facets Overview HTML (stats_overview.html) into the output directory.'
    )
    # Serving options removed per project requirements.
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_tfdv_pipeline(
        data_path=args.data_path,
        output_dir=args.output_dir,
        add_eval_anomalies=not args.no_anomaly_injection,
        eval_fraction=args.eval_fraction,
        random_state=args.random_state,
        write_facets_html=args.write_facets_html
    )


