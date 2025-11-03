## TFDV Lab - Containerized

This project converts the original `TFDV_Lab1.ipynb` into a runnable Python script (`main.py`) and provides a Docker image to execute TensorFlow Data Validation (TFDV) over the Adult census dataset.

### What the script does

- **Loads** the Adult dataset from `data/adult.data` with canonical column names.
- **Splits** into train/eval DataFrames (default 80/20).
- **Optionally injects anomalies** into the eval set using `util.add_extra_rows()` (enabled by default).
- **Generates statistics** for train and eval using TFDV.
- **Infers a schema** from the train statistics.
- **Validates** eval statistics against the inferred schema to detect anomalies.
- **Saves artifacts** as protobuf text files under `outputs/`:
  - `train_stats.pbtxt`
  - `eval_stats.pbtxt`
  - `schema.pbtxt`
  - `anomalies.pbtxt`

Note: Visualization cells from the notebook are omitted in this headless script. Artifacts are saved as human‑readable protobuf text files for inspection and versioning.

### Requirements (local, without Docker)

Install Python 3.10 and run:

```bash
pip install -r requirements.txt
```

Run the script locally:

```bash
python main.py --data_path data/adult.data --output_dir outputs
```

Optional flags:

- `--no_anomaly_injection` to skip adding synthetic anomalies to the eval set
- `--eval_fraction <float>` (default `0.2`)
- `--random_state <int>` (default `42`)
- `--write_facets_html` to generate `outputs/stats_overview.html` (Facets)

### Build and run with Docker

From the project root (`LAB4/`):

```bash
docker build -t tfdv-lab .
docker run --rm -v %cd%/outputs:/app/outputs tfdv-lab
```

On macOS/Linux, replace the `-v` path with `$(pwd)/outputs:/app/outputs`.

Customize arguments (examples):

```bash
docker run --rm -v %cd%/outputs:/app/outputs \
  tfdv-lab python main.py --data_path data/adult.data --output_dir outputs --no_anomaly_injection

docker run --rm -v %cd%/outputs:/app/outputs \
  tfdv-lab python main.py --eval_fraction 0.3 --random_state 123
 
# Generate Facets HTML (no serving). Open the file from your host after run:
docker run --rm -v %cd%/outputs:/app/outputs \
  tfdv-lab python main.py --write_facets_html
```

### Repository layout

- `main.py`: CLI script that runs the TFDV workflow end-to-end.
- `util.py`: Helper to inject synthetic anomalies into eval data.
- `data/adult.data`: Input dataset (UCI Adult).
- `outputs/`: Generated artifacts (created at runtime).
- `requirements.txt`: Python dependencies.
- `Dockerfile`: Container spec to run the pipeline.
- `TFDV_Lab1.ipynb`: Original notebook reference.

### Outputs

All artifacts are protobuf messages written in text format for readability:

- `train_stats.pbtxt`: DatasetFeatureStatisticsList from train DataFrame
- `eval_stats.pbtxt`: DatasetFeatureStatisticsList from eval DataFrame
- `schema.pbtxt`: Inferred schema from train statistics
- `anomalies.pbtxt`: Validation anomalies comparing eval stats to schema

Open any of these in a text editor or diff tool to review and track changes.


