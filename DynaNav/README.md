# DynaNav

DynaNav is the Isaac Sim benchmark environment used by TIC-VLA. It contains the benchmark runner, benchmark configs, TIC-VLA behavior scripts, and local scene assets. 

## Setup

Install Isaac Sim and the DynaNav Python dependencies from the repository root first:

```bash
source .env.testing
"${ISAAC_SIM_PYTHON}" -m pip install -e .
"${ISAAC_SIM_PYTHON}" -m pip install -r requirements-test.txt
```

Edit `.env.testing` before running so it points to your Isaac Sim install, model, and checkpoint.

## Run

Run the smoke benchmark from the repository root:

```bash
DynaNav/run_benchmark.sh DynaNav/configs/benchmark_example.yaml
# Full benchmark set:
DynaNav/run_benchmark.sh DynaNav/configs/benchmark_full.yaml
```

## Data Collection

Data collection uses Isaac Sim Python and the configs in `configs/data_collection/`:

```bash
source .env.testing
cd DynaNav

"${ISAAC_SIM_PYTHON}" collect_data.py -c configs/data_collection/office.yaml
"${ISAAC_SIM_PYTHON}" collect_data.py -c configs/data_collection/outdoor.yaml
"${ISAAC_SIM_PYTHON}" collect_data.py -c configs/data_collection/hospital.yaml
"${ISAAC_SIM_PYTHON}" collect_data.py -c configs/data_collection/warehouse.yaml
```

Each config writes data under `${TICVLA_DYNANAV_ROOT}/output/<scene>`. Use `--save_usd` to save the generated scene, `--debug_print` for verbose Isaac logs, and `--sensor_placment_file path/to/cameras.json` when using explicit camera placements.

To convert collected camera JSON poses into a trajectory CSV:

```bash
python camera_pose_trajectory.py output/office --out_csv output/office/trajectory.csv
```
