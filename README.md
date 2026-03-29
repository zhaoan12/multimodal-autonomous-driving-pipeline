# Multimodal Autonomous Driving Pipeline

Research-style Python project for building and validating a multimodal autonomous driving workflow that combines LiDAR geometry, camera detections, labeled scene annotations, and large-scale QA generation.

## What this repository does

This repository implements a compact but runnable end-to-end pipeline for:

1. projecting LiDAR points into camera image coordinates using camera intrinsics and LiDAR-to-camera extrinsics,
2. relating 2D image detections to real-world coordinates using projected LiDAR support points,
3. validating grounded detections against labeled driving-scene annotations,
4. generating scene-grounded question-answer pairs from labeled scenes using prompt templates,
5. filtering inconsistent or unusable QA generations,
6. demonstrating the full workflow on synthetic sample data with reproducible scripts.

The included sample data is synthetic and intentionally small. It exists to make the repository executable without external downloads. Replace it with real labeled autonomous driving scenes for research usage.

## Repository structure

```text
.
├── configs/
│   ├── calibration.example.json
│   ├── qa_generation.example.json
│   └── validation.example.json
├── data/
│   └── sample/
│       ├── scene_0001.json
│       └── scene_manifest.json
├── scripts/
│   ├── generate_qa.py
│   └── validate_projection.py
├── src/
│   └── mmdrive_pipeline/
│       ├── data/
│       ├── geometry/
│       ├── mapping/
│       ├── qa/
│       ├── validation/
│       └── pipeline.py
└── tests/
```

## Core modules

- `mmdrive_pipeline.geometry`
  - calibration loading and rigid transforms
  - LiDAR-to-camera projection
- `mmdrive_pipeline.mapping`
  - bounding-box to 3D grounding using projected LiDAR support points
- `mmdrive_pipeline.validation`
  - comparison of grounded detections against labeled object positions
- `mmdrive_pipeline.qa`
  - prompt templates, structured QA records, mock generation backend, and output filtering
- `mmdrive_pipeline.pipeline`
  - high-level entrypoints used by both scripts and library consumers

## Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e .
```

This project intentionally uses only the Python standard library at runtime. No external dataset or model dependency is required for the sample workflow.

## Data format

Each labeled scene JSON contains:

- `camera_intrinsics`
- `lidar_to_camera`
- `point_cloud`
- `objects`
- per-object optional `detection`

Each object stores a world or LiDAR-frame position label plus an associated image-space detection when available.

## Example workflow

Run projection validation:

```bash
python scripts/validate_projection.py
```

Run QA generation:

```bash
python scripts/generate_qa.py
```

Run tests:

```bash
python -m unittest discover -s tests -v
```

Generated outputs are written to:

- `data/generated/validation_report.json`
- `data/generated/qa_pairs.json`

## Library usage

Validation:

```python
from mmdrive_pipeline import run_validation_pipeline

reports = run_validation_pipeline(
    "data/sample/scene_manifest.json",
    distance_tolerance_m=2.0,
    min_support_points=2,
)
```

QA generation:

```python
from mmdrive_pipeline import run_qa_generation_pipeline

records = run_qa_generation_pipeline(
    "data/sample/scene_manifest.json",
    num_pairs=3,
    filter_output=True,
)
```

## Method summary

### 1. Projection

LiDAR points are transformed into the camera frame with the provided extrinsics and then projected with a pinhole camera model:

- `X_camera = R * X_lidar + t`
- `u = fx * X / Z + cx`
- `v = fy * Y / Z + cy`

Points behind the camera or outside the image bounds are filtered out.

### 2. Detection-to-world grounding

For each image detection, the pipeline finds projected LiDAR points that fall inside the detection bounding box. If enough support points exist, it uses the median LiDAR and camera coordinates as a robust representative object location.

### 3. Validation

Grounded detections are matched back to labeled scene objects and scored using Euclidean distance against annotated positions. The sample validation report includes:

- match rate
- mean distance error
- per-object support counts

### 4. QA generation

The repository renders a scene-grounded prompt from annotation metadata and object lists. A mock backend is included for reproducible offline execution. In a real deployment, replace that backend with an API-backed LLM client.

### 5. Filtering

Generated QA pairs are filtered for:

- empty questions or answers
- duplicate questions
- outputs not grounded in scene object labels or IDs

## Replacing placeholders with real research assets

To adapt this repository for a real project:

1. replace `data/sample` with real labeled scenes,
2. extend the dataset loader for the exact schema used by your dataset,
3. plug in calibrated camera and LiDAR parameters from your platform,
4. replace the mock QA backend with a real LLM provider,
5. strengthen filtering with additional semantic or model-based checks,
6. add dataset-scale evaluation and experiment tracking.

## Limitations

- The bundled sample scene is synthetic and tiny.
- The QA backend is deterministic and mock-only.
- Object matching currently assumes one detection per labeled object in the sample schema.
- Occlusion reasoning and multi-camera fusion are out of scope for this baseline repository.

## Development notes

- Scripts are thin wrappers around package entrypoints.
- Tests use `unittest` so they run in minimal environments.
- JSON configs are provided for runnable examples.
- YAML loading remains supported only when `PyYAML` is installed separately.

## License

MIT

