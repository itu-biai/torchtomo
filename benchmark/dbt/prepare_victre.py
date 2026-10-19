"""Download one TCIA VICTRE phantom and convert its DBT projections to line integrals.

The conversion is the BIAI Lab DBT 2026v1 pipeline in ``torchtomo.victre``. Each
bin factor writes ``input_bin{factor}.npz`` with padded line integrals, the
valid-ray mask, per-view air levels, and VICTRE's own reconstruction binned onto
the same grid. That reconstruction is an arbitrary-scale FBP, a comparison and
not a ground truth.
"""

import argparse
import io
import json
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from torchtomo.victre import (
    DENSITIES,
    DENSITY_SLICES,
    convert_projections,
    density_from_comments,
    read_projections,
    read_reference,
    read_stack,
    reference_on_grid,
)

TCIA = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
DEFAULT_PATIENT = "694607065"


def fetch(url: str, attempts: int = 4) -> bytes:
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=900) as response:
                return response.read()
        except OSError:
            if attempt == attempts - 1:
                raise
            time.sleep(30 * (attempt + 1))
    raise RuntimeError(url)


def patient_series(patient: str, cache: Path) -> dict[str, dict]:
    path = cache / f"{patient}-series.json"
    if not path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        path.write_bytes(fetch(f"{TCIA}/getSeries?Collection=VICTRE&PatientID={patient}&format=json"))
    return {item["SeriesDescription"]: item for item in json.loads(path.read_text())}


def download_series(uid: str, target: Path) -> Path:
    marker = target / ".complete"
    if not marker.exists():
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(fetch(f"{TCIA}/getImage?SeriesInstanceUID={uid}"))) as archive:
            archive.extractall(target)
        marker.touch()
    return target


def prepare(patient: str, cache: Path, output: Path, bin_factors: list[int], density: str | None) -> None:
    series = patient_series(patient, cache)
    missing = {"Projection images", "DBT slices"} - set(series)
    if missing:
        raise FileNotFoundError(f"VICTRE patient {patient} has no {sorted(missing)} series")
    projection_dir = download_series(series["Projection images"]["SeriesInstanceUID"], cache / patient / "projections")
    reference_dir = download_series(series["DBT slices"]["SeriesInstanceUID"], cache / patient / "reference")
    first = read_stack(projection_dir)[0]
    comments = str(getattr(first, "PatientComments", ""))
    density = density or density_from_comments(comments)
    views = read_projections(projection_dir)
    reference = read_reference(reference_dir)
    if reference.shape[0] != DENSITY_SLICES[density]:
        raise ValueError(f"{density} reconstruction has {reference.shape[0]} slices, not {DENSITY_SLICES[density]}")
    output.mkdir(parents=True, exist_ok=True)
    for bin_factor in bin_factors:
        started = time.perf_counter()
        projections, valid, air = convert_projections(views, bin_factor)
        metadata = {
            "patient": patient,
            "density": density,
            "patient_comments": comments,
            "bin_factor": bin_factor,
            "native_projection_shape": list(views.shape),
            "native_reference_shape": list(reference.shape),
            "projection_series": series["Projection images"]["SeriesInstanceUID"],
            "reference_series": series["DBT slices"]["SeriesInstanceUID"],
            "air_dn": air.tolist(),
            "conversion_seconds": time.perf_counter() - started,
            "citation": "Badano et al., VICTRE, The Cancer Imaging Archive, DOI 10.7937/TCIA.2019.ho23nxaw",
        }
        np.savez_compressed(
            output / f"input_bin{bin_factor}.npz",
            projections=projections,
            valid=valid,
            reference=reference_on_grid(reference, bin_factor),
            metadata=np.array(json.dumps(metadata)),
        )
        print(json.dumps({key: metadata[key] for key in ("patient", "density", "bin_factor")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patient", default=DEFAULT_PATIENT)
    parser.add_argument("--density", choices=DENSITIES, help="override the density read from PatientComments")
    parser.add_argument("--cache", type=Path, default=Path("benchmark/dbt/cache"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bin-factor", type=int, nargs="+", default=[4, 8])
    args = parser.parse_args()
    prepare(args.patient, args.cache, args.output, args.bin_factor, args.density)
