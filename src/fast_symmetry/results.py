"""Public plane/result data and lossless serialization of detector diagnostics."""

from dataclasses import asdict, dataclass, field
from typing import Any

from .method import FLOOR, METHOD, METHOD_ID


@dataclass(frozen=True)
class ProductionPrediction:
    """One accepted world plane, with original/refined verification diagnostics."""

    raw_rank: int
    native_id: int
    normal_before_refinement: list[float]
    normal: list[float]
    offset: float
    height: float
    p90: float
    p99: float
    field_value: float
    diagnostics: dict = field(default_factory=dict)

    def __getattr__(self, name):
        diagnostics = object.__getattribute__(self, "diagnostics")
        if name in diagnostics:
            return diagnostics[name]
        raise AttributeError(name)

    def to_dict(self):
        """Flatten public attributes and raw diagnostics into the existing JSON schema."""
        values = asdict(self)
        diagnostics = values.pop("diagnostics")
        return {**values, **diagnostics}


@dataclass(frozen=True)
class ProductionDetectionResult:
    """Accepted planes and the complete unchanged detection record."""

    method_id: str
    seed: int
    center: list[float]
    diagonal: float
    pre_d1_native_ids: list[int]
    post_d1_native_ids: list[int]
    accepted_native_ids: list[int]
    predictions: list[ProductionPrediction]
    graph: dict[str, Any]

    diagnostics: dict = field(default_factory=dict)

    def __getattr__(self, name):
        diagnostics = object.__getattribute__(self, "diagnostics")
        if name in diagnostics:
            return diagnostics[name]
        raise AttributeError(name)

    def to_dict(self):
        """Flatten public attributes and raw diagnostics into the existing JSON schema."""
        return {
            **self.diagnostics,
            "method_id": self.method_id,
            "seed": self.seed,
            "center": self.center,
            "diagonal": self.diagonal,
            "pre_d1_native_ids": self.pre_d1_native_ids,
            "post_d1_native_ids": self.post_d1_native_ids,
            "accepted_native_ids": self.accepted_native_ids,
            "predictions": [p.to_dict() for p in self.predictions],
            "graph": self.graph,
        }


def from_record(record):
    """Wrap the unchanged raw inference record in the public result dataclasses."""
    predictions = []
    for p in record["predictions"]:
        predictions.append(
            ProductionPrediction(
                raw_rank=p["raw_rank"],
                native_id=p["native_id"],
                normal_before_refinement=p["normal_original"],
                normal=p["normal"],
                offset=p["offset"],
                height=p["height"],
                p90=p["p90"],
                p99=p["p99"],
                field_value=p["optimization"]["initial_value"],
                diagnostics=p,
            )
        )
    return ProductionDetectionResult(
        method_id=METHOD_ID,
        seed=record["seed"],
        center=record["center"],
        diagonal=record["diagonal"],
        pre_d1_native_ids=[
            p["native_id"]
            for p in record["raw_top256"]
            if p["height"] >= FLOOR * record["maximum_extracted_peak_height"]
        ][:64],
        post_d1_native_ids=[p["native_id"] for p in record["candidates"]],
        accepted_native_ids=[p["native_id"] for p in record["predictions"]],
        predictions=predictions,
        graph={
            "kind": "projective_icosphere",
            "subdivisions": METHOD["icosphere_level"],
            "directions": METHOD["quotient_directions"],
            "full_sphere_directions": 2 * METHOD["quotient_directions"],
            "exact_plateaus": True,
            "projective": True,
        },
        diagnostics={k: v for k, v in record.items() if k != "predictions"},
    )
