"""Load serialized ProblemData without pickle, network access, or forecasting."""
import json
from pathlib import Path
import numpy as np
from .model import ProblemData

def load_frozen_instance(directory: str | Path) -> ProblemData:
    directory = Path(directory)
    metadata = json.loads((directory / 'metadata.json').read_text(encoding='utf-8'))
    with np.load(directory / 'arrays.npz', allow_pickle=False) as arrays:
        fields = {name: arrays[name].copy() for name in arrays.files}
    return ProblemData(**metadata['problem_fields'], **fields)
