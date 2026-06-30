"""Self-contained end-to-end example (no network, no real model needed).

Simulates a company with an old embedding model migrating to a new one:
  1. generate old vectors + a synthetic "new model"
  2. run the full migration (sample -> train -> gate -> transform)
  3. print the confidence report and where the migrated vectors landed

Run:  python examples/quickstart.py
"""

import sys
import tempfile
from pathlib import Path

# Allow running as a plain script from anywhere: put the project root on the path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from app.core.pipeline import run_migration
from app.models.migration import MigrationConfig
from app.stores import FileStore, load_vectors, save_vectors
from app.stores.synthetic import make_related_spaces


def main() -> None:
    work = Path(tempfile.mkdtemp(prefix="emf-quickstart-"))
    n, d_old, d_new = 5000, 64, 96

    # 1. The company's stored old vectors, and a synthetic "new model".
    old, new = make_related_spaces(n, d_old, d_new, noise=0.02, seed=0)
    ids = [f"doc-{i}" for i in range(n)]
    save_vectors(work / "old.npz", ids, old)
    id_to_new = {ids[i]: new[i] for i in range(n)}
    new_model = lambda texts: np.array([id_to_new[t] for t in texts], dtype=np.float32)

    # 2. Run the migration: re-embed only 3% of the corpus.
    store = FileStore(work / "old.npz")
    config = MigrationConfig(
        sample_fraction=0.03,
        k=10,
        confidence_threshold=0.90,
        output_dir=str(work),
        artifacts_dir=str(work / "artifacts"),
        output_collection="corpus_v2",
    )
    result = run_migration(store, new_model, texts=None, config=config)

    # 3. Report.
    print(result.report.to_text())
    print()
    if result.transformed:
        out_ids, out_vecs = load_vectors(result.output_path)
        print(f"Migrated {len(out_ids)} vectors (dim {out_vecs.shape[1]}) -> {result.output_path}")
        print(f"Mapper artifact : {result.mapper_path}")
    else:
        print(f"Migration skipped: {result.skipped_reason}")


if __name__ == "__main__":
    main()
