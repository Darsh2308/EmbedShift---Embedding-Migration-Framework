"""Command-line interface for the Embedding Migration Framework.

    emf serve [--host --port]                     run the HTTP API
    emf migrate --source old.npz --texts t.jsonl  run a full migration

The new model is provided as an embedder:
  --embedder NAME            a registered embedder (default: hashing)
  --embedder-module mod:fn   import a callable (texts -> vectors) from a module

Run without installing via:  python -m app.cli ...
"""

from __future__ import annotations

import argparse
import importlib
import sys

from app import __version__
from app.core.pipeline import Embedder, run_migration
from app.embedders import get_embedder, list_embedders
from app.models.migration import MigrationConfig
from app.stores import make_store


def _resolve_embedder(name: str, module_spec: str | None) -> Embedder:
    if module_spec:
        mod_name, sep, fn_name = module_spec.partition(":")
        if not sep:
            raise SystemExit("--embedder-module must be 'module:function'")
        module = importlib.import_module(mod_name)
        return getattr(module, fn_name)
    return get_embedder(name)


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("app.main:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def _cmd_migrate(args: argparse.Namespace) -> int:
    if args.backend == "file" and not args.source:
        raise SystemExit("file backend requires --source")
    if args.backend in ("qdrant", "pinecone") and not args.collection:
        raise SystemExit(f"{args.backend} backend requires --collection (the source index/collection)")

    store = make_store(
        args.backend,
        source_path=args.source,
        collection=args.collection,
        location=args.location,
        url=args.url,
        api_key=args.api_key,
        host=args.host,
        namespace=args.namespace,
        cloud=args.cloud,
        region=args.region,
    )
    dest_store = None
    if args.backend != "file":
        dest_store = store  # write the new collection back into the same DB

    try:
        embed = _resolve_embedder(args.embedder, args.embedder_module)
    except (ImportError, AttributeError, KeyError) as exc:
        raise SystemExit(f"could not load embedder: {exc}")

    config = MigrationConfig(
        mapper_kind=args.mapper_kind,
        sample_fraction=args.sample_fraction,
        sample_size=args.sample_size,
        k=args.k,
        confidence_threshold=args.threshold,
        output_collection=args.output_collection,
        output_dir=args.output_dir,
        artifacts_dir=args.artifacts_dir,
        force=args.force,
    )

    result = run_migration(store, embed, args.texts, config, dest_store=dest_store)
    print(result.to_text())
    return 0 if result.transformed else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="emf", description="Embedding Migration Framework")
    parser.add_argument("--version", action="version", version=f"emf {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the HTTP API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")
    serve.set_defaults(func=_cmd_serve)

    mig = sub.add_parser("migrate", help="run a full migration")
    mig.add_argument("--backend", default="file", choices=("file", "qdrant", "pinecone"))
    mig.add_argument("--source", help="old vectors file (file backend)")
    mig.add_argument("--collection", help="source collection/index (qdrant/pinecone backend)")
    mig.add_argument("--location", help="qdrant location (':memory:' or path)")
    mig.add_argument("--url", help="qdrant server url")
    mig.add_argument("--api-key", dest="api_key")
    mig.add_argument("--host", help="pinecone index host")
    mig.add_argument("--namespace", default="", help="pinecone namespace")
    mig.add_argument("--cloud", default="aws", help="pinecone serverless cloud (new index)")
    mig.add_argument("--region", default="us-east-1", help="pinecone serverless region (new index)")
    mig.add_argument("--texts", help="id->text jsonl for the sample")
    mig.add_argument("--embedder", default="hashing", help=f"registered embedder {list_embedders()}")
    mig.add_argument("--embedder-module", help="import a custom embedder as 'module:function'")
    mig.add_argument("--mapper-kind", default="auto", choices=("auto", "linear", "mlp"))
    mig.add_argument("--sample-fraction", type=float, default=0.03)
    mig.add_argument("--sample-size", type=int, default=None)
    mig.add_argument("--k", type=int, default=10)
    mig.add_argument("--threshold", type=float, default=0.90)
    mig.add_argument("--output-collection", default="corpus_v2")
    mig.add_argument("--output-dir", default="data")
    mig.add_argument("--artifacts-dir", default="artifacts")
    mig.add_argument("--force", action="store_true", help="transform even if the gate fails")
    mig.set_defaults(func=_cmd_migrate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
