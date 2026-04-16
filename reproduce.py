"""
Reproduction script for:
  labeling_session.labeling_schemas must have at most 16 elements

A ReviewApp can hold at most 16 label schemas. This script creates
17 label schemas on a single ReviewApp, triggering the limit on the
17th call to create_label_schema().

Usage:
    uv run python reproduce.py
    uv run python reproduce.py --clean          # delete all existing schemas first
    uv run python reproduce.py --dry-run
    uv run python reproduce.py --experiment /Shared/my-experiment
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from dotenv import load_dotenv

EXPERIMENT_NAME = "labeling-schema-maxfields-repro"
NUM_SCHEMAS = 17  # one more than the limit of 16


def setup_mlflow(experiment_name: str) -> str:
    import mlflow

    if not os.environ.get("DATABRICKS_HOST"):
        hostname = os.environ.get("DATABRICKS_SERVER_HOSTNAME", "")
        if hostname:
            os.environ["DATABRICKS_HOST"] = f"https://{hostname}"
        else:
            from databricks.sdk import WorkspaceClient

            w = WorkspaceClient()
            os.environ["DATABRICKS_HOST"] = w.config.host

    mlflow.set_tracking_uri("databricks")
    if not experiment_name.startswith("/"):
        experiment_name = f"/Shared/{experiment_name}"
    mlflow.set_experiment(experiment_name)
    return experiment_name


def get_review_app(experiment_id: str):
    from databricks.agents.review_app import get_review_app

    return get_review_app(experiment_id)


def clean_schemas(experiment_name: str) -> None:
    """Delete all existing label schemas from the ReviewApp."""
    import mlflow

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        print("No experiment found, nothing to clean.")
        return

    review_app = get_review_app(experiment.experiment_id)
    schemas = list(review_app.label_schemas)
    if not schemas:
        print("No existing label schemas to clean.")
        return

    print(f"Deleting {len(schemas)} existing label schemas...")
    for schema in schemas:
        for attempt in range(5):
            try:
                review_app.delete_label_schema(schema.name)
                print(f"  Deleted: {schema.name}")
                break
            except Exception as exc:
                if "429" in str(exc) or "Rate limit" in str(exc):
                    wait = 2 ** (attempt + 1)
                    print(f"  Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                else:
                    raise
        time.sleep(1)
    print("Clean done.\n")


def main(argv: list[str] | None = None) -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Reproduce the 16-schema limit on a ReviewApp"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--clean", action="store_true", help="Delete all existing schemas first")
    parser.add_argument("--experiment", default=EXPERIMENT_NAME)
    args = parser.parse_args(argv)

    schema_names = [f"repro_field_{i:02d}" for i in range(NUM_SCHEMAS)]
    print(f"Will create {NUM_SCHEMAS} label schemas on one ReviewApp (limit is 16)\n")

    if args.dry_run:
        for name in schema_names:
            print(f"  {name}")
        return 0

    from mlflow.genai.label_schemas import InputText, create_label_schema

    full_experiment_name = setup_mlflow(args.experiment)

    if args.clean:
        clean_schemas(full_experiment_name)

    for i, name in enumerate(schema_names):
        for attempt in range(5):
            try:
                create_label_schema(
                    name=name,
                    type="feedback",
                    title=f"Field {i}",
                    input=InputText(),
                    overwrite=True,
                    instruction=f"Feedback field {i}.",
                )
                print(f"  [{i+1:2d}/{NUM_SCHEMAS}] Created: {name}")
                break
            except Exception as exc:
                if "429" in str(exc) or "Rate limit" in str(exc):
                    wait = 2 ** (attempt + 1)
                    print(f"  Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"\n  ERROR on schema #{i+1} ({name}): {exc}")
                    return 1
        if i < len(schema_names) - 1:
            time.sleep(3)

    print(f"\nAll {NUM_SCHEMAS} schemas created.")

    # Step 2: Create a sample trace in the experiment
    import mlflow

    print("Creating a sample trace...")
    with mlflow.start_run():

        @mlflow.trace
        def sample_prediction(question: str) -> str:
            return f"This is a sample answer to: {question}"

        sample_prediction("What is the quality of this report?")

    print("Sample trace created.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
