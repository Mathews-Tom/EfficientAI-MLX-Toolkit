"""Example: Project Workspace Management Integration

Demonstrates the complete workflow of using WorkspaceManager with MLOpsClient
for managing multiple projects in the toolkit.
"""

from pathlib import Path

from mlops.client import MLOpsClient
from mlops.workspace import WorkspaceManager


def example_basic_workspace_usage():
    """Example 1: Basic workspace management"""
    print("\n=== Example 1: Basic Workspace Usage ===\n")

    mgr = WorkspaceManager()

    # Create workspaces for multiple projects
    projects = ["lora-finetuning-mlx", "model-compression-mlx", "coreml-style-transfer"]

    for project in projects:
        workspace = mgr.create_workspace(
            project_name=project,
            metadata={"description": f"{project} workspace"},
        )
        print(f"Created workspace: {workspace.project_name}")
        print(f"  Root path: {workspace.root_path}")
        print(f"  MLFlow path: {workspace.mlflow_path}")
        print(f"  DVC path: {workspace.dvc_path}")

    # List all workspaces
    print("\nAll workspaces:")
    for ws in mgr.list_workspaces():
        print(f"  - {ws.project_name}")


def example_mlops_client_integration():
    """Example 2: MLOpsClient automatically uses WorkspaceManager"""
    print("\n=== Example 2: MLOpsClient Integration ===\n")

    # Create client (automatically creates workspace)
    client = MLOpsClient.from_project("lora-finetuning-mlx")

    print(f"Project: {client.project_name}")
    print(f"Workspace path: {client.workspace_path}")

    if client.workspace:
        print(f"MLFlow experiment ID: {client.workspace.mlflow_experiment_id}")
        print(f"DVC remote path: {client.workspace.dvc_remote_path}")
        print(f"BentoML tag prefix: {client.workspace.bentoml_tag_prefix}")

    # Get comprehensive status
    status = client.get_status()
    if "workspace" in status:
        print("\nWorkspace info available in status:")
        print(f"  Created: {status['workspace']['created_at']}")
        print(f"  Updated: {status['workspace']['updated_at']}")


def example_workspace_status_monitoring():
    """Example 3: Monitor workspace status and statistics"""
    print("\n=== Example 3: Workspace Status Monitoring ===\n")

    mgr = WorkspaceManager()

    # Create some workspaces
    for project in ["project-1", "project-2", "project-3"]:
        mgr.get_or_create_workspace(project)

    # Get status for all workspaces
    all_status = mgr.get_all_workspaces_status()

    print("Workspace Status Summary:")
    for status in all_status:
        print(f"\n{status['project_name']}:")
        print(f"  Experiment ID: {status.get('mlflow_experiment_id', 'Not set')}")
        print(f"  Directories exist:")
        if "directory_stats" in status:
            for key, value in status["directory_stats"].items():
                if key.endswith("_exists"):
                    print(f"    {key}: {value}")


def example_workspace_lifecycle():
    """Example 4: Complete workspace lifecycle"""
    print("\n=== Example 4: Workspace Lifecycle ===\n")

    mgr = WorkspaceManager()

    # Create
    print("Creating workspace...")
    workspace = mgr.create_workspace(
        project_name="lifecycle-demo",
        mlflow_experiment_id="exp-001",
        metadata={"stage": "development"},
    )
    print(f"Created: {workspace.project_name}")

    # Update
    print("\nUpdating workspace metadata...")
    updated = mgr.update_workspace_metadata(
        "lifecycle-demo",
        mlflow_experiment_id="exp-002",
        metadata={"stage": "production"},
    )
    print(f"Updated experiment ID: {updated.mlflow_experiment_id}")
    print(f"Updated metadata: {updated.metadata}")

    # Get status
    print("\nGetting workspace status...")
    status = mgr.get_workspace_status("lifecycle-demo")
    print(f"Status: {status['project_name']} - {status['mlflow_experiment_id']}")

    # Delete (commented out for safety)
    # print("\nDeleting workspace...")
    # mgr.delete_workspace("lifecycle-demo", force=True)
    # print("Workspace deleted")


def example_multi_project_analytics():
    """Example 5: Cross-project analytics"""
    print("\n=== Example 5: Cross-Project Analytics ===\n")

    mgr = WorkspaceManager()

    # Create multiple project workspaces
    projects = {
        "lora-finetuning": {"exp_id": "exp-lora-123", "models": 5},
        "model-compression": {"exp_id": "exp-compress-456", "models": 3},
        "style-transfer": {"exp_id": "exp-style-789", "models": 2},
    }

    for project_name, config in projects.items():
        mgr.get_or_create_workspace(
            project_name,
            mlflow_experiment_id=config["exp_id"],
        )

    # Dashboard-style aggregation
    print("Project Dashboard:")
    workspaces = mgr.list_workspaces()

    print(f"\nTotal projects: {len(workspaces)}")
    for ws in workspaces:
        print(f"\n{ws.project_name}:")
        print(f"  Experiment: {ws.mlflow_experiment_id}")
        print(f"  Created: {ws.created_at.strftime('%Y-%m-%d %H:%M')}")
        print(f"  Paths configured:")
        print(f"    MLFlow: {ws.mlflow_path.exists()}")
        print(f"    DVC: {ws.dvc_path.exists()}")
        print(f"    Models: {ws.models_path.exists()}")


def main():
    """Run all examples"""
    print("\n" + "=" * 70)
    print("PROJECT WORKSPACE MANAGEMENT EXAMPLES")
    print("=" * 70)

    try:
        example_basic_workspace_usage()
        example_mlops_client_integration()
        example_workspace_status_monitoring()
        example_workspace_lifecycle()
        example_multi_project_analytics()

        print("\n" + "=" * 70)
        print("All examples completed successfully!")
        print("=" * 70 + "\n")

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
