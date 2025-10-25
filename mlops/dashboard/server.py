"""Dashboard Server

FastAPI server for unified MLOps dashboard with project filtering,
experiment tracking, model registry, monitoring, and alerts.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from mlops.dashboard.data_aggregator import DashboardDataAggregator

logger = logging.getLogger(__name__)


def create_dashboard_app(
    repo_root: str | Path | None = None,
    title: str = "EfficientAI MLOps Dashboard",
) -> FastAPI:
    """Create FastAPI dashboard application

    Args:
        repo_root: Repository root directory
        title: Dashboard title

    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title=title,
        description="Unified MLOps dashboard for EfficientAI-MLX-Toolkit",
        version="1.0.0",
    )

    # Get dashboard directory
    dashboard_dir = Path(__file__).parent

    # Mount static files
    static_dir = dashboard_dir / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
        logger.info("Mounted static files from: %s", static_dir)

    # Setup templates
    templates_dir = dashboard_dir / "templates"
    templates = Jinja2Templates(directory=str(templates_dir))
    logger.info("Loaded templates from: %s", templates_dir)

    # Initialize data aggregator
    aggregator = DashboardDataAggregator(repo_root=repo_root)

    # ==================== Dashboard Routes ====================

    @app.get("/", response_class=HTMLResponse)
    async def dashboard_home(request: Request) -> HTMLResponse:
        """Dashboard home page - overview of all projects"""
        try:
            data = aggregator.get_all_data()

            return templates.TemplateResponse(
                "overview.html",
                {
                    "request": request,
                    "title": title,
                    "workspaces": data.workspaces,
                    "stats": data.cross_project_stats,
                },
            )
        except Exception as e:
            logger.error("Failed to load dashboard home: %s", e)
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/data", response_class=JSONResponse)
    async def get_all_data() -> JSONResponse:
        """Get all dashboard data as JSON"""
        try:
            data = aggregator.get_all_data()
            return JSONResponse(content=data.to_dict())
        except Exception as e:
            logger.error("Failed to get dashboard data: %s", e)
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/overview", response_class=HTMLResponse)
    async def overview(request: Request) -> HTMLResponse:
        """Overview page - all projects summary"""
        try:
            data = aggregator.get_all_data()

            return templates.TemplateResponse(
                "overview.html",
                {
                    "request": request,
                    "title": "Overview",
                    "workspaces": data.workspaces,
                    "stats": data.cross_project_stats,
                    "alerts": data.alerts[:10],  # Show top 10 alerts
                },
            )
        except Exception as e:
            logger.error("Failed to load overview: %s", e)
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/experiments", response_class=HTMLResponse)
    async def experiments(request: Request, project: str | None = None) -> HTMLResponse:
        """Experiments page - MLFlow experiments by project"""
        try:
            data = aggregator.get_all_data()

            # Filter by project if specified
            if project:
                experiments_data = {project: data.experiments.get(project, [])}
            else:
                experiments_data = data.experiments

            return templates.TemplateResponse(
                "experiments.html",
                {
                    "request": request,
                    "title": "Experiments",
                    "experiments": experiments_data,
                    "workspaces": data.workspaces,
                    "selected_project": project,
                },
            )
        except Exception as e:
            logger.error("Failed to load experiments: %s", e)
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/experiments", response_class=JSONResponse)
    async def get_experiments(project: str | None = None) -> JSONResponse:
        """Get experiments data as JSON"""
        try:
            data = aggregator.get_all_data()

            if project:
                experiments_data = {project: data.experiments.get(project, [])}
            else:
                experiments_data = data.experiments

            return JSONResponse(content={"experiments": experiments_data})
        except Exception as e:
            logger.error("Failed to get experiments: %s", e)
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/models", response_class=HTMLResponse)
    async def models(request: Request, project: str | None = None) -> HTMLResponse:
        """Models page - unified model registry"""
        try:
            data = aggregator.get_all_data()

            # Filter by project if specified
            if project:
                models_data = [m for m in data.models if m["project_name"] == project]
            else:
                models_data = data.models

            return templates.TemplateResponse(
                "models.html",
                {
                    "request": request,
                    "title": "Models",
                    "models": models_data,
                    "workspaces": data.workspaces,
                    "selected_project": project,
                },
            )
        except Exception as e:
            logger.error("Failed to load models: %s", e)
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/models", response_class=JSONResponse)
    async def get_models(project: str | None = None) -> JSONResponse:
        """Get models data as JSON"""
        try:
            data = aggregator.get_all_data()

            if project:
                models_data = [m for m in data.models if m["project_name"] == project]
            else:
                models_data = data.models

            return JSONResponse(content={"models": models_data})
        except Exception as e:
            logger.error("Failed to get models: %s", e)
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/monitoring", response_class=HTMLResponse)
    async def monitoring(request: Request, project: str | None = None) -> HTMLResponse:
        """Monitoring page - Evidently drift/performance"""
        try:
            data = aggregator.get_all_data()

            # Filter by project if specified
            if project:
                monitoring_data = {project: data.monitoring_status.get(project, {})}
            else:
                monitoring_data = data.monitoring_status

            return templates.TemplateResponse(
                "monitoring.html",
                {
                    "request": request,
                    "title": "Monitoring",
                    "monitoring": monitoring_data,
                    "workspaces": data.workspaces,
                    "selected_project": project,
                },
            )
        except Exception as e:
            logger.error("Failed to load monitoring: %s", e)
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/monitoring", response_class=JSONResponse)
    async def get_monitoring(project: str | None = None) -> JSONResponse:
        """Get monitoring data as JSON"""
        try:
            data = aggregator.get_all_data()

            if project:
                monitoring_data = {project: data.monitoring_status.get(project, {})}
            else:
                monitoring_data = data.monitoring_status

            return JSONResponse(content={"monitoring": monitoring_data})
        except Exception as e:
            logger.error("Failed to get monitoring: %s", e)
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/alerts", response_class=HTMLResponse)
    async def alerts(request: Request, project: str | None = None) -> HTMLResponse:
        """Alerts page - alert management"""
        try:
            data = aggregator.get_all_data()

            # Filter by project if specified
            if project:
                alerts_data = [a for a in data.alerts if a["project_name"] == project]
            else:
                alerts_data = data.alerts

            return templates.TemplateResponse(
                "alerts.html",
                {
                    "request": request,
                    "title": "Alerts",
                    "alerts": alerts_data,
                    "workspaces": data.workspaces,
                    "selected_project": project,
                    "stats": data.cross_project_stats,
                },
            )
        except Exception as e:
            logger.error("Failed to load alerts: %s", e)
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/alerts", response_class=JSONResponse)
    async def get_alerts(project: str | None = None) -> JSONResponse:
        """Get alerts data as JSON"""
        try:
            data = aggregator.get_all_data()

            if project:
                alerts_data = [a for a in data.alerts if a["project_name"] == project]
            else:
                alerts_data = data.alerts

            return JSONResponse(content={"alerts": alerts_data})
        except Exception as e:
            logger.error("Failed to get alerts: %s", e)
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/hardware", response_class=HTMLResponse)
    async def hardware(request: Request, project: str | None = None) -> HTMLResponse:
        """Hardware page - Apple Silicon metrics"""
        try:
            data = aggregator.get_all_data()

            # Filter by project if specified
            if project:
                metrics_data = {project: data.apple_silicon_metrics.get(project, {})}
            else:
                metrics_data = data.apple_silicon_metrics

            return templates.TemplateResponse(
                "hardware.html",
                {
                    "request": request,
                    "title": "Hardware Metrics",
                    "metrics": metrics_data,
                    "workspaces": data.workspaces,
                    "selected_project": project,
                },
            )
        except Exception as e:
            logger.error("Failed to load hardware: %s", e)
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/hardware", response_class=JSONResponse)
    async def get_hardware(project: str | None = None) -> JSONResponse:
        """Get hardware metrics as JSON"""
        try:
            data = aggregator.get_all_data()

            if project:
                metrics_data = {project: data.apple_silicon_metrics.get(project, {})}
            else:
                metrics_data = data.apple_silicon_metrics

            return JSONResponse(content={"metrics": metrics_data})
        except Exception as e:
            logger.error("Failed to get hardware: %s", e)
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/project/{project_name}", response_class=HTMLResponse)
    async def project_detail(request: Request, project_name: str) -> HTMLResponse:
        """Project detail page"""
        try:
            project_data = aggregator.get_project_data(project_name)

            return templates.TemplateResponse(
                "project.html",
                {
                    "request": request,
                    "title": f"Project: {project_name}",
                    "project_name": project_name,
                    "workspace": project_data["workspace"],
                    "experiments": project_data["experiments"],
                    "models": project_data["models"],
                    "alerts": project_data["alerts"],
                    "monitoring_status": project_data["monitoring_status"],
                    "apple_silicon_metrics": project_data["apple_silicon_metrics"],
                },
            )
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            logger.error("Failed to load project detail: %s", e)
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/project/{project_name}", response_class=JSONResponse)
    async def get_project_data(project_name: str) -> JSONResponse:
        """Get project data as JSON"""
        try:
            project_data = aggregator.get_project_data(project_name)
            return JSONResponse(content=project_data)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            logger.error("Failed to get project data: %s", e)
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Health check endpoint"""
        return {"status": "ok"}

    logger.info("Dashboard application created")
    return app


class DashboardServer:
    """Dashboard server wrapper

    Provides a simple interface for running the dashboard server
    with configuration and lifecycle management.
    """

    def __init__(
        self,
        repo_root: str | Path | None = None,
        host: str = "0.0.0.0",
        port: int = 8000,
    ):
        """Initialize dashboard server

        Args:
            repo_root: Repository root directory
            host: Server host address
            port: Server port
        """
        self.repo_root = Path(repo_root) if repo_root else Path.cwd()
        self.host = host
        self.port = port
        self.app = create_dashboard_app(repo_root=self.repo_root)

        logger.info("Initialized DashboardServer (host=%s, port=%d)", host, port)

    def run(self, reload: bool = False) -> None:
        """Run the dashboard server

        Args:
            reload: Enable auto-reload for development

        Example:
            >>> server = DashboardServer()
            >>> server.run()
            # Access at http://localhost:8000
        """
        import uvicorn

        logger.info("Starting dashboard server at http://%s:%d", self.host, self.port)

        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            reload=reload,
            log_level="info",
        )


def main() -> None:
    """CLI entry point for dashboard server"""
    import argparse

    parser = argparse.ArgumentParser(description="EfficientAI MLOps Dashboard")
    parser.add_argument("--host", default="0.0.0.0", help="Server host address")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--repo-root", type=Path, help="Repository root directory")

    args = parser.parse_args()

    server = DashboardServer(
        repo_root=args.repo_root,
        host=args.host,
        port=args.port,
    )

    server.run(reload=args.reload)


if __name__ == "__main__":
    main()
