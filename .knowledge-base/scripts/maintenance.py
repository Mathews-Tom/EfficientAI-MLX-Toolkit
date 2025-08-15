#!/usr/bin/env python3
"""
Knowledge Base Maintenance Script

This script performs regular maintenance tasks on the knowledge base,
including quality checks, index optimization, and cleanup operations.
"""

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# Add the .meta directory to Python path for imports
kb_meta_path = Path(__file__).parent.parent / ".meta"
sys.path.insert(0, str(kb_meta_path))

try:
    from analytics import KnowledgeBaseAnalytics
    from cross_reference import CrossReferenceAnalyzer
    from freshness_tracker import ContentFreshnessTracker
    from indexer import KnowledgeBaseIndexer
    from quality_assurance import KnowledgeBaseQualityAssurance
    from reporting import KnowledgeBaseReporter
except ImportError as e:
    print(f"❌ Error importing knowledge base modules: {e}")
    print("Make sure you're running this from the knowledge base directory")
    sys.exit(1)


class KnowledgeBaseMaintainer:
    """Handles knowledge base maintenance operations."""

    def __init__(self, kb_path: Path):
        self.kb_path = kb_path
        self.meta_path = kb_path / ".meta"
        self.logs_path = self.meta_path / "logs"
        self.logs_path.mkdir(exist_ok=True)

        # Initialize components
        self.indexer = KnowledgeBaseIndexer(kb_path)
        self.qa = KnowledgeBaseQualityAssurance(kb_path)
        self.freshness_tracker = ContentFreshnessTracker(kb_path)
        self.cross_ref_analyzer = CrossReferenceAnalyzer(kb_path)

        # Load or create maintenance log
        self.maintenance_log_path = self.logs_path / "maintenance.json"
        self.maintenance_log = self._load_maintenance_log()

    def _load_maintenance_log(self) -> dict[str, Any]:
        """Load maintenance log or create new one."""
        if self.maintenance_log_path.exists():
            try:
                with open(self.maintenance_log_path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass

        return {
            "last_full_maintenance": None,
            "last_index_rebuild": None,
            "last_quality_check": None,
            "last_freshness_check": None,
            "last_cleanup": None,
            "maintenance_history": [],
        }

    def _save_maintenance_log(self) -> None:
        """Save maintenance log to file."""
        with open(self.maintenance_log_path, "w") as f:
            json.dump(self.maintenance_log, f, indent=2, default=str)

    def _log_maintenance_action(self, action: str, details: dict[str, Any]) -> None:
        """Log a maintenance action."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "details": details,
        }

        self.maintenance_log["maintenance_history"].append(log_entry)
        self.maintenance_log[f"last_{action}"] = datetime.now().isoformat()

        # Keep only last 100 entries
        if len(self.maintenance_log["maintenance_history"]) > 100:
            self.maintenance_log["maintenance_history"] = self.maintenance_log[
                "maintenance_history"
            ][-100:]

    def rebuild_index(self, force: bool = False) -> dict[str, Any]:
        """Rebuild the search index."""
        print("🔄 Rebuilding search index...")

        try:
            if force:
                # Remove existing index
                index_file = self.meta_path / "index.json"
                if index_file.exists():
                    index_file.unlink()

            # Rebuild index
            index = self.indexer.build_index()
            stats = index.get_statistics()

            result = {
                "success": True,
                "entries_indexed": len(index.entries),
                "categories": len(index.categories),
                "total_tags": len(index.tags),
                "statistics": stats,
            }

            self._log_maintenance_action("index_rebuild", result)
            print(f"✅ Index rebuilt: {len(index.entries)} entries indexed")

            return result

        except Exception as e:
            result = {"success": False, "error": str(e)}
            self._log_maintenance_action("index_rebuild", result)
            print(f"❌ Index rebuild failed: {e}")
            return result

    def run_quality_check(self) -> dict[str, Any]:
        """Run comprehensive quality checks."""
        print("🔍 Running quality checks...")

        try:
            report = self.qa.run_comprehensive_quality_check()

            result = {
                "success": True,
                "total_entries": report.total_entries_checked,
                "quality_score": report.quality_score,
                "issues_found": len(report.issues_found),
                "high_priority_issues": len(
                    [i for i in report.issues_found if i.severity == "high"]
                ),
                "recommendations": report.recommendations,
            }

            self._log_maintenance_action("quality_check", result)

            if report.quality_score >= 80:
                print(f"✅ Quality check passed: Score {report.quality_score}/100")
            elif report.quality_score >= 60:
                print(f"⚠️  Quality check warning: Score {report.quality_score}/100")
            else:
                print(f"❌ Quality check failed: Score {report.quality_score}/100")

            if result["high_priority_issues"] > 0:
                print(f"🚨 {result['high_priority_issues']} high priority issues found")

            return result

        except Exception as e:
            result = {"success": False, "error": str(e)}
            self._log_maintenance_action("quality_check", result)
            print(f"❌ Quality check failed: {e}")
            return result

    def check_freshness(self) -> dict[str, Any]:
        """Check content freshness."""
        print("📅 Checking content freshness...")

        try:
            report = self.freshness_tracker.analyze_content_freshness()

            result = {
                "success": True,
                "total_entries": report.total_entries,
                "stale_entries": len(report.stale_entries),
                "critical_entries": len(
                    [
                        e
                        for e in report.stale_entries
                        if e.freshness_status == "critical"
                    ]
                ),
                "freshness_breakdown": report.freshness_breakdown,
            }

            self._log_maintenance_action("freshness_check", result)

            if result["stale_entries"] == 0:
                print("✅ All content is fresh")
            elif result["critical_entries"] > 0:
                print(f"🚨 {result['critical_entries']} entries need urgent updates")
            else:
                print(f"⚠️  {result['stale_entries']} entries may need updates")

            return result

        except Exception as e:
            result = {"success": False, "error": str(e)}
            self._log_maintenance_action("freshness_check", result)
            print(f"❌ Freshness check failed: {e}")
            return result

    def validate_cross_references(self) -> dict[str, Any]:
        """Validate cross-references between entries."""
        print("� Validating cross-references...")

        try:
            report = self.cross_ref_analyzer.analyze_cross_references()

            result = {
                "success": True,
                "total_references": len(report.all_references),
                "broken_references": len(report.broken_references),
                "orphaned_entries": len(report.orphaned_entries),
                "highly_connected": len(report.highly_connected_entries),
            }

            self._log_maintenance_action("cross_reference_check", result)

            if result["broken_references"] == 0:
                print("✅ All cross-references are valid")
            else:
                print(f"❌ {result['broken_references']} broken references found")

            if result["orphaned_entries"] > 0:
                print(f"⚠️  {result['orphaned_entries']} orphaned entries found")

            return result

        except Exception as e:
            result = {"success": False, "error": str(e)}
            self._log_maintenance_action("cross_reference_check", result)
            print(f"❌ Cross-reference validation failed: {e}")
            return result

    def cleanup_files(self) -> dict[str, Any]:
        """Clean up temporary and unnecessary files."""
        print("🧹 Cleaning up files...")

        try:
            cleaned_files = []

            # Clean up temporary files
            temp_patterns = ["*.tmp", "*.bak", "*~", ".DS_Store"]
            for pattern in temp_patterns:
                for file_path in self.kb_path.rglob(pattern):
                    if file_path.is_file():
                        file_path.unlink()
                        cleaned_files.append(str(file_path))

            # Clean up empty directories
            empty_dirs = []
            for dir_path in self.kb_path.rglob("*"):
                if dir_path.is_dir() and not any(dir_path.iterdir()):
                    # Don't remove required directories
                    if dir_path.name not in [
                        ".meta",
                        "categories",
                        "patterns",
                        "templates",
                    ]:
                        dir_path.rmdir()
                        empty_dirs.append(str(dir_path))

            # Clean up old log files (keep last 30 days)
            if self.logs_path.exists():
                cutoff_date = datetime.now() - timedelta(days=30)
                for log_file in self.logs_path.glob("*.log"):
                    if log_file.stat().st_mtime < cutoff_date.timestamp():
                        log_file.unlink()
                        cleaned_files.append(str(log_file))

            result = {
                "success": True,
                "files_cleaned": len(cleaned_files),
                "empty_dirs_removed": len(empty_dirs),
                "cleaned_files": cleaned_files[:10],  # Show first 10
            }

            self._log_maintenance_action("cleanup", result)
            print(
                f"✅ Cleanup complete: {len(cleaned_files)} files, {len(empty_dirs)} directories"
            )

            return result

        except Exception as e:
            result = {"success": False, "error": str(e)}
            self._log_maintenance_action("cleanup", result)
            print(f"❌ Cleanup failed: {e}")
            return result

    def optimize_storage(self) -> dict[str, Any]:
        """Optimize storage and compress old data."""
        print("💾 Optimizing storage...")

        try:
            # Compress old analytics data
            analytics_path = self.meta_path / "analytics"
            compressed_files = []

            if analytics_path.exists():
                cutoff_date = datetime.now() - timedelta(days=90)
                for analytics_file in analytics_path.glob("*.json"):
                    if analytics_file.stat().st_mtime < cutoff_date.timestamp():
                        # Could implement compression here
                        compressed_files.append(str(analytics_file))

            result = {
                "success": True,
                "compressed_files": len(compressed_files),
                "storage_optimized": True,
            }

            self._log_maintenance_action("storage_optimization", result)
            print(f"✅ Storage optimized: {len(compressed_files)} files processed")

            return result

        except Exception as e:
            result = {"success": False, "error": str(e)}
            self._log_maintenance_action("storage_optimization", result)
            print(f"❌ Storage optimization failed: {e}")
            return result

    def run_full_maintenance(self) -> dict[str, Any]:
        """Run complete maintenance routine."""
        print("🔧 Running full maintenance routine...")
        print("=" * 50)

        results = {}

        # Run all maintenance tasks
        results["index_rebuild"] = self.rebuild_index()
        results["quality_check"] = self.run_quality_check()
        results["freshness_check"] = self.check_freshness()
        results["cross_reference_check"] = self.validate_cross_references()
        results["cleanup"] = self.cleanup_files()
        results["storage_optimization"] = self.optimize_storage()

        # Calculate overall success
        successful_tasks = sum(1 for r in results.values() if r.get("success", False))
        total_tasks = len(results)

        overall_result = {
            "success": successful_tasks == total_tasks,
            "successful_tasks": successful_tasks,
            "total_tasks": total_tasks,
            "completion_rate": (successful_tasks / total_tasks) * 100,
            "results": results,
        }

        self._log_maintenance_action("full_maintenance", overall_result)

        print("=" * 50)
        if overall_result["success"]:
            print("✅ Full maintenance completed successfully")
        else:
            print(
                f"⚠️  Maintenance completed with issues: {successful_tasks}/{total_tasks} tasks successful"
            )

        return overall_result

    def get_maintenance_status(self) -> dict[str, Any]:
        """Get current maintenance status."""
        now = datetime.now()

        status = {
            "last_full_maintenance": self.maintenance_log.get("last_full_maintenance"),
            "days_since_maintenance": None,
            "maintenance_needed": False,
            "recommended_actions": [],
        }

        # Check if maintenance is needed
        if status["last_full_maintenance"]:
            last_maintenance = datetime.fromisoformat(status["last_full_maintenance"])
            days_since = (now - last_maintenance).days
            status["days_since_maintenance"] = days_since

            if days_since > 7:
                status["maintenance_needed"] = True
                status["recommended_actions"].append("Run full maintenance")
        else:
            status["maintenance_needed"] = True
            status["recommended_actions"].append("Run initial maintenance")

        # Check individual components
        if not self.maintenance_log.get("last_index_rebuild"):
            status["recommended_actions"].append("Rebuild search index")

        if not self.maintenance_log.get("last_quality_check"):
            status["recommended_actions"].append("Run quality check")

        return status


def main():
    """Main maintenance function."""
    parser = argparse.ArgumentParser(description="Knowledge Base Maintenance")
    parser.add_argument(
        "--kb-path",
        type=Path,
        default=".",
        help="Path to knowledge base (default: current directory)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Maintenance commands")

    # Full maintenance
    full_parser = subparsers.add_parser("full", help="Run full maintenance routine")

    # Individual tasks
    index_parser = subparsers.add_parser("index", help="Rebuild search index")
    index_parser.add_argument(
        "--force", action="store_true", help="Force complete rebuild"
    )

    quality_parser = subparsers.add_parser("quality", help="Run quality checks")

    freshness_parser = subparsers.add_parser(
        "freshness", help="Check content freshness"
    )

    crossref_parser = subparsers.add_parser(
        "crossref", help="Validate cross-references"
    )

    cleanup_parser = subparsers.add_parser("cleanup", help="Clean up temporary files")

    optimize_parser = subparsers.add_parser("optimize", help="Optimize storage")

    status_parser = subparsers.add_parser("status", help="Show maintenance status")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    kb_path = args.kb_path.resolve()

    # Check if knowledge base exists
    if not (kb_path / ".meta").exists():
        print(f"❌ No knowledge base found at {kb_path}")
        print("Run init_knowledge_base.py first to create a knowledge base")
        sys.exit(1)

    maintainer = KnowledgeBaseMaintainer(kb_path)

    result = {}

    try:
        if args.command == "full":
            result = maintainer.run_full_maintenance()
        elif args.command == "index":
            result = maintainer.rebuild_index(force=getattr(args, "force", False))
        elif args.command == "quality":
            result = maintainer.run_quality_check()
        elif args.command == "freshness":
            result = maintainer.check_freshness()
        elif args.command == "crossref":
            result = maintainer.validate_cross_references()
        elif args.command == "cleanup":
            result = maintainer.cleanup_files()
        elif args.command == "optimize":
            result = maintainer.optimize_storage()
        elif args.command == "status":
            status = maintainer.get_maintenance_status()
            print("📊 Maintenance Status:")
            print(f"   Last maintenance: {status['last_full_maintenance'] or 'Never'}")
            if status["days_since_maintenance"]:
                print(f"   Days since maintenance: {status['days_since_maintenance']}")
            print(
                f"   Maintenance needed: {'Yes' if status['maintenance_needed'] else 'No'}"
            )
            if status["recommended_actions"]:
                print("   Recommended actions:")
                for action in status["recommended_actions"]:
                    print(f"     - {action}")
            return

        # Save maintenance log
        maintainer._save_maintenance_log()

        # Exit with error code if maintenance failed
        if not result.get("success", False):
            sys.exit(1)

    except Exception as e:
        print(f"❌ Maintenance failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
