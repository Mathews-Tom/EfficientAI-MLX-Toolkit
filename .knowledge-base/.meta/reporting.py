"""
Knowledge Base Analytics Reporting and Insights

This module provides advanced reporting and insights generation for knowledge
base analytics, including trend analysis, predictive insights, and automated
report generation.
"""

import json
import logging
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from analytics import KnowledgeBaseAnalytics
from models import KnowledgeBaseIndex

logger = logging.getLogger(__name__)


@dataclass
class TrendAnalysis:
    """
    Trend analysis results for knowledge base usage.
    """

    metric_name: str
    time_period: tuple[datetime, datetime]
    trend_direction: str  # increasing, decreasing, stable
    growth_rate: float
    data_points: list[tuple[datetime, float]]
    insights: list[str]


@dataclass
class InsightReport:
    """
    Comprehensive insights report with actionable recommendations.
    """

    generated_at: datetime
    analysis_period: tuple[datetime, datetime]
    key_insights: list[str]
    trend_analyses: list[TrendAnalysis]
    predictions: list[str]
    recommendations: list[str]
    risk_factors: list[str]
    opportunities: list[str]


class KnowledgeBaseReporter:
    """
    Advanced reporting and insights system for knowledge base analytics.
    """

    def __init__(self, analytics: KnowledgeBaseAnalytics):
        self.analytics = analytics
        self.kb_path = analytics.kb_path

        # Configuration
        self.min_data_points = 7
        self.trend_threshold = 0.1  # 10% change threshold

        # Statistics
        self.stats = {
            "reports_generated": 0,
            "trends_analyzed": 0,
            "insights_generated": 0,
            "visualizations_created": 0,
        }

    def generate_trend_analysis(
        self, metric: str, days: int = 30, granularity: str = "daily"
    ) -> TrendAnalysis | None:
        """
        Generate trend analysis for a specific metric.

        Args:
            metric: Metric to analyze (views, searches, creates, updates)
            days: Number of days to analyze
            granularity: Time granularity (daily, weekly)

        Returns:
            TrendAnalysis object with trend information
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            # Get time series data
            data_points = self._get_time_series_data(
                metric, start_date, end_date, granularity
            )

            if len(data_points) < self.min_data_points:
                logger.warning(
                    f"Insufficient data points for trend analysis: {len(data_points)}"
                )
                return None

            # Calculate trend
            trend_direction, growth_rate = self._calculate_trend(data_points)

            # Generate insights
            insights = self._generate_trend_insights(
                metric, trend_direction, growth_rate, data_points
            )

            self.stats["trends_analyzed"] += 1

            return TrendAnalysis(
                metric_name=metric,
                time_period=(start_date, end_date),
                trend_direction=trend_direction,
                growth_rate=growth_rate,
                data_points=data_points,
                insights=insights,
            )

        except Exception as e:
            logger.error(f"Failed to generate trend analysis for {metric}: {e}")
            return None

    def _get_time_series_data(
        self, metric: str, start_date: datetime, end_date: datetime, granularity: str
    ) -> list[tuple[datetime, float]]:
        """Get time series data for a specific metric."""
        data_points = []

        try:
            with sqlite3.connect(self.analytics.db_path) as conn:
                cursor = conn.cursor()

                # Map metric to event type
                event_type_map = {
                    "views": "view",
                    "searches": "search",
                    "creates": "create",
                    "updates": "update",
                }

                event_type = event_type_map.get(metric, metric)

                if granularity == "daily":
                    date_trunc = "date(timestamp)"
                else:  # weekly
                    date_trunc = "strftime('%Y-%W', timestamp)"

                cursor.execute(
                    f"""
                    SELECT {date_trunc} as date_group, COUNT(*) as count
                    FROM events
                    WHERE event_type = ?
                    AND timestamp >= ?
                    AND timestamp <= ?
                    GROUP BY date_group
                    ORDER BY date_group
                    """,
                    (event_type, start_date.isoformat(), end_date.isoformat()),
                )

                for date_str, count in cursor.fetchall():
                    try:
                        if granularity == "weekly":
                            year, week = date_str.split("-")
                            date_obj = datetime.strptime(
                                f"{year}-W{week}-1", "%Y-W%W-%w"
                            )
                        else:
                            date_obj = datetime.strptime(date_str, "%Y-%m-%d")

                        data_points.append((date_obj, float(count)))
                    except ValueError:
                        continue

        except Exception as e:
            logger.error(f"Failed to get time series data: {e}")

        return data_points

    def _calculate_trend(
        self, data_points: list[tuple[datetime, float]]
    ) -> tuple[str, float]:
        """Calculate trend direction and growth rate."""
        if len(data_points) < 2:
            return "stable", 0.0

        values = [point[1] for point in data_points]
        n = len(values)

        # Simple linear regression for trend
        x_values = list(range(n))
        x_mean = sum(x_values) / n
        y_mean = sum(values) / n

        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, values))
        denominator = sum((x - x_mean) ** 2 for x in x_values)

        if denominator == 0:
            return "stable", 0.0

        slope = numerator / denominator

        # Calculate growth rate as percentage
        first_value = values[0] if values[0] != 0 else 1
        last_value = values[-1]
        growth_rate = ((last_value - first_value) / first_value) * 100

        # Determine trend direction
        if abs(growth_rate) < self.trend_threshold * 100:
            trend_direction = "stable"
        elif slope > 0:
            trend_direction = "increasing"
        else:
            trend_direction = "decreasing"

        return trend_direction, growth_rate

    def _generate_trend_insights(
        self,
        metric: str,
        trend_direction: str,
        growth_rate: float,
        data_points: list[tuple[datetime, float]],
    ) -> list[str]:
        """Generate insights based on trend analysis."""
        insights = []

        # Basic trend insight
        if trend_direction == "increasing":
            insights.append(
                f"{metric.title()} are trending upward with {growth_rate:.1f}% growth"
            )
        elif trend_direction == "decreasing":
            insights.append(
                f"{metric.title()} are declining with {growth_rate:.1f}% decrease"
            )
        else:
            insights.append(f"{metric.title()} remain stable with minimal change")

        # Volatility analysis
        values = [point[1] for point in data_points]
        if len(values) > 1:
            mean_value = sum(values) / len(values)
            variance = sum((v - mean_value) ** 2 for v in values) / len(values)
            std_dev = variance**0.5

            if std_dev > mean_value * 0.5:
                insights.append(f"High volatility detected in {metric}")
            elif std_dev < mean_value * 0.1:
                insights.append(f"{metric.title()} show consistent patterns")

        return insights

    def generate_comprehensive_insights(
        self, index: KnowledgeBaseIndex, days: int = 30
    ) -> InsightReport:
        """
        Generate comprehensive insights report.

        Args:
            index: Knowledge base index
            days: Number of days to analyze

        Returns:
            InsightReport with comprehensive analysis
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            # Generate trend analyses
            trend_analyses = []
            for metric in ["views", "searches", "creates", "updates"]:
                trend = self.generate_trend_analysis(metric, days=days)
                if trend:
                    trend_analyses.append(trend)

            # Generate insights
            key_insights = self._generate_key_insights(index, days)
            predictions = self._generate_predictions(trend_analyses, index)
            recommendations = self._generate_recommendations(
                trend_analyses, index, days
            )
            risk_factors = self._identify_risk_factors(trend_analyses, index)
            opportunities = self._identify_opportunities(trend_analyses, index, days)

            self.stats["reports_generated"] += 1
            self.stats["insights_generated"] += len(key_insights)

            return InsightReport(
                generated_at=datetime.now(),
                analysis_period=(start_date, end_date),
                key_insights=key_insights,
                trend_analyses=trend_analyses,
                predictions=predictions,
                recommendations=recommendations,
                risk_factors=risk_factors,
                opportunities=opportunities,
            )

        except Exception as e:
            logger.error(f"Failed to generate comprehensive insights: {e}")
            return InsightReport(
                generated_at=datetime.now(),
                analysis_period=(start_date, end_date),
                key_insights=[],
                trend_analyses=[],
                predictions=[],
                recommendations=[],
                risk_factors=[],
                opportunities=[],
            )

    def _generate_key_insights(self, index: KnowledgeBaseIndex, days: int) -> list[str]:
        """Generate key insights from analytics data."""
        insights = []

        try:
            report = self.analytics.generate_analytics_report(index, days=days)

            # Content insights
            if report.popular_entries:
                top_entry = report.popular_entries[0]
                insights.append(
                    f"'{top_entry[0]}' is your top performer with {top_entry[1]} views"
                )

            # Search insights
            if report.popular_searches:
                search_volume = sum(count for _, count in report.popular_searches)
                insights.append(f"Total search volume: {search_volume} queries")

            # Category insights
            if report.category_usage:
                total_views = sum(report.category_usage.values())
                top_category = max(report.category_usage.items(), key=lambda x: x[1])
                dominance = (top_category[1] / total_views) * 100
                insights.append(
                    f"'{top_category[0]}' dominates with {dominance:.1f}% of views"
                )

            # Knowledge gaps
            if report.knowledge_gaps:
                insights.append(
                    f"Identified {len(report.knowledge_gaps)} knowledge gaps"
                )

        except Exception as e:
            logger.error(f"Failed to generate key insights: {e}")

        return insights

    def _generate_predictions(
        self, trend_analyses: list[TrendAnalysis], index: KnowledgeBaseIndex
    ) -> list[str]:
        """Generate predictive insights."""
        predictions = []

        for trend in trend_analyses:
            if trend.trend_direction == "increasing" and trend.growth_rate > 20:
                predictions.append(
                    f"{trend.metric_name.title()} likely to continue growing"
                )
            elif trend.trend_direction == "decreasing" and trend.growth_rate < -20:
                predictions.append(f"{trend.metric_name.title()} decline may continue")

        return predictions

    def _generate_recommendations(
        self, trend_analyses: list[TrendAnalysis], index: KnowledgeBaseIndex, days: int
    ) -> list[str]:
        """Generate recommendations."""
        recommendations = []

        for trend in trend_analyses:
            if trend.metric_name == "views" and trend.trend_direction == "decreasing":
                recommendations.append("Consider refreshing popular content")
            elif (
                trend.metric_name == "searches"
                and trend.trend_direction == "increasing"
            ):
                recommendations.append(
                    "High search activity - consider expanding popular topics"
                )

        return recommendations

    def _identify_risk_factors(
        self, trend_analyses: list[TrendAnalysis], index: KnowledgeBaseIndex
    ) -> list[str]:
        """Identify risk factors."""
        risks = []

        for trend in trend_analyses:
            if trend.metric_name == "views" and trend.growth_rate < -30:
                risks.append("Significant decline in content views")
            elif trend.metric_name == "creates" and trend.growth_rate < -50:
                risks.append("Sharp drop in content creation")

        return risks

    def _identify_opportunities(
        self, trend_analyses: list[TrendAnalysis], index: KnowledgeBaseIndex, days: int
    ) -> list[str]:
        """Identify opportunities."""
        opportunities = []

        for trend in trend_analyses:
            if trend.metric_name == "searches" and trend.growth_rate > 50:
                opportunities.append("High search growth indicates content opportunity")

        return opportunities

    def export_insights_report(
        self,
        insights_report: InsightReport,
        output_path: Path,
        format: str = "markdown",
    ) -> bool:
        """
        Export insights report to file.

        Args:
            insights_report: Report to export
            output_path: Output file path
            format: Export format (markdown, json)

        Returns:
            True if successful
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if format.lower() == "markdown":
                self._export_markdown_report(insights_report, output_path)
            elif format.lower() == "json":
                self._export_json_report(insights_report, output_path)
            else:
                raise ValueError(f"Unsupported format: {format}")

            logger.info(f"Exported insights report to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export insights report: {e}")
            return False

    def _export_markdown_report(self, report: InsightReport, output_path: Path) -> None:
        """Export report as Markdown."""
        content = [
            "# Knowledge Base Insights Report",
            "",
            f"**Generated:** {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Period:** {report.analysis_period[0].strftime('%Y-%m-%d')} to {report.analysis_period[1].strftime('%Y-%m-%d')}",
            "",
            "## Key Insights",
            "",
        ]

        for insight in report.key_insights:
            content.append(f"- {insight}")

        content.extend(["", "## Trend Analysis", ""])

        for trend in report.trend_analyses:
            content.extend(
                [
                    f"### {trend.metric_name.title()}",
                    f"- **Trend:** {trend.trend_direction.title()}",
                    f"- **Growth:** {trend.growth_rate:.1f}%",
                    "",
                ]
            )

        if report.recommendations:
            content.extend(["## Recommendations", ""])
            for rec in report.recommendations:
                content.append(f"- {rec}")

        output_path.write_text("\n".join(content), encoding="utf-8")

    def _export_json_report(self, report: InsightReport, output_path: Path) -> None:
        """Export report as JSON."""
        report_dict = asdict(report)
        report_dict["generated_at"] = report.generated_at.isoformat()
        report_dict["analysis_period"] = [
            report.analysis_period[0].isoformat(),
            report.analysis_period[1].isoformat(),
        ]

        with open(output_path, "w") as f:
            json.dump(report_dict, f, indent=2, default=str)

    def get_reporting_stats(self) -> dict[str, Any]:
        """Get reporting statistics."""
        return {
            **self.stats,
            "analytics_db_path": str(self.analytics.db_path),
            "min_data_points": self.min_data_points,
            "trend_threshold": self.trend_threshold,
        }


def create_reporter(analytics: KnowledgeBaseAnalytics) -> KnowledgeBaseReporter:
    """
    Factory function to create a knowledge base reporter.

    Args:
        analytics: Analytics system instance

    Returns:
        Configured reporter instance
    """
    return KnowledgeBaseReporter(analytics)


def main():
    """Command-line interface for reporting system."""
    import argparse

    parser = argparse.ArgumentParser(description="Knowledge Base Reporting")
    parser.add_argument("kb_path", help="Path to knowledge base directory")
    parser.add_argument(
        "--insights", action="store_true", help="Generate insights report"
    )
    parser.add_argument("--days", type=int, default=30, help="Days to analyze")
    parser.add_argument("--output", help="Output file path")
    parser.add_argument("--format", default="markdown", help="Output format")

    args = parser.parse_args()

    try:
        from analytics import create_analytics_tracker
        from indexer import KnowledgeBaseIndexer

        # Load analytics and index
        analytics = create_analytics_tracker(Path(args.kb_path))
        indexer = KnowledgeBaseIndexer(Path(args.kb_path))
        index = indexer.load_index_from_file() or indexer.build_index()

        # Create reporter
        reporter = create_reporter(analytics)

        if args.insights:
            print("Generating insights report...")
            insights = reporter.generate_comprehensive_insights(index, days=args.days)

            if args.output:
                reporter.export_insights_report(
                    insights, Path(args.output), format=args.format
                )
                print(f"Report exported to {args.output}")
            else:
                print(f"Key Insights: {len(insights.key_insights)}")
                print(f"Trends Analyzed: {len(insights.trend_analyses)}")
                print(f"Recommendations: {len(insights.recommendations)}")

        else:
            stats = reporter.get_reporting_stats()
            print(f"Reports generated: {stats['reports_generated']}")
            print(f"Trends analyzed: {stats['trends_analyzed']}")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
