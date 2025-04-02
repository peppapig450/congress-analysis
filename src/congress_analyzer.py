from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, NamedTuple, TypedDict

import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots

PROJECT_ROOT = Path(__file__).parent.parent

LOG_DIR = PROJECT_ROOT / "output" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(funcName)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "congress_analysis.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class TenureInfo(TypedDict):
    id: str
    tenure_years: float
    start_date: datetime
    end_date: datetime
    congresses: list[int]
    chamber: str
    num_terms: int


class CongressDate(NamedTuple):
    start_date: str
    end_date: str | None
    congress_num: int | None
    chamber: str | None


class CongressAnalyzer:
    def __init__(self, json_dir: str = "../data/extracted/BioguideProfiles") -> None:
        """Initialize the CongressAnalyzer with a directory path.

        Args:
            json_dir (str): Directory containing JSON files with congressional data.
        """
        self.json_dir = (Path(__file__).parent / json_dir).resolve()
        self.tenure_data: list[TenureInfo] = []
        self.df: None | pd.DataFrame = None

    def calculate_exact_years(self, start: datetime, end: datetime) -> float:
        """Calculate the exact years between two datetime objects, accounting for leap years.

        Args:
            start (datetime): Start date of the tenure.
            end (datetime): End date of the tenure.

        Returns:
            float: Tenure duration in years.
        """
        delta = end - start
        return delta.total_seconds() / (365.25 * 24 * 3600)

    def process_json_files(self) -> None:
        """Process all JSON files in the specified directory and load into DataFrame."""
        logger.info("Starting processing of JSON files in %s", self.json_dir)
        if not self.json_dir.exists():
            logger.error("Directory %s does not exist", self.json_dir)
            raise FileNotFoundError(f"Directory {self.json_dir} not found")

        for json_file in self.json_dir.glob("*.json"):
            self._process_single_file(json_file)

        self.df = self.load_dataframe()

    def _process_single_file(self, json_file: Path) -> None:
        """Process a single JSON file and append tenure data.

        Args:
            json_file (Path): Path to the JSON file.

        Returns:
            bool: True if job positions were found, False otherwise.
        """
        try:
            with json_file.open("r", encoding="utf-8") as file:
                data = json.load(file)

            member = data.get("data", data)
            us_congress_bio_id = member.get("usCongressBioId", "Unknown")
            job_positions = member.get("jobPositions", [])

            if not job_positions:
                logger.warning("No job positions found in file: %s", json_file)
                return

            dates = self._extract_dates(job_positions, json_file)
            if dates and (
                tenure_info := self._calculate_tenure(dates, us_congress_bio_id)
            ):
                self.tenure_data.append(tenure_info)

        except json.JSONDecodeError:
            logger.exception("JSON parsing error in %s: %s", json_file)
        except Exception:
            logger.exception("Unexpected error processing %s", json_file)

    def _extract_dates(
        self, positions: list[dict[str, Any]], json_file: Path
    ) -> list[CongressDate]:
        """Extract and validate dates from job positions.

        Args:
            positions (list[dict]): List of job position dictionaries.
            json_file (Path): Path to the JSON file for logging.

        Returns:
            List[CongressDate]: List of extracted date records.
        """
        dates: list[CongressDate] = []
        for position in positions:
            try:
                congress_aff = position.get("congressAffiliation", {})
                congress = congress_aff.get("congress", {})
                job = position.get("job", {})

                start_date = congress.get("startDate")
                end_date = congress.get("endDate")
                congress_num = congress.get("congressNumber")
                chamber = job.get("name")

                if start_date:
                    datetime.strptime(start_date, "%Y-%m-%d")
                    if end_date:
                        datetime.strptime(end_date, "%Y-%m-%d")
                    if congress_num is not None and not isinstance(congress_num, int):
                        logger.warning("Invalid congress number in %s", json_file)
                        congress_num = None
                    dates.append(
                        CongressDate(start_date, end_date, congress_num, chamber)
                    )
            except ValueError as e:
                logger.warning("Invalid date format in %s: %s", json_file, e)
            except AttributeError as e:
                logger.warning("Date extraction error in %s: %s", json_file, str(e))
        return dates

    def _calculate_tenure(
        self, dates: list[CongressDate], bio_id: str
    ) -> TenureInfo | None:
        """Calculate tenure duration and related metrics.

        Args:
            dates (List[CongressDate]): List of date records.
            bio_id (str): Unique identifier for the member.

        Returns:
            TenureInfo | None: Tenure information or None if calculation fails.
        """
        try:
            dates.sort(key=lambda x: x[0])
            start = datetime.strptime(dates[0].start_date, "%Y-%m-%d")
            last_end = (
                dates[-1].end_date
                if dates[-1].end_date
                else datetime.now().strftime("%Y-%m-%d")
            )
            end = datetime.strptime(last_end, "%Y-%m-%d")

            tenure_info: TenureInfo = {
                "id": bio_id,
                "tenure_years": self.calculate_exact_years(start, end),
                "start_date": start,
                "end_date": end,
                "congresses": [
                    d.congress_num for d in dates if d.congress_num is not None
                ],
                "chamber": dates[0].chamber if dates[0].chamber else "Unknown",
                "num_terms": len(dates),
            }
            return tenure_info
        except (ValueError, TypeError):
            logger.exception("Tenure calculation error for %s", bio_id)
            return None

    def load_dataframe(self) -> pd.DataFrame:
        """Load data into Pandas DataFrame"""
        if not self.tenure_data:
            logger.warning("No tenure data available to load into DataFrame")
            return pd.DataFrame()
        self.df = pd.DataFrame(self.tenure_data)
        logger.info("DataFrame loaded with %d records", len(self.df))
        return self.df

    def analyze_tenure(self) -> dict[str, Any]:
        """Calculate comprehensive tenure stats."""
        if self.df is None or self.df.empty:
            logger.warning("DataFrame is empty or not loaded")
            self.df = self.load_dataframe()

        tenure_stats = (
            self.df["tenure_years"].agg(["mean", "median", "max", "min"]).to_dict()
        )
        analysis = {
            "avg_tenure": tenure_stats["mean"],
            "median_tenure": tenure_stats["median"],
            "max_tenure": tenure_stats["max"],
            "min_tenure": tenure_stats["min"],
            "chamber_breakdown": self.df.groupby("chamber")["tenure_years"]
            .mean()
            .to_dict(),
        }
        return analysis

    def create_visualizations(self) -> None:
        """Create interactive Plotly visualizatiosn of the data."""
        if self.df is None or self.df.empty:
            self.df = self.load_dataframe()

        VIS_DIR = PROJECT_ROOT / "output" / "visualizations"
        VIS_DIR.mkdir(parents=True, exist_ok=True)

        # Create subplot figure with 2 rows and 2 columns
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Average Tenure by Congress",
                "Tenure Distribution",
                "Tenure by Chamber",
                "Terms vs Tenure",
            ),
            vertical_spacing=0.15,
        )

        # Plot 1: Average tenure by Congress (Line Plot)
        exploded_df = self.df.explode("congresses")
        avg_tenure_by_congress = (
            exploded_df.groupby("congresses")["tenure_years"].mean().reset_index()
        )
        fig.add_scatter(
            x=avg_tenure_by_congress["congresses"],
            y=avg_tenure_by_congress["tenure_years"],
            mode="lines+markers",
            name="Avg Tenure",
            hovertemplate="Congress %{x}<br>Years: %{y:.2f}",
            row=1,
            col=1,
        )

        # Plot 2: Tenure distribution (Histogram)
        fig.add_histogram(
            x=self.df["tenure_years"],
            nbinsx=30,
            name="Tenure Dist",
            hovertemplate="Tenure: %{x}<br>Count: %{y}",
            row=1,
            col=2,
        )

        # Plot 3: Box plot by chamber
        fig.add_box(
            x=self.df["chamber"],
            y=self.df["tenure_years"],
            name="Tenure by Chamber",
            boxpoints="outliers",
            hovertemplate="Chamber: %{x}<br>Tenure: %{y:.2f}",
            row=2,
            col=1,
        )

        # Plot 4: Scatter plot of terms vs tenure
        fig.add_scatter(
            x=self.df["num_terms"],
            y=self.df["tenure_years"],
            mode="markers",
            marker={
                "color": self.df["chamber"].astype("category").cat.codes,
                "colorscale": "viridis",
                "showscale": True,
            },
            text=self.df["chamber"],
            name="Terms vs Tenure",
            hovertemplate="Terms: %{x}<br>Years: %{y:.2f}<br>Chamber: %{text}",
            row=2,
            col=2,
        )

        # Update layout
        fig.update_layout(
            height=800,
            width=1200,
            title_text="Congressional Tenure Analysis",
            showlegend=True,
            template="plotly_white",
            font={"size": 12},
        )

        # Update axes labels
        fig.update_xaxes(title_text="Congress Number", row=1, col=1)
        fig.update_xaxes(title_text="Tenure (Years)", row=1, col=2)
        fig.update_xaxes(title_text="Chamber", row=2, col=1)
        fig.update_xaxes(title_text="Number of Terms", row=2, col=2)

        fig.update_yaxes(title_text="Average Tenure (Years)", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=2)
        fig.update_yaxes(title_text="Tenure (Years)", row=2, col=1)
        fig.update_yaxes(title_text="Tenure (Years)", row=2, col=2)

        # Save and show
        fig.write_html(VIS_DIR / "congress_tenure_analysis.html")
        fig.show()

        scatter_fig = px.scatter(
            self.df,
            x="num_terms",
            y="tenure_years",
            color="chamber",
            size="tenure_years",
            hover_data=["id", "start_date", "end_date"],
            title="Interactive Tenure Analysis",
            labels={
                "num_terms": "Number of Terms",
                "tenure_years": "Tenure (Years)",
                "chamber": "Chamber",
            },
        )
        scatter_fig.update_layout(
            template="plotly_white",
            font={"size": 12},
            showlegend=True,
        )
        scatter_fig.write_html(VIS_DIR / "detailed_tenure_scatter.html")
        scatter_fig.show()

    def export_results(self, output_dir: str = "../output") -> None:
        """Export analysis results and visualizations"""
        output_path = (Path(__file__).parent / output_dir).resolve()
        results_dir = output_path / "results"
        vis_dir = output_path / "visualizations"
        results_dir.mkdir(parents=True, exist_ok=True)
        vis_dir.mkdir(parents=True, exist_ok=True)

        try:
            if self.df is not None:
                self.df.to_csv(results_dir / "tenure_data.csv", index=False)
            with (results_dir / "analysis_summary.json").open("w") as f:
                json.dump(self.analyze_tenure(), f, indent=2)
            logger.info("Results exported to %s", output_dir)
        except Exception:
            logger.exception("Error exporting results")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze congressional tenure data.")
    parser.add_argument(
        "--input-dir",
        default="data/extracted/BioguideProfiles",
        help="Input JSON directory",
    )
    parser.add_argument(
        "--output-dir", default="output", help="Output directory for results"
    )
    args = parser.parse_args()

    try:
        analyzer = CongressAnalyzer(json_dir=args.input_dir)
        analyzer.process_json_files()
        analyzer.df = analyzer.load_dataframe()

        print("Tenure Analysis Summary:")
        for key, value in analyzer.analyze_tenure().items():
            print(f"{key}: {value}")

        analyzer.create_visualizations()
        analyzer.export_results(output_dir=args.output_dir)
    except Exception as e:
        logger.error("Main execution failed: %s", e)
