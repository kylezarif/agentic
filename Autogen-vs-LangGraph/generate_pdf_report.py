"""
PDF Report Generator
===================
Generates a comprehensive PDF report from experimental results.
"""

import json
import datetime
from pathlib import Path
from typing import List, Dict, Any

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    Image
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT


class PDFReportGenerator:
    """Generates PDF reports from experimental results."""

    def __init__(self, output_filename: str = "experimental_report.pdf"):
        self.output_path = Path("results") / output_filename
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self.doc = SimpleDocTemplate(
            str(self.output_path),
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18,
        )

        self.styles = getSampleStyleSheet()
        self.story = []

        # Custom styles
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#2E86AB'),
            spaceAfter=30,
            alignment=TA_CENTER
        )

        self.heading_style = ParagraphStyle(
            'CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#A23B72'),
            spaceAfter=12,
            spaceBefore=12
        )

    def add_title_page(self, title: str, subtitle: str = ""):
        """Add title page."""
        self.story.append(Spacer(1, 2*inch))

        self.story.append(Paragraph(title, self.title_style))
        self.story.append(Spacer(1, 0.3*inch))

        if subtitle:
            subtitle_style = ParagraphStyle(
                'Subtitle',
                parent=self.styles['Normal'],
                fontSize=14,
                textColor=colors.grey,
                alignment=TA_CENTER
            )
            self.story.append(Paragraph(subtitle, subtitle_style))
            self.story.append(Spacer(1, 0.5*inch))

        # Date
        date_text = f"Generated: {datetime.datetime.now().strftime('%B %d, %Y at %I:%M %p')}"
        date_style = ParagraphStyle(
            'Date',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=colors.grey,
            alignment=TA_CENTER
        )
        self.story.append(Paragraph(date_text, date_style))

        self.story.append(PageBreak())

    def add_section(self, title: str, content: str = ""):
        """Add a section with title."""
        self.story.append(Paragraph(title, self.heading_style))

        if content:
            self.story.append(Paragraph(content, self.styles['Normal']))

        self.story.append(Spacer(1, 0.2*inch))

    def add_results_table(self, data: List[List[str]], headers: List[str]):
        """Add a formatted results table."""
        table_data = [headers] + data

        table = Table(table_data, repeatRows=1)
        table.setStyle(TableStyle([
            # Header styling
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E86AB')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),

            # Body styling
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ALIGN', (0, 1), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
        ]))

        self.story.append(table)
        self.story.append(Spacer(1, 0.3*inch))

    def load_results(self, results_dir: Path) -> Dict[str, Any]:
        """Load all result JSON files."""
        results = {"autogen": [], "langgraph": []}

        for json_file in results_dir.glob("*.json"):
            try:
                with open(json_file, "r") as f:
                    data = json.load(f)

                    framework = data.get("framework", "").lower()
                    if "autogen" in framework:
                        results["autogen"].append(data)
                    elif "langgraph" in framework:
                        results["langgraph"].append(data)
            except Exception as e:
                print(f"Error loading {json_file}: {e}")

        return results

    def generate_report(self, results_dir: Path = Path("results/pilot")):
        """Generate complete PDF report."""

        # Title page
        self.add_title_page(
            "Agentic AI Frameworks Comparative Study",
            "AutoGen vs LangGraph - Pilot Experiment Results"
        )

        # Executive summary
        self.add_section(
            "Executive Summary",
            "This report presents the results of a pilot comparative evaluation of "
            "AutoGen and LangGraph frameworks using standardized benchmarks and "
            "custom GEval metrics aligned with the study's research questions."
        )
        self.story.append(Spacer(1, 0.2*inch))

        # Load results
        all_results = self.load_results(results_dir)

        # Results section
        self.add_section("Experimental Results")

        # AutoGen results
        if all_results["autogen"]:
            self.add_section("AutoGen Framework Results", "")

            for i, result in enumerate(all_results["autogen"]):
                data = [
                    ["Metric", "Value"],
                    ["Benchmark", result.get("benchmark", "N/A")],
                    ["Model", result.get("model", "N/A")],
                    ["Temperature", str(result.get("temperature", "N/A"))],
                    ["Problems", str(result.get("n_problems", 0))],
                    ["Correct", str(result.get("correct", 0))],
                    ["Total", str(result.get("total", 0))],
                    ["Accuracy", f"{result.get('accuracy', 0):.1%}"],
                ]

                self.add_results_table(data[1:], data[0])

        # LangGraph results
        if all_results["langgraph"]:
            self.add_section("LangGraph Framework Results", "")

            for i, result in enumerate(all_results["langgraph"]):
                data = [
                    ["Metric", "Value"],
                    ["Benchmark", result.get("benchmark", "N/A")],
                    ["Model", result.get("model", "N/A")],
                    ["Temperature", str(result.get("temperature", "N/A"))],
                    ["Problems", str(result.get("n_problems", 0))],
                    ["Correct", str(result.get("correct", 0))],
                    ["Total", str(result.get("total", 0))],
                    ["Accuracy", f"{result.get('accuracy', 0):.1%}"],
                ]

                self.add_results_table(data[1:], data[0])

        # Comparison
        if all_results["autogen"] and all_results["langgraph"]:
            self.story.append(PageBreak())
            self.add_section("Comparative Analysis")

            autogen_acc = all_results["autogen"][0].get("accuracy", 0)
            langgraph_acc = all_results["langgraph"][0].get("accuracy", 0)

            comparison_data = [
                ["Framework", "Accuracy", "Winner"],
                ["AutoGen", f"{autogen_acc:.1%}", "✓" if autogen_acc > langgraph_acc else ""],
                ["LangGraph", f"{langgraph_acc:.1%}", "✓" if langgraph_acc > autogen_acc else ""],
            ]

            if autogen_acc == langgraph_acc:
                comparison_data[1][2] = "="
                comparison_data[2][2] = "="

            self.add_results_table(comparison_data[1:], comparison_data[0])

            # Analysis text
            if autogen_acc > langgraph_acc:
                winner_text = "AutoGen outperformed LangGraph in this pilot evaluation."
            elif langgraph_acc > autogen_acc:
                winner_text = "LangGraph outperformed AutoGen in this pilot evaluation."
            else:
                winner_text = "Both frameworks achieved identical performance in this pilot evaluation."

            self.story.append(Paragraph(winner_text, self.styles['Normal']))

        # Detailed results
        self.story.append(PageBreak())
        self.add_section("Detailed Problem-by-Problem Results")

        for framework, results_list in all_results.items():
            if results_list:
                self.add_section(f"{framework.upper()} - Problem Details", "")

                for result in results_list:
                    if "results" in result:
                        for problem in result["results"]:
                            problem_data = [
                                ["Field", "Value"],
                                ["Problem ID", str(problem.get("problem_id", "N/A"))],
                                ["Question", problem.get("question", "N/A")[:100] + "..."],
                                ["Expected", str(problem.get("expected_answer", "N/A"))],
                                ["Correct", "✓" if problem.get("is_correct") else "✗"],
                            ]

                            self.add_results_table(problem_data[1:], problem_data[0])

        # Conclusion
        self.story.append(PageBreak())
        self.add_section(
            "Conclusion",
            "This pilot experiment successfully demonstrated the evaluation framework. "
            "Both AutoGen and LangGraph frameworks were tested using identical conditions "
            "(model: gpt-4o-mini, temperature: 0.3). The framework is ready for full-scale "
            "evaluation with 50-100 problems across 4 benchmarks (GSM8K, HumanEval, ARC, MATH) "
            "with 5-10 repetitions as specified in the experimental design."
        )

        # Build PDF
        self.doc.build(self.story)
        print(f"\n✅ PDF Report generated: {self.output_path.absolute()}")

        return self.output_path


def main():
    """Generate PDF report from pilot results."""
    print("\n" + "="*70)
    print("GENERATING PDF REPORT")
    print("="*70)

    generator = PDFReportGenerator("Pilot_Experiment_Report.pdf")
    output_file = generator.generate_report(Path("results/pilot"))

    print(f"\n📄 Report saved to: {output_file}")
    print("="*70)


if __name__ == "__main__":
    main()
