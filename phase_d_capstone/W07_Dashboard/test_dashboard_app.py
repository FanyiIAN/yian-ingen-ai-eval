from __future__ import annotations

import json
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


DASHBOARD_DIR = Path(__file__).resolve().parent
APP_PATH = DASHBOARD_DIR / "app.py"


class Week7DashboardAppTests(unittest.TestCase):
    def launch(self) -> AppTest:
        app = AppTest.from_file(str(APP_PATH), default_timeout=20).run()
        self.assertEqual(list(app.exception), [])
        return app

    def select_persona(self, app: AppTest, persona: str) -> AppTest:
        app.sidebar.radio[0].set_value(persona).run()
        self.assertEqual(list(app.exception), [], persona)
        self.assertEqual(app.sidebar.radio[0].value, persona)
        return app

    def test_product_manager_is_a_focused_two_view_interface(self) -> None:
        app = self.launch()
        self.assertEqual(app.sidebar.radio[0].value, "Product manager")
        self.assertEqual([tab.label for tab in app.tabs], ["Platform Risk", "RAG Readiness"])
        self.assertEqual(len(app.metric), 6)
        self.assertEqual(len(app.dataframe), 0)
        self.assertEqual(
            [heading.value for heading in app.subheader],
            ["Platform Risk", "RAG Readiness"],
        )
        self.assertNotIn("Data Sources & Reproduction", [heading.value for heading in app.subheader])

    def test_executive_is_three_indicators_without_technical_tabs(self) -> None:
        app = self.select_persona(self.launch(), "Executive")
        self.assertEqual(len(app.metric), 3)
        self.assertEqual(len(app.tabs), 0)
        self.assertEqual(len(app.selectbox), 0)
        self.assertEqual(len(app.get("plotly_chart")), 0)
        self.assertIn("Recommended action", " ".join(item.value for item in app.markdown))

    def test_engineer_exposes_all_required_technical_views(self) -> None:
        app = self.select_persona(self.launch(), "AI evaluation engineer")
        self.assertEqual(
            [tab.label for tab in app.tabs],
            ["Model Scorecard", "RAG Performance", "Robustness Snapshot", "Data Sources & Reproduction"],
        )
        self.assertEqual(len(app.metric), 9)
        self.assertEqual(len(app.dataframe), 2)
        self.assertEqual(len(app.get("plotly_chart")), 7)
        headings = [heading.value for heading in app.subheader]
        for required in [
            "Model Scorecard",
            "RAG Performance",
            "Robustness Snapshot",
            "Data Sources & Reproduction",
        ]:
            self.assertIn(required, headings)
        text = " ".join(item.value for item in app.markdown)
        self.assertIn("11 presentation CSVs", text)
        self.assertIn("8 frozen Week 2–6 source artifacts", text)
        self.assertNotIn("ten committed CSVs", text)

    def test_all_interactive_choices_render_and_update(self) -> None:
        app = self.launch()
        for model in list(app.selectbox[0].options):
            app.selectbox[0].set_value(model).run()
            self.assertEqual(list(app.exception), [], model)
            self.assertEqual(app.selectbox[0].value, model)
        for platform in list(app.selectbox[1].options):
            app.selectbox[1].set_value(platform).run()
            self.assertEqual(list(app.exception), [], platform)
            self.assertEqual(app.selectbox[1].value, platform)
        rag_radio = next(widget for widget in app.radio if widget.label == "Knowledge-base track")
        for track in ["Fari", "Senpai"]:
            rag_radio.set_value(track).run()
            rag_radio = next(widget for widget in app.radio if widget.label == "Knowledge-base track")
            self.assertEqual(list(app.exception), [], track)
            self.assertEqual(rag_radio.value, track)

        app = self.select_persona(app, "AI evaluation engineer")
        vlm_select = next(widget for widget in app.selectbox if widget.label == "Image condition")
        for condition in list(vlm_select.options):
            vlm_select.set_value(condition).run()
            vlm_select = next(widget for widget in app.selectbox if widget.label == "Image condition")
            self.assertEqual(list(app.exception), [], condition)
            self.assertEqual(vlm_select.value, condition)

    def test_masked_input_lines_use_distinguishable_colors(self) -> None:
        app = self.select_persona(self.launch(), "AI evaluation engineer")
        specs = [json.loads(element.proto.spec) for element in app.get("plotly_chart")]
        line_chart = next(
            spec
            for spec in specs
            if {trace.get("name") for trace in spec["data"]}
            == {"FLAN-T5 Base", "Llama 3.1 8B Instruct", "Mistral 7B Instruct v0.2"}
        )
        colors = {trace["line"]["color"] for trace in line_chart["data"]}
        self.assertEqual(colors, {"#2A66B7", "#20B8CD", "#E2A33A"})

    def test_light_theme_and_contrast_overrides_are_registered(self) -> None:
        config = (DASHBOARD_DIR / ".streamlit" / "config.toml").read_text(encoding="utf-8")
        app_source = APP_PATH.read_text(encoding="utf-8")
        self.assertIn('base = "light"', config)
        self.assertIn('[data-testid="stMetricLabel"]', app_source)
        self.assertIn('button[role="tab"] p', app_source)
        self.assertIn('[data-testid="stWidgetLabel"]', app_source)


if __name__ == "__main__":
    unittest.main()
