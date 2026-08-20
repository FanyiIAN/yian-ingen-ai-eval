"""Week 7 stakeholder dashboard.

Every displayed result is loaded from a reviewed CSV. Dashboard launch runs no
model inference, retrieval, aggregation, or LLM Judge call.
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
NAVY = "#10233F"
BLUE = "#2A66B7"
CYAN = "#20B8CD"
AMBER = "#E2A33A"
MUTED = "#64748B"

st.set_page_config(
    page_title="InGen AI Evaluation | Week 7",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .stApp { background: #F5F7FB; color: #10233F; }
      [data-testid="stSidebar"] { background: #0D1C31; color: white; }
      [data-testid="stSidebar"] * { color: #E7EEF8; }
      .hero {
        background: linear-gradient(125deg, #10233F 0%, #173C69 62%, #1B788D 100%);
        color: white; border-radius: 18px; padding: 28px 30px 24px; margin-bottom: 18px;
        box-shadow: 0 12px 30px rgba(16, 35, 63, 0.16);
      }
      .hero .eyebrow { color: #77DBE7; font-size: 0.78rem; letter-spacing: 0.14em; font-weight: 700; }
      .hero h1 { margin: 6px 0 6px; font-size: 2.25rem; color: white !important; }
      .hero p { margin: 0; color: #DCE8F6; max-width: 860px; }
      .evidence-banner {
        border-left: 5px solid #E2A33A; background: #FFF8E7; color: #553D0C;
        padding: 12px 16px; border-radius: 9px; margin: 4px 0 18px;
      }
      .decision-card {
        background: white; border: 1px solid #E2E8F0; border-radius: 14px;
        padding: 18px 20px; box-shadow: 0 4px 16px rgba(15, 23, 42, 0.05);
      }
      .decision-card h3 { color: #10233F; margin-top: 0; }
      .decision-card p { color: #475569; margin-bottom: 0; }
      div[data-testid="stMetric"] {
        background: white; border: 1px solid #E2E8F0; padding: 14px 16px;
        border-radius: 13px; box-shadow: 0 4px 14px rgba(15, 23, 42, 0.04);
      }
      [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * {
        color: #475569 !important; opacity: 1 !important;
      }
      [data-testid="stMetricValue"], [data-testid="stMetricValue"] * { color: #10233F !important; }
      [data-testid="stAppViewContainer"] [data-testid="stCaptionContainer"],
      [data-testid="stAppViewContainer"] [data-testid="stCaptionContainer"] p,
      [data-testid="stAppViewContainer"] [data-testid="stWidgetLabel"],
      [data-testid="stAppViewContainer"] [data-testid="stWidgetLabel"] p,
      [data-testid="stAppViewContainer"] label,
      [data-testid="stAppViewContainer"] label p {
        color: #475569 !important; opacity: 1 !important;
      }
      button[role="tab"] p { color: #475569 !important; opacity: 1 !important; }
      button[role="tab"][aria-selected="true"] p { color: #C83E49 !important; }
      [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
      [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p,
      [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
      [data-testid="stSidebar"] label,
      [data-testid="stSidebar"] label p { color: #E7EEF8 !important; }
      .stApp h2, .stApp h3, .stApp h4 { color: #10233F !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_csv(name: str) -> pd.DataFrame:
    path = DATA_DIR / name
    if path.parent != DATA_DIR or not path.is_file():
        raise FileNotFoundError(f"Registered dashboard CSV is missing: {name}")
    return pd.read_csv(path)


def style_figure(figure: go.Figure) -> go.Figure:
    """Keep charts legible regardless of the browser color preference."""
    figure.update_layout(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font={"color": NAVY},
        margin={"l": 48, "r": 28, "t": 62, "b": 48},
    )
    figure.update_xaxes(gridcolor="#E8EEF5", linecolor="#CBD5E1")
    figure.update_yaxes(gridcolor="#E8EEF5", linecolor="#CBD5E1")
    return figure


def compact_number(value: float) -> str:
    return f"{float(value):g}"


scorecard = load_csv("model_scorecard.csv")
failure_heatmap = load_csv("failure_heatmap.csv")
failure_concerns = load_csv("platform_failure_concerns.csv")
executive_summary = load_csv("executive_summary.csv")
rag_performance = load_csv("rag_performance.csv")
rag_configurations = load_csv("rag_configurations.csv")
robustness = load_csv("robustness_summary.csv")
masked_curves = load_csv("masked_input_curves.csv")
vlm = load_csv("vlm_performance.csv")
manifest = load_csv("data_manifest.csv")
metadata = load_csv("dashboard_metadata.csv").iloc[0]

dashboard_version = str(metadata["dashboard_version"])
evaluation_seed = int(metadata["seed"])
calibration_alpha = float(metadata["judge_calibration_alpha"])
calibration_threshold = float(metadata["judge_calibration_threshold"])
reviewer_minimum = int(metadata["minimum_independent_reviewers"])
rag_runtime_target = str(metadata["rag_runtime_target"]).replace("_", " ")

st.sidebar.markdown("## ◈ InGen Evaluation")
st.sidebar.caption(f"Phase D · Week 7 · dashboard v{dashboard_version}")
persona = st.sidebar.radio(
    "Audience",
    ["AI evaluation engineer", "Product manager", "Executive"],
    index=1,
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Evidence status**")
st.sidebar.warning("Diagnostic proxy — not deployment validated")
st.sidebar.caption(
    f"Judge calibration: α = {calibration_alpha:.4f}, below the preregistered {calibration_threshold:.2f} gate."
)
st.sidebar.caption(f"All cited runs use seed {evaluation_seed} and frozen model revisions.")

persona_design = {
    "AI evaluation engineer": (
        "Audit every result",
        "Full metrics, factor settings, coverage, source hashes, and reproduction commands.",
    ),
    "Product manager": (
        "Find platform risk and RAG trade-offs",
        "A focused decision view for platform concerns and Fari/Senpai knowledge-base readiness.",
    ),
    "Executive": (
        "Decision status: validation required",
        "Three registered indicators and one action; technical evidence stays out of the decision path.",
    ),
}


def render_hero() -> None:
    title, copy = persona_design[persona]
    st.markdown(
        f"""
        <div class="hero">
          <div class="eyebrow">WEEK 7 · AI MODEL EVALUATION DASHBOARD</div>
          <h1>{title}</h1>
          <p>{copy}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="evidence-banner"><b>Interpretation boundary.</b> “Readiness” below is a diagnostic
        severity-weighted proxy from public/synthetic evaluation data. It is not a product safety certification,
        deployed PIC measurement, or validated model ranking.</div>
        """,
        unsafe_allow_html=True,
    )


def render_executive_view() -> None:
    columns = st.columns(3)
    for column, (_, metric) in zip(columns, executive_summary.iterrows()):
        with column:
            st.metric(
                metric["label"],
                f"{compact_number(metric['value'])} {metric['unit']}",
                help=metric["detail"],
            )
    st.markdown(
        f"""
        <div class="decision-card">
          <h3>Recommended action</h3>
          <p>Do not select a deployment model from the diagnostic ordering. First run model-blind,
          severity-stratified calibration with at least {reviewer_minimum} independent domain experts, then
          validate the selected system in a closed-loop simulator with sensor dropout and safe-recovery metrics.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info(
        "Decision meaning: the current evidence can prioritise follow-up tests, but it cannot approve a model "
        "or platform for deployment."
    )


def render_model_scorecard(*, engineer_detail: bool) -> None:
    st.subheader("Model Scorecard" if engineer_detail else "Platform Risk")
    st.caption(
        "Severity-weighted Task and Grounding scores use a 1–5 rubric. The 0–100 proxy is their arithmetic "
        "mean divided by five; score coverage is shown separately."
    )
    models = list(scorecard.loc[scorecard["platform"] == "Portfolio", "model"])
    selected_model = st.selectbox(
        "Model to inspect",
        models,
        index=models.index("Mistral 7B Instruct v0.2"),
        key="scorecard_model",
    )
    rows = scorecard.loc[
        (scorecard["model"] == selected_model) & (scorecard["platform"] != "Portfolio")
    ]
    score_fig = px.bar(
        rows,
        x="platform",
        y="diagnostic_readiness_proxy_0_to_100",
        color="minimum_dimension_coverage",
        color_continuous_scale=["#F3C969", "#20B8CD", "#2A66B7"],
        range_y=[0, 100],
        labels={
            "platform": "Platform",
            "diagnostic_readiness_proxy_0_to_100": "Diagnostic readiness proxy",
            "minimum_dimension_coverage": "Score coverage",
        },
        title=f"{selected_model}: five-platform diagnostic proxy",
    )
    score_fig.update_layout(template="plotly_white", coloraxis_colorbar_tickformat=".0%")
    st.plotly_chart(style_figure(score_fig), theme=None)

    if engineer_detail:
        st.dataframe(
            rows[
                [
                    "platform",
                    "model_revision",
                    "task_severity_weighted_1_to_5",
                    "grounding_severity_weighted_1_to_5",
                    "diagnostic_readiness_proxy_0_to_100",
                    "minimum_dimension_coverage",
                    "mean_latency_ms",
                    "evidence_status",
                ]
            ],
            hide_index=True,
            width="stretch",
        )

    st.markdown("#### Failure-mode distribution heat map")
    heat_columns = ["unsafe", "hallucination", "off_policy", "refusal", "partial", "unresolved"]
    heat_fig = px.imshow(
        failure_heatmap[heat_columns],
        x=[name.replace("_", " ").title() for name in heat_columns],
        y=failure_heatmap["platform"],
        labels={"x": "Failure mode", "y": "Platform", "color": "% of observations"},
        color_continuous_scale=["#F7FAFC", "#F3C969", "#D65A5A"],
        text_auto=".1f",
        aspect="auto",
    )
    heat_fig.update_layout(template="plotly_white")
    st.plotly_chart(style_figure(heat_fig), theme=None)
    st.caption(
        "Unresolved means scoring uncertainty, not model behaviour. Percentages pool three frozen model "
        "observations per scenario and are diagnostic rather than incident rates."
    )

    st.markdown("#### Top three observed failure concerns per platform")
    platform = st.selectbox(
        "Choose a platform",
        list(failure_heatmap["platform"]),
        key="concern_platform",
    )
    concerns = failure_concerns.loc[failure_concerns["platform"] == platform]
    concern_columns = st.columns(3)
    for column, (_, concern) in zip(concern_columns, concerns.iterrows()):
        with column:
            st.metric(
                f"#{int(concern['rank'])} {str(concern['failure_code']).replace('_', ' ').title()}",
                f"{int(concern['observed_count'])} observed",
                help=(
                    f"{concern['observed_rate_pct']}% of pooled model-scenario observations. "
                    "Ranked by count; consequence priority breaks ties."
                ),
            )
    if not engineer_detail:
        st.info(
            "How to use this: select a platform and read the cards from left to right. They rank observed "
            "failure counts; they do not estimate the deployed product's real-world failure rate."
        )


def render_rag_performance(*, show_frontier: bool) -> None:
    st.subheader("RAG Performance" if show_frontier else "RAG Readiness")
    st.caption(
        "Base and RAG use the same frozen Llama revision and matched questions. Base faithfulness is blank "
        "because there is no retrieved context against which faithfulness can be measured."
    )
    track = st.radio(
        "Knowledge-base track",
        ["Fari", "Senpai"],
        horizontal=True,
        key="rag_track",
    )
    rows = rag_performance.loc[rag_performance["platform"] == track]
    base = rows.loc[rows["condition"] == "BASE"].iloc[0]
    rag = rows.loc[rows["condition"] == "RAG"].iloc[0]
    names = ["Answer relevance", "Faithfulness", "Required-point coverage"]
    metric_fig = go.Figure(
        data=[
            go.Bar(
                name="Base",
                x=names,
                y=[base["answer_relevance"], base["faithfulness"], base["required_point_coverage"]],
                marker_color=MUTED,
            ),
            go.Bar(
                name="RAG",
                x=names,
                y=[rag["answer_relevance"], rag["faithfulness"], rag["required_point_coverage"]],
                marker_color=CYAN,
            ),
        ]
    )
    metric_fig.update_layout(
        barmode="group",
        template="plotly_white",
        yaxis_range=[0, 1],
        yaxis_title="Diagnostic metric",
    )
    st.plotly_chart(style_figure(metric_fig), theme=None)
    r1, r2, r3 = st.columns(3)
    r1.metric(
        "RAG quality proxy",
        f"{rag['diagnostic_quality_proxy_0_to_100']:.1f}/100",
        help="Mean of relevance, faithfulness, and coverage; diagnostic only.",
    )
    r2.metric("Evidence recall@k", f"{rag['evidence_fact_recall_at_k']:.2f}")
    r3.metric("P50 generation latency", f"{rag['generation_latency_p50_ms'] / 1000:.2f}s")
    st.warning(str(rag["deployment_readiness"]))

    if not show_frontier:
        st.info(
            "Product interpretation: RAG improves the displayed knowledge metrics, but latency is measured in "
            "seconds and deployment readiness remains unestablished. This supports test prioritisation, not approval."
        )
        return

    st.markdown("#### Week 5 configuration frontier")
    config_fig = px.scatter(
        rag_configurations,
        x="p50_question_to_response_ms",
        y="quality_harmonic_mean",
        color="is_pareto",
        symbol="reranking",
        size="top_k",
        hover_name="variant_id",
        hover_data=[
            "chunk_size_tokens",
            "mean_answer_relevance",
            "faithfulness_coverage",
            "required_point_coverage_coverage",
        ],
        color_discrete_map={True: AMBER, False: BLUE},
        labels={
            "p50_question_to_response_ms": "P50 question-to-response latency (ms)",
            "quality_harmonic_mean": "Faithfulness–coverage harmonic mean",
            "is_pareto": "Pareto-optimal",
        },
        title=(
            f"{int(metadata['registered_rag_cells'])} registered cells; "
            f"{int(metadata['pareto_rag_cells'])} Pareto points highlighted"
        ),
    )
    config_fig.update_layout(template="plotly_white")
    st.plotly_chart(style_figure(config_fig), theme=None)
    balanced = rag_configurations.loc[
        rag_configurations["is_balanced_choice"] == True  # noqa: E712
    ].iloc[0]
    st.info(
        f"Balanced diagnostic choice: {balanced['variant_id']} — "
        f"{int(balanced['chunk_size_tokens'])}-token chunks, top-k {int(balanced['top_k'])}, "
        f"{balanced['reranking'].replace('_', ' ')} reranking. This is conditional on one corpus, stack, "
        f"{rag_runtime_target} run, and uncalibrated metrics."
    )


def render_robustness_snapshot() -> None:
    st.subheader("Robustness Snapshot")
    st.caption(
        "Consistency is paired with stable pass/fail counts so a model cannot appear robust merely by failing consistently."
    )
    semantic_fig = px.bar(
        robustness,
        x="model",
        y="semantic_consistency",
        color="stable_fail_scenarios",
        range_y=[0, 1],
        color_continuous_scale=["#20B8CD", "#F3C969", "#D65A5A"],
        labels={
            "model": "Model",
            "semantic_consistency": "Semantic consistency",
            "stable_fail_scenarios": "Stable failures",
        },
        title="Semantic consistency with stable-failure context",
    )
    semantic_fig.update_layout(template="plotly_white")
    st.plotly_chart(style_figure(semantic_fig), theme=None)

    st.markdown("#### Masked-input degradation")
    masked_fig = px.line(
        masked_curves,
        x="mask_ratio",
        y="severity_weighted_mean_task_accuracy",
        color="model",
        color_discrete_map={
            "FLAN-T5 Base": BLUE,
            "Llama 3.1 8B Instruct": CYAN,
            "Mistral 7B Instruct v0.2": AMBER,
        },
        markers=True,
        range_y=[0, 5],
        labels={
            "mask_ratio": "Masked input ratio",
            "severity_weighted_mean_task_accuracy": "Severity-weighted Task score",
        },
    )
    masked_fig.update_layout(template="plotly_white")
    st.plotly_chart(style_figure(masked_fig), theme=None)

    st.markdown("#### VLM multimodal proxy performance")
    condition = st.selectbox(
        "Image condition",
        ["clean", "gaussian_noise_std_0.08", "brightness_0.60"],
        key="vlm_condition",
    )
    rows = vlm.loc[vlm["condition"] == condition]
    vlm_fig = px.bar(
        rows,
        x="model",
        y="mean_total_score_0_to_5",
        color="latency_p50_ms",
        range_y=[0, 5],
        color_continuous_scale=[CYAN, BLUE, NAVY],
        labels={
            "model": "VLM",
            "mean_total_score_0_to_5": "Mean diagnostic score",
            "latency_p50_ms": "P50 latency (ms)",
        },
    )
    vlm_fig.update_layout(template="plotly_white")
    st.plotly_chart(style_figure(vlm_fig), theme=None)
    st.caption(
        f"{int(metadata['vlm_scenarios_per_condition'])} public-image proxies per condition and model; "
        f"{int(metadata['vlm_requests_per_model'])} matched requests per architecture. "
        "No deployed camera or product system was measured."
    )


def render_data_sources() -> None:
    st.subheader("Data Sources & Reproduction")
    st.markdown(
        f"This is an **audit view for evaluation engineers**. The dashboard loads **11 presentation CSVs**; "
        f"the table registers the **{len(manifest)} frozen Week 2–6 source artifacts** used to build them. "
        "A SHA-256 value is the exact file fingerprint: if a source changes, its fingerprint changes. "
        "Dashboard launch performs no model inference, retrieval, aggregation, or Judge scoring."
    )
    st.info(
        "Use this page to answer two questions: Where did a displayed number come from? Can another evaluator "
        "rebuild the same presentation CSVs from the same frozen inputs?"
    )
    st.dataframe(manifest, hide_index=True, width="stretch")
    st.markdown("#### Reproduction links")
    st.markdown(
        "- `phase_b_evaluation/W03_Extended_Benchmark.ipynb` — three-model benchmark\n"
        "- `phase_b_evaluation/W03_RAG_Long_Source_Evaluation.ipynb` — matched long-source RAG\n"
        "- `phase_b_evaluation/W04_Robustness_Eval.ipynb` — semantic and masked robustness\n"
        "- `phase_b_evaluation/W04_Multimodal_Eval.ipynb` — two-VLM comparison\n"
        "- `phase_c_synthesis/W05_RAG_Optimisation.ipynb` — full-factorial RAG optimisation\n"
        "- `phase_c_synthesis/W06_Submission_Index.md` — methodology and evidence synthesis"
    )
    st.code(
        "python phase_d_capstone/W07_Dashboard/build_dashboard_data.py\n"
        "python -m unittest phase_d_capstone/W07_Dashboard.test_dashboard_contract -v",
        language="bash",
    )


render_hero()

if persona == "Executive":
    render_executive_view()
elif persona == "Product manager":
    st.markdown("### Product decision view")
    st.caption(
        "Use Platform Risk to identify where observed failures concentrate. Use RAG Readiness to compare "
        "knowledge quality with latency. Technical audit material is hidden in this audience mode."
    )
    tab_risk, tab_rag = st.tabs(["Platform Risk", "RAG Readiness"])
    with tab_risk:
        render_model_scorecard(engineer_detail=False)
    with tab_rag:
        render_rag_performance(show_frontier=False)
else:
    st.markdown("### Technical evaluation status")
    columns = st.columns(3)
    columns[0].metric("Judge agreement α", f"{calibration_alpha:.4f}")
    columns[1].metric("Calibration gate", f"{calibration_threshold:.2f} · failed")
    columns[2].metric("Frozen source artifacts", f"{len(manifest)} files")
    st.warning(
        "The Judge agreement missed the preregistered gate. Treat every ordering and readiness value as "
        "diagnostic evidence rather than a validated leaderboard."
    )
    tab_scorecard, tab_rag, tab_robustness, tab_evidence = st.tabs(
        ["Model Scorecard", "RAG Performance", "Robustness Snapshot", "Data Sources & Reproduction"]
    )
    with tab_scorecard:
        render_model_scorecard(engineer_detail=True)
    with tab_rag:
        render_rag_performance(show_frontier=True)
    with tab_robustness:
        render_robustness_snapshot()
    with tab_evidence:
        render_data_sources()

st.markdown("---")
st.caption(
    f"Week 7 dashboard v{dashboard_version} · public/synthetic proxy evidence · "
    f"frozen model revisions · seed {evaluation_seed} · no live inference"
)
