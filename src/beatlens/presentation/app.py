"""Local Streamlit research demo for QTDB T-peak explanations."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from beatlens.explainers.temporal_attribution import integrated_gradients, temporal_lime
from beatlens.infrastructure.qtdb import QTDBBeatExample, iter_qtdb_examples
from beatlens.models.qtdb_tpeak import QTDBTpeakRegressor

LEFT_SECONDS = 1.0
LIME_SEEDS = (19, 41, 73, 101)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _normalize(example: QTDBBeatExample) -> np.ndarray:
    signal = example.signal.astype(np.float32)
    valid = example.valid_samples[:, None].astype(np.float32)
    count = max(float(valid.sum()), 1.0)
    means = (signal * valid).sum(axis=0, keepdims=True) / count
    variances = (((signal - means) ** 2) * valid).sum(axis=0, keepdims=True) / count
    scales = np.sqrt(variances).clip(min=1e-5)
    return (((signal - means) / scales) * valid).T.astype(np.float32)


def _to_offset_ms(model: torch.nn.Module, inputs: torch.Tensor, fs: float) -> torch.Tensor:
    normalized_sample = model(inputs)
    absolute_sample = (normalized_sample + 1.0) * (inputs.shape[-1] - 1) / 2.0
    return (absolute_sample - round(fs * LEFT_SECONDS)) * (1000.0 / fs)


def _predict_ms(model: torch.nn.Module, batch: np.ndarray, fs: float) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        predictions = []
        for start in range(0, len(batch), 128):
            inputs = torch.from_numpy(batch[start : start + 128])
            predictions.extend(_to_offset_ms(model, inputs, fs).cpu().numpy())
    return np.asarray(predictions)


def _top_bins(scores: np.ndarray, fraction: float = 0.10) -> set[int]:
    count = max(1, math.ceil(len(scores) * fraction))
    return set(np.argsort(np.abs(scores))[-count:].astype(int).tolist())


def _jaccard(left: np.ndarray, right: np.ndarray) -> float:
    a, b = _top_bins(left), _top_bins(right)
    return len(a & b) / max(len(a | b), 1)


def _segment_sum(values: np.ndarray, width: int) -> np.ndarray:
    return np.asarray(
        [values[start : min(start + width, len(values))].sum() for start in range(0, len(values), width)]
    )


def _compute_explanations(
    model: torch.nn.Module,
    example: QTDBBeatExample,
    lime_perturbations: int = 512,
    ig_steps: int = 48,
) -> dict[str, Any]:
    import torch
    from scipy.ndimage import gaussian_filter1d

    fs = example.sampling_rate_hz
    normalized = _normalize(example)
    input_tensor = torch.from_numpy(normalized[None])
    output_fn = lambda raw, rate=fs: (
        (raw + 1.0) * (input_tensor.shape[-1] - 1) / 2.0 - round(rate * LEFT_SECONDS)
    ) * (1000.0 / rate)
    with torch.no_grad():
        prediction_ms = float(output_fn(model(input_tensor)).item())
    reference_ms = example.event_offsets["t_peak"][0] / fs * 1000.0
    segment_samples = max(1, round(fs * 0.025))
    zero_baseline = torch.zeros_like(input_tensor)
    smooth_np = gaussian_filter1d(normalized, sigma=12.0, axis=-1, mode="nearest").astype(np.float32)
    smooth_baseline = torch.from_numpy(smooth_np[None])

    explanations: dict[str, Any] = {
        "prediction_ms": prediction_ms,
        "reference_ms": reference_ms,
        "absolute_error_ms": abs(prediction_ms - reference_ms),
        "normalized_input": normalized,
        "segment_samples": segment_samples,
    }
    ig_segments: dict[str, np.ndarray] = {}
    lime_segments: dict[str, np.ndarray] = {}
    for name, baseline_tensor, baseline_np in (
        ("zero", zero_baseline, np.zeros_like(normalized)),
        ("smooth", smooth_baseline, smooth_np),
    ):
        attribution, completeness = integrated_gradients(
            model, input_tensor, output_fn, baseline=baseline_tensor, steps=ig_steps
        )
        ig_magnitude = attribution[0].abs().sum(dim=0).cpu().numpy()
        ig_signed = attribution[0].sum(dim=0).cpu().numpy()
        ig_segments[name] = _segment_sum(ig_magnitude, segment_samples)

        predictor = lambda batch, rate=fs: _predict_ms(model, batch, rate)
        lime_results = [
            temporal_lime(
                normalized,
                predictor,
                segment_samples=segment_samples,
                replacement=baseline_np,
                perturbations=lime_perturbations,
                seed=seed,
            )
            for seed in LIME_SEEDS
        ]
        coefficients = np.stack([result.coefficients for result in lime_results])
        lime_signed = coefficients.mean(axis=0)
        lime_magnitude = np.abs(coefficients).mean(axis=0)
        lime_segments[name] = lime_magnitude
        explanations[f"{name}_ig"] = {
            "signed_ms": _segment_sum(ig_signed, segment_samples),
            "magnitude": ig_segments[name],
            "completeness_residual_ms": completeness,
        }
        explanations[f"{name}_lime"] = {
            "signed_ms": lime_signed,
            "magnitude": lime_magnitude,
            "weighted_r2": float(np.mean([result.weighted_r2 for result in lime_results])),
        }

    explanations["ig_baseline_jaccard"] = _jaccard(ig_segments["zero"], ig_segments["smooth"])
    explanations["lime_baseline_jaccard"] = _jaccard(lime_segments["zero"], lime_segments["smooth"])
    return explanations


def _render() -> None:
    import plotly.graph_objects as go
    import streamlit as st
    from scipy.ndimage import gaussian_filter1d
    from plotly.subplots import make_subplots

    st.set_page_config(page_title="BeatLens T-peak XAI", page_icon="〰️", layout="wide")
    st.title("BeatLens · T-peak explanation lab")
    st.caption(
        "Research prototype on public QTDB annotations. Explanations describe this model's output; "
        "they are not clinical or causal evidence."
    )

    root = _repository_root()
    qtdb_dir = Path(st.sidebar.text_input("QTDB directory", str(root / "data/qt-database-1.0.0")))
    split_csv = Path(st.sidebar.text_input("Split manifest", str(root / "outputs/qtdb-audit/qtdb-split.csv")))
    checkpoint_path = Path(
        st.sidebar.text_input(
            "Model checkpoint",
            str(root / "outputs/qtdb-tpeak-known-subject-groups/best.pt"),
        )
    )
    st.sidebar.markdown("**Explanation references**")
    st.sidebar.write("Primary: zero in per-window z-scored input")
    st.sidebar.write("Sensitivity: 48 ms Gaussian-smoothed waveform")

    if not qtdb_dir.exists() or not split_csv.exists() or not checkpoint_path.exists():
        st.error("Local QTDB data, split manifest, or T-peak checkpoint was not found. See the run instructions below.")
        st.code(
            "uv sync --extra qtdb --extra demo\n"
            "uv run streamlit run src/beatlens/presentation/app.py",
            language="bash",
        )
        st.stop()

    @st.cache_resource
    def load_model(checkpoint: str) -> QTDBTpeakRegressor:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        network = QTDBTpeakRegressor(input_leads=2)
        network.load_state_dict(payload["model_state_dict"])
        network.eval()
        return network

    @st.cache_data(show_spinner="Loading validation examples…")
    def load_examples(data_dir: str, manifest: str) -> list[QTDBBeatExample]:
        return [
            example
            for example in iter_qtdb_examples(Path(data_dir), Path(manifest), "validation")
            if len(example.event_offsets["t_peak"]) == 1
        ]

    model = load_model(str(checkpoint_path))
    try:
        examples = load_examples(str(qtdb_dir), str(split_csv))
    except Exception as error:
        st.error(f"Could not load the QTDB validation examples: {error}")
        st.stop()
    if not examples:
        st.error("The validation split contains no single-T-peak examples.")
        st.stop()

    by_record: dict[str, list[QTDBBeatExample]] = {}
    for example in examples:
        by_record.setdefault(example.record_id, []).append(example)
    records = sorted(by_record)
    default_record = records.index("sel102") if "sel102" in records else 0
    selected_record = st.selectbox("Validation record", records, index=default_record)
    record_examples = by_record[selected_record]
    selected_beat_index = st.selectbox(
        "Annotated beat",
        range(len(record_examples)),
        format_func=lambda index: (
            f"beat {record_examples[index].beat_index} · "
            f"anchor {record_examples[index].anchor_sample} samples"
        ),
    )
    selected_example = record_examples[selected_beat_index]
    if "demo_result" not in st.session_state or st.session_state.get("demo_key") != (
        selected_record,
        selected_example.beat_index,
    ):
        st.session_state.pop("demo_result", None)
    compute = st.button("Compute IG and LIME explanations", type="primary")
    if compute:
        with st.spinner("Computing baseline comparisons…"):
            st.session_state["demo_result"] = _compute_explanations(model, selected_example)
            st.session_state["demo_key"] = (selected_record, selected_example.beat_index)

    st.markdown(
        f"**Source:** {selected_example.source_database}, record {selected_example.source_record_id}  "
        f"· **Leads:** {' / '.join(selected_example.lead_names)}  "
        f"· **Sampling:** {selected_example.sampling_rate_hz:g} Hz"
    )
    result = st.session_state.get("demo_result")
    if result is None:
        st.info("Choose a beat and compute the explanations to compare both references.")
        st.stop()

    metric_cols = st.columns(4)
    metric_cols[0].metric("Reference T peak", f"{result['reference_ms']:.1f} ms")
    metric_cols[1].metric("Predicted T peak", f"{result['prediction_ms']:.1f} ms")
    metric_cols[2].metric("Absolute error", f"{result['absolute_error_ms']:.1f} ms")
    metric_cols[3].metric("IG baseline top-bin overlap", f"{result['ig_baseline_jaccard']:.2f}")

    signal = result["normalized_input"]
    fs = selected_example.sampling_rate_hz
    times = np.arange(signal.shape[-1]) / fs * 1000.0 - LEFT_SECONDS * 1000.0
    waveform = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)
    for channel, lead in enumerate(selected_example.lead_names[:2]):
        waveform.add_trace(
            go.Scatter(x=times, y=signal[channel], mode="lines", name=lead, line={"width": 1}),
            row=channel + 1,
            col=1,
        )
        waveform.add_vline(
            x=result["reference_ms"], line_dash="dash", line_color="#16a34a", row=channel + 1, col=1
        )
        waveform.add_vline(
            x=result["prediction_ms"], line_dash="dot", line_color="#dc2626", row=channel + 1, col=1
        )
        waveform.update_yaxes(title_text=lead, row=channel + 1, col=1)
    waveform.update_xaxes(title_text="Time relative to selected-beat anchor (ms)", row=2, col=1)
    waveform.update_layout(height=420, margin={"l": 20, "r": 20, "t": 25, "b": 20}, legend={"orientation": "h"})
    st.plotly_chart(waveform, width="stretch")
    st.caption("Green dashed line: reference T peak · red dotted line: model prediction")

    width = result["segment_samples"]
    centers = np.asarray(
        [
            (start + min(start + width, signal.shape[-1])) / 2 / fs * 1000.0 - LEFT_SECONDS * 1000.0
            for start in range(0, signal.shape[-1], width)
        ]
    )
    explanation_cols = st.columns(2)
    for method, title, score_key, ylabel in (
        ("ig", "Integrated Gradients", "magnitude", "Attribution magnitude (ms)"),
        ("lime", "Temporal LIME", "signed_ms", "Surrogate coefficient (ms)")
    ):
        figure = go.Figure()
        for baseline, label, color in (
            ("zero", "Zero / mean-valued reference · primary", "#2563eb"),
            ("smooth", "Smoothed-waveform reference · sensitivity", "#ea580c"),
        ):
            values = result[f"{baseline}_{method}"][score_key]
            figure.add_trace(
                go.Scatter(
                    x=centers,
                    y=values,
                    mode="lines+markers",
                    name=label,
                    line={"width": 1.5, "color": color},
                    marker={"size": 4},
                )
            )
        figure.add_vline(x=result["reference_ms"], line_dash="dash", line_color="#16a34a")
        figure.add_vline(x=result["prediction_ms"], line_dash="dot", line_color="#dc2626")
        figure.update_layout(
            title=title,
            xaxis_title="Time relative to beat anchor (ms)",
            yaxis_title=ylabel,
            height=380,
            margin={"l": 20, "r": 15, "t": 50, "b": 25},
            legend={"orientation": "h", "y": -0.25},
        )
        explanation_cols[0 if method == "ig" else 1].plotly_chart(figure, width="stretch")

    quality_cols = st.columns(4)
    quality_cols[0].metric("IG completeness · zero", f"{result['zero_ig']['completeness_residual_ms']:.2f} ms")
    quality_cols[1].metric("IG completeness · smooth", f"{result['smooth_ig']['completeness_residual_ms']:.2f} ms")
    quality_cols[2].metric("LIME local R² · zero", f"{result['zero_lime']['weighted_r2']:.2f}")
    quality_cols[3].metric("LIME local R² · smooth", f"{result['smooth_lime']['weighted_r2']:.2f}")
    st.metric("LIME baseline top-bin overlap", f"{result['lime_baseline_jaccard']:.2f}")

    st.subheader("Inspect a perturbation")
    st.caption(
        "Replace the same number of time bins with the selected reference, once using the highest-ranked "
        "bins and once using a random matched set. Compare the resulting waveforms and model outputs."
    )
    probe_cols = st.columns(3)
    probe_method = probe_cols[0].selectbox("Attribution", ["Integrated Gradients", "Temporal LIME"])
    probe_baseline = probe_cols[1].selectbox("Replacement reference", ["Zero / mean", "Smoothed waveform"])
    probe_fraction = probe_cols[2].slider("Bins to replace", min_value=5, max_value=40, value=20, step=5)
    probe_seed = st.number_input("Random comparison seed", min_value=0, value=42, step=1)

    baseline_name = "zero" if probe_baseline == "Zero / mean" else "smooth"
    method_name = "ig" if probe_method == "Integrated Gradients" else "lime"
    attribution_scores = np.asarray(result[f"{baseline_name}_{method_name}"]["magnitude"])
    n_bins = len(attribution_scores)
    n_replace = max(1, math.ceil(n_bins * probe_fraction / 100.0))
    top_bins = np.argsort(np.abs(attribution_scores))[-n_replace:]
    random_bins = np.random.default_rng(int(probe_seed)).choice(n_bins, size=n_replace, replace=False)
    replacement = (
        np.zeros_like(signal)
        if baseline_name == "zero"
        else gaussian_filter1d(signal, sigma=12.0, axis=-1, mode="nearest")
    )

    def replace_bins(indices: np.ndarray) -> np.ndarray:
        perturbed = signal.copy()
        for index in indices:
            start = int(index) * width
            end = min(start + width, signal.shape[-1])
            perturbed[:, start:end] = replacement[:, start:end]
        return perturbed

    top_input = replace_bins(top_bins)
    random_input = replace_bins(random_bins)
    predictions = _predict_ms(
        model,
        np.stack([signal, top_input, random_input]).astype(np.float32),
        fs,
    )
    top_shift = float(abs(predictions[1] - predictions[0]))
    random_shift = float(abs(predictions[2] - predictions[0]))
    probe_metrics = st.columns(3)
    probe_metrics[0].metric("Original prediction", f"{predictions[0]:.1f} ms")
    probe_metrics[1].metric("Top-bin shift", f"{top_shift:.1f} ms", delta=f"{predictions[1] - predictions[0]:+.1f} ms")
    probe_metrics[2].metric(
        "Random-bin shift", f"{random_shift:.1f} ms", delta=f"{predictions[2] - predictions[0]:+.1f} ms"
    )

    probe_figure = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)
    for channel, lead in enumerate(selected_example.lead_names[:2]):
        probe_figure.add_trace(
            go.Scatter(x=times, y=signal[channel], mode="lines", name=f"{lead} · original", line={"width": 1}),
            row=channel + 1,
            col=1,
        )
        probe_figure.add_trace(
            go.Scatter(x=times, y=top_input[channel], mode="lines", name=f"{lead} · top-ranked", line={"width": 1}),
            row=channel + 1,
            col=1,
        )
        probe_figure.add_trace(
            go.Scatter(
                x=times,
                y=random_input[channel],
                mode="lines",
                name=f"{lead} · random",
                line={"width": 1, "dash": "dot"},
            ),
            row=channel + 1,
            col=1,
        )
        probe_figure.add_vline(
            x=result["reference_ms"], line_dash="dash", line_color="#16a34a", row=channel + 1, col=1
        )
        probe_figure.add_vline(
            x=float(predictions[0]), line_dash="dot", line_color="#dc2626", row=channel + 1, col=1
        )
        probe_figure.update_yaxes(title_text=lead, row=channel + 1, col=1)
    probe_figure.update_xaxes(title_text="Time relative to beat anchor (ms)", row=2, col=1)
    probe_figure.update_layout(
        title=f"{probe_method} · {probe_baseline} reference",
        height=500,
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
        legend={"orientation": "h"},
    )
    st.plotly_chart(probe_figure, width="stretch")
    st.caption(
        f"Replaced {n_replace} of {n_bins} bins ({probe_fraction}%). Green dashed: reference T peak; "
        "red dotted: original model prediction. Waveform replacement can create artificial signals, "
        "so prediction shifts are sensitivity probes, not causal or physiological evidence."
    )
    st.warning(
        "The maps can change with the reference baseline. Segment replacement creates artificial waveforms; "
        "these views show model sensitivity, not causal or clinical evidence."
    )


def main() -> None:
    _render()


if __name__ == "__main__":
    main()
