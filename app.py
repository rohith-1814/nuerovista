import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay

from alzheimers_mri.constants import CLASS_LABELS, CLASS_NAMES, MODEL_NAMES
from alzheimers_mri.data import preprocess_image, validate_mri
from alzheimers_mri.explainability import grad_cam, overlay_heatmap, shap_superpixel_values
from alzheimers_mri.history import read_history, record_prediction
from alzheimers_mri.models import checkpoint_path, load_checkpoint
from alzheimers_mri.reports import create_pdf_report

st.set_page_config(page_title="NeuroVista | MRI research dashboard", page_icon="🧠", layout="wide")
st.markdown(
    """
    <style>
    :root {
      --nv-ink: #172b3a;
      --nv-muted: #607181;
      --nv-brand: #146c70;
      --nv-line: #e4ebef;
      --nv-surface: #ffffff;
      --nv-canvas: #f4f7f8;
    }
    .stApp { background: var(--nv-canvas); color: var(--nv-ink); }
    .block-container { max-width: 1480px; padding: 2rem 2.2rem 3rem; }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] {
      background: #fff;
      border-right: 1px solid var(--nv-line);
    }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.2rem; }
    .hero {
      display: flex; justify-content: space-between; align-items: center; gap: 1rem;
      padding: 1.55rem 1.8rem; border: 1px solid #dce9e9; border-radius: 18px;
      background: linear-gradient(115deg, #fff 15%, #edf7f6 100%);
      margin: .4rem 0 1.2rem;
    }
    .brand { display: flex; gap: 1rem; align-items: center; }
    .brand-mark {
      display: grid; place-items: center; width: 48px; height: 48px;
      border-radius: 14px; color: #fff; background: #146c70;
      font-size: 1.45rem; font-weight: 700;
    }
    .hero h1 { margin: 0; color: var(--nv-ink); font-size: 1.75rem; letter-spacing: -.035em; }
    .hero p { margin: .2rem 0 0; color: var(--nv-muted); font-size: .92rem; }
    .hero-tag {
      border: 1px solid #cfe3df; border-radius: 999px; padding: .45rem .8rem;
      color: #245b55; background: #f5fbf9; font-size: .78rem;
      font-weight: 650; white-space: nowrap;
    }
    .notice {
      padding: .8rem 1rem; border: 1px solid #f1dfb8; border-left: 4px solid #ce8b24;
      background: #fffaf0; border-radius: 10px; margin: .8rem 0 1.1rem;
      color: #694d1d; font-size: .9rem;
    }
    [data-testid="stMetric"] {
      background: var(--nv-surface); border: 1px solid var(--nv-line);
      border-radius: 12px; padding: .8rem 1rem;
    }
    [data-testid="stMetricLabel"] { color: var(--nv-muted); }
    [data-testid="stTabs"] [role="tablist"] {
      gap: .45rem; border-bottom: 1px solid var(--nv-line);
    }
    [data-testid="stTabs"] button[role="tab"] {
      border-radius: 9px 9px 0 0; padding: .7rem 1rem; color: var(--nv-muted);
    }
    [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
      color: var(--nv-brand); border-bottom-color: var(--nv-brand); font-weight: 650;
    }
    [data-testid="stFileUploader"] section {
      border: 1px dashed #b6c8cc; border-radius: 12px; background: #fbfdfd;
    }
    .section-eyebrow {
      margin-bottom: .3rem; color: var(--nv-brand); text-transform: uppercase;
      letter-spacing: .1em; font-size: .72rem; font-weight: 750;
    }
    .status-ready, .status-needed {
      display: inline-block; border-radius: 999px; padding: .3rem .65rem;
      font-size: .76rem; font-weight: 650;
    }
    .status-ready { color: #176044; background: #e9f6ef; }
    .status-needed { color: #78571d; background: #fff4dc; }
    div.stButton > button[kind="primary"] {
      background: var(--nv-brand); border-color: var(--nv-brand);
    }
    div.stButton > button { border-radius: 9px; font-weight: 600; }
    @media (max-width: 720px) {
      .block-container { padding: 1.2rem 1rem 2rem; }
      .hero { padding: 1.1rem; align-items: flex-start; }
      .hero-tag { display: none; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading trained classifier…")
def load_model_cached(path, model_name, file_bytes=None, checkpoint_version=None):
    if file_bytes is None:
        return load_checkpoint(path, model_name, torch.device("cpu"))[0]
    import io

    return load_checkpoint(io.BytesIO(file_bytes), model_name, torch.device("cpu"))[0]


def read_metrics(model_name):
    path = Path("models") / f"{model_name}_metrics.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


st.markdown(
    '<div class="hero"><div class="brand"><div class="brand-mark">N</div>'
    '<div><div class="section-eyebrow">Brain MRI research workspace</div>'
    "<h1>NeuroVista</h1><p>Classification · Explainability · Model evaluation</p>"
    '</div></div><div class="hero-tag">RESEARCH WORKSPACE</div></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="notice"><strong>Research use only.</strong> This tool is not a '
    "medical device and cannot diagnose Alzheimer's disease. Predictions must not "
    "guide treatment or replace review by a qualified clinician.</div>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown('<div class="section-eyebrow">Workspace</div>', unsafe_allow_html=True)
    st.header("Analysis setup")
    st.caption("Configure the model and study details for this session.")
    patient_id = st.text_input("Patient / study ID (optional)", max_chars=64)
    selected_model = st.selectbox("Classifier", MODEL_NAMES, format_func=lambda value: value.replace("_", " ").title())
    selected_checkpoint = checkpoint_path(selected_model)
    if selected_checkpoint.is_file():
        st.markdown('<span class="status-ready">● Model ready</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-needed">● Trained weights required</span>', unsafe_allow_html=True)
    st.caption("A trained checkpoint is required to generate predictions.")
    checkpoint_upload = st.file_uploader(
        "Use a checkpoint from another location", type=["pth", "pt"], key=f"checkpoint-{selected_model}"
    )
    st.divider()
    st.markdown('<div class="section-eyebrow">System</div>', unsafe_allow_html=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.caption(f"**Inference device** · {device.type.upper()}")
    st.caption("Scans and study history remain on this machine.")
    st.caption("Accepted input · PNG, JPG, TIFF, BMP")

analysis_tab, evaluation_tab, history_tab, guide_tab = st.tabs(
    ["01  Analyze scan", "02  Model performance", "03  Study history", "Guide"]
)

with analysis_tab:
    st.markdown('<div class="section-eyebrow">Imaging workspace</div>', unsafe_allow_html=True)
    st.markdown("### Upload and review")
    st.caption("Analyze a single 2D brain MRI image with a trained model. DICOM series and 3D volumes are not supported.")
    left, right = st.columns([1.2, 1])
    with left:
        uploaded_image = st.file_uploader(
            "Choose a brain MRI image",
            type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
            help="Upload one 2D image. Basic image checks do not confirm MRI modality.",
        )
        image = None
        if uploaded_image:
            try:
                image = Image.open(uploaded_image).convert("RGB")
            except (OSError, ValueError) as error:
                st.error(f"Could not read the uploaded image: {error}")
        if image:
            st.image(image, caption=f"{uploaded_image.name} · {image.width} × {image.height}", use_container_width=True)
            valid, message = validate_mri(image)
            (st.success if valid else st.error)(message)
    with right:
        st.markdown("#### Analysis result")
        if image is None:
            with st.container(border=True):
                st.markdown("**Your analysis will appear here**")
                st.write("1. Upload a 2D scan.")
                st.write("2. Review image checks and model output.")
                st.write("3. Explore explanations and download a report.")
                if not selected_checkpoint.is_file() and not checkpoint_upload:
                    st.caption("A trained checkpoint is required. See the Guide tab for setup.")
        elif not valid:
            st.warning("Analysis is blocked because basic image checks failed.")
        else:
            checkpoint_file = checkpoint_upload.getvalue() if checkpoint_upload else None
            model_file = checkpoint_path(selected_model)
            if checkpoint_file is None and not model_file.exists():
                st.warning(f"No trained checkpoint found at `{model_file}`. Follow the setup guide to train the model.")
            else:
                checkpoint_version = (
                    hashlib.sha256(checkpoint_file).hexdigest()
                    if checkpoint_file is not None
                    else str(model_file.stat().st_mtime_ns)
                )
                try:
                    model = load_model_cached(
                        str(model_file),
                        selected_model,
                        checkpoint_file,
                        checkpoint_version,
                    )
                    tensor = preprocess_image(image).to(device)
                    model = model.to(device).eval()
                    with torch.no_grad():
                        probabilities = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
                    predicted_index = int(np.argmax(probabilities))
                    confidence = float(probabilities[predicted_index])
                    predicted_label = CLASS_LABELS[CLASS_NAMES[predicted_index]]
                    warning = confidence < 0.60
                    st.metric("Predicted category", predicted_label, f"{confidence:.1%} probability")
                    if warning:
                        st.warning("Low confidence: treat this output as especially uncertain.")
                    st.caption("Softmax probabilities are not calibrated clinical confidence or a measure of diagnostic certainty.")
                    probability_frame = pd.DataFrame(
                        {
                            "Category": [CLASS_LABELS[name] for name in CLASS_NAMES],
                            "Probability": probabilities,
                        }
                    ).set_index("Category")
                    st.bar_chart(probability_frame)
                    image_signature = hashlib.sha256(uploaded_image.getvalue()).hexdigest()
                    explanation_key = (image_signature, selected_model, predicted_index)
                    if st.session_state.get("explanation_key") != explanation_key:
                        st.session_state.pop("grad_cam_image", None)
                        st.session_state.pop("shap_plot", None)
                        st.session_state["explanation_key"] = explanation_key
                    grad_image = None
                    explain_col1, explain_col2 = st.columns(2)
                    with explain_col1:
                        if st.button("Generate Grad-CAM", use_container_width=True):
                            try:
                                with st.spinner("Generating Grad-CAM…"):
                                    heatmap = grad_cam(model, selected_model, tensor, predicted_index)
                                    grad_image = overlay_heatmap(image, heatmap)
                                    st.session_state["grad_cam_image"] = grad_image
                            except (RuntimeError, ValueError, IndexError) as error:
                                st.error(f"Grad-CAM could not be generated: {error}")
                    with explain_col2:
                        if st.button("Generate SHAP", use_container_width=True):
                            try:
                                with st.spinner("Estimating SHAP region contributions…"):
                                    shap_heatmap = shap_superpixel_values(model, image, device)
                                    fig, ax = plt.subplots()
                                    ax.imshow(image.resize((224, 224)))
                                    plot = ax.imshow(shap_heatmap, cmap="coolwarm", alpha=0.55)
                                    ax.axis("off")
                                    fig.colorbar(plot, ax=ax, fraction=0.046, pad=0.04, label="Relative SHAP value")
                                    st.session_state["shap_plot"] = fig
                            except (ImportError, RuntimeError, ValueError, IndexError) as error:
                                st.error(f"SHAP explanation could not be generated: {error}")
                    grad_image = st.session_state.get("grad_cam_image")
                    if grad_image is not None:
                        st.image(grad_image, caption="Grad-CAM · highlights influential regions, not disease lesions", use_container_width=True)
                    shap_plot = st.session_state.get("shap_plot")
                    if shap_plot is not None:
                        st.pyplot(shap_plot, clear_figure=False)
                    record_key = (
                        uploaded_image.name,
                        image_signature,
                        selected_model,
                        patient_id,
                        predicted_index,
                        confidence,
                    )
                    if st.session_state.get("last_record_key") != record_key:
                        record = record_prediction(
                            patient_id,
                            uploaded_image.name,
                            selected_model,
                            predicted_label,
                            confidence,
                            {CLASS_LABELS[name]: float(probabilities[i]) for i, name in enumerate(CLASS_NAMES)},
                        )
                        st.session_state["last_record"] = record
                        st.session_state["last_record_key"] = record_key
                    record = st.session_state["last_record"]
                    pdf = create_pdf_report(
                        record,
                        [
                            [CLASS_LABELS[name], f"{probabilities[i]:.1%}"]
                            for i, name in enumerate(CLASS_NAMES)
                        ],
                        grad_image,
                    )
                    st.download_button(
                        "Download prediction report (PDF)",
                        data=pdf,
                        file_name="mri_research_report.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
                except (ValueError, RuntimeError, OSError, KeyError) as error:
                    st.error(f"Unable to run this checkpoint: {error}")

with evaluation_tab:
    st.markdown('<div class="section-eyebrow">Validation</div>', unsafe_allow_html=True)
    st.subheader("Held-out evaluation")
    st.caption("Metrics are only shown when generated by the training workflow from a supplied test split.")
    comparison = []
    for model_name in MODEL_NAMES:
        metrics = read_metrics(model_name)
        if metrics is not None:
            comparison.append(
                {
                    "Model": model_name.replace("_", " ").title(),
                    "Accuracy": metrics["accuracy"],
                    "Macro precision": metrics["precision_macro"],
                    "Macro recall": metrics["recall_macro"],
                    "Macro F1": metrics["f1_macro"],
                }
            )
    if comparison:
        st.markdown("#### Model comparison")
        st.dataframe(
            pd.DataFrame(comparison).set_index("Model").style.format("{:.1%}"),
            use_container_width=True,
        )
    metric_columns = st.columns(2)
    for index, model_name in enumerate(MODEL_NAMES):
        metrics = read_metrics(model_name)
        with metric_columns[index]:
            st.markdown(f"#### {model_name.replace('_', ' ').title()}")
            if metrics is None:
                st.info(f"No held-out test metrics found for {model_name}.")
                continue
            st.caption(f"Evaluation split: {metrics.get('evaluation_split', 'unspecified')}")
            cards = st.columns(4)
            for card, label, key in zip(
                cards,
                ("Accuracy", "Precision", "Recall", "F1"),
                ("accuracy", "precision_macro", "recall_macro", "f1_macro"),
            ):
                card.metric(label, f"{metrics[key]:.1%}")
            matrix = np.asarray(metrics["confusion_matrix"])
            figure, axis = plt.subplots(figsize=(5, 4))
            ConfusionMatrixDisplay(matrix, display_labels=[CLASS_LABELS[name] for name in CLASS_NAMES]).plot(
                ax=axis, cmap="Blues", colorbar=False, xticks_rotation=35
            )
            axis.set_title("Held-out confusion matrix")
            st.pyplot(figure)
            plt.close(figure)

with history_tab:
    st.markdown('<div class="section-eyebrow">Local records</div>', unsafe_allow_html=True)
    st.subheader("Local patient / study history")
    st.caption("History is stored locally as a CSV without encryption. Use de-identified study data.")
    history = read_history()
    if history.empty:
        st.info("No predictions have been recorded in this workspace.")
    else:
        visible = history.drop(columns=["probabilities"], errors="ignore").sort_values(
            "timestamp", ascending=False
        )
        st.dataframe(visible, use_container_width=True, hide_index=True)
        st.download_button(
            "Export history (CSV)",
            history.to_csv(index=False).encode("utf-8"),
            file_name="patient_history.csv",
            mime="text/csv",
        )

with guide_tab:
    st.markdown('<div class="section-eyebrow">Documentation</div>', unsafe_allow_html=True)
    st.subheader("Getting started")
    st.markdown(
        """
        1. Install Python 3.10+ and the packages in `requirements.txt`.
        2. By default, training downloads `Falah/Alzheimer_MRI` from Hugging Face. Its source labels are mapped to the four categories below.
        3. Train one or both models. Training downloads the torchvision ImageNet initialization on first use; the resulting MRI classifier weights are saved locally.
        4. Run `streamlit run app.py` and analyze a supported 2D image.

        ```bash
        python -m pip install -r requirements.txt
        python train.py --models resnet50 efficientnet_b0 --epochs 10
        streamlit run app.py
        ```

        For local ImageFolder data instead, use `python train.py --source local --data data`.

        ```text
        data/
          train/{non_demented,very_mild,mild,moderate}/
          val/{non_demented,very_mild,mild,moderate}/
          test/{non_demented,very_mild,mild,moderate}/  # recommended; held out
        ```

        The Hugging Face dataset provides a train/test split but does not document patient IDs, so patient-level leakage cannot be ruled out. Metrics are not clinical validation. Classifier labels reflect dataset labels and are not a clinical staging system. For custom data, split by patient (not by image). Training uses pretrained ImageNet initialization and must be validated for the intended MRI acquisition and population.
        """
    )
    st.subheader("Important limitations")
    st.markdown(
        """
        - This research prototype is not a diagnostic tool, medical device, or substitute for a clinician.
        - The app accepts individual 2D image files, not DICOM, full MRI volumes, clinical records, or brain-region segmentation.
        - Image checks are basic plausibility checks; they cannot authenticate MRI modality, detect all artifacts, or assess scan suitability.
        - Grad-CAM and SHAP are approximate model-behavior explanations, not causal evidence or proof of pathology.
        - Softmax scores can be overconfident and are not calibrated probabilities of disease.
        - History is stored locally in `patient_history.csv` without encryption or access controls. Use synthetic/de-identified data; do not enter identifiable health information.
        """
    )
