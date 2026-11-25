"""OvaScope web interface (Streamlit).

    streamlit run app/streamlit_app.py

The UI never loads TensorFlow itself: every prediction, upload and retraining
request goes through the FastAPI service (OVASCOPE_API_URL, default
http://localhost:8000), so what you see is what the API really does.
Dataset insights and evaluation results are read from the artifacts/ folder.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import altair as alt
import pandas as pd
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.config import load_config  # noqa: E402

CFG = load_config()
API = os.environ.get("OVASCOPE_API_URL", "http://localhost:8000").rstrip("/")
ARTIFACTS = CFG["paths"]["artifacts_dir"]
st.set_page_config(page_title="OvaScope", page_icon="🔬", layout="wide")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def api_get(path, **kwargs):
    try:
        response = requests.get(f"{API}{path}", timeout=10, **kwargs)
        return response.json() if response.ok else None
    except requests.RequestException:
        return None


def admin_headers():
    key = st.session_state.get("admin_key", "")
    return {"X-API-Key": key} if key else {}


def show_error(response):
    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        detail = response.text
    st.error(f"{response.status_code}: {detail}")


def format_duration(seconds):
    seconds = int(seconds or 0)
    return f"{seconds // 3600}h {seconds % 3600 // 60}m {seconds % 60}s"


@st.cache_data
def load_classes():
    return api_get("/classes") or json.loads((CFG["paths"]["processed_dir"] / "class_map.json").read_text())


@st.cache_data
def load_image_stats():
    path = ARTIFACTS / "metrics" / "crop_image_stats.csv"
    return pd.read_csv(path) if path.exists() else None


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def page_dashboard():
    st.title("🔬 OvaScope MLOps")
    st.caption("Classifies a microscope crop of **one** parasite egg into one of 11 classes "
               "(Chula-ParasiteEgg-11). Research prototype - not a clinically validated diagnostic tool.")
    health = api_get("/health")
    metrics = api_get("/metrics")
    status = api_get("/retrain/status")
    if health is None:
        st.error(f"API unreachable at {API}. Start it with `uvicorn api.main:app`.")
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Application status", "Online" if health["status"] == "ok" else "Degraded")
    c2.metric("Model availability", "Loaded" if health["model_loaded"] else "Not loaded")
    c3.metric("Current model version", health["model_version"] or "-")
    c4.metric("Service uptime (this process)", format_duration(health["service_uptime_seconds"]))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Prediction requests (all time)", metrics["prediction_requests_total"])
    c2.metric("Average response time", f"{metrics['average_latency_ms']:.1f} ms" if metrics["average_latency_ms"] else "-")
    c3.metric("New images waiting", status["pending_new_images"])
    latest = status.get("latest_job")
    c4.metric("Latest retraining", latest["status"] if latest else "none yet")
    if status["retraining_recommended"]:
        st.warning(f"Retraining recommended: {status['pending_new_images']} new labelled images "
                   f"(threshold {status['trigger_threshold']}).")
    st.caption("Uptime counts from when the current API process started; no historical uptime is stored. "
               "Model availability is separate: the service can be up while no model is loaded.")


def page_predict():
    st.title("Predict a parasite egg")
    st.write("Upload a crop centred on a **single** egg (JPEG/PNG/BMP/TIFF). "
             "Full microscope photos with several eggs are out of scope.")
    file = st.file_uploader("Image", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"])
    if not file:
        return
    left, right = st.columns([1, 1.3])
    left.image(file.getvalue(), caption=file.name, use_container_width=True)
    if right.button("Predict", type="primary"):
        try:
            response = requests.post(f"{API}/predict", files={"file": (file.name, file.getvalue())}, timeout=60)
        except requests.RequestException as error:
            right.error(f"API unreachable: {error}")
            return
        if response.status_code == 422:   # the egg gate rejected the image
            detail = response.json().get("detail", {})
            right.warning(f"**Not a parasite egg image.** {detail.get('message', '')}")
            right.caption(f"Egg score {detail.get('egg_score', 0):.3f} (needs ≥ {detail.get('threshold', 0):.3f}). "
                          "Tip: use a microscope photo and crop tightly around one egg.")
            return
        if not response.ok:
            with right:
                show_error(response)
            return
        result = response.json()
        right.subheader(result["predicted_class"])
        right.metric("Confidence (model score)", f"{result['confidence']:.1%}")
        top = pd.DataFrame(result["top_k"])
        right.altair_chart(alt.Chart(top).mark_bar().encode(
            x=alt.X("confidence:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
            y=alt.Y("class_name:N", sort="-x", title=None)).properties(height=140), use_container_width=True)
        right.caption(f"Model version: `{result['model_version']}` · class id {result['predicted_class_id']} · "
                      f"official label “{result['original_label']}”")
        check = result.get("input_check", {})
        if check.get("enabled"):
            right.caption(f"Input check passed · egg score {check['egg_score']:.3f} (threshold {check['threshold']:.3f})")
        right.info(result["note"])


def page_insights():
    st.title("Dataset insights")
    stats = load_image_stats()
    if stats is None:
        st.warning("Run `python scripts/audit_dataset.py` to compute image statistics.")
        return
    classes = {c["folder"]: c["display_name"] for c in load_classes()}
    stats["class"] = stats["folder"].map(classes)
    st.write(f"Measured on all **{len(stats):,}** egg crops (train {sum(stats.split == 'train'):,}, "
             f"validation {sum(stats.split == 'val'):,}, held-out test {sum(stats.split == 'test'):,}).")

    st.subheader("Class distribution")
    counts = stats.groupby(["class", "split"]).size().reset_index(name="crops")
    st.altair_chart(alt.Chart(counts).mark_bar().encode(
        x=alt.X("crops:Q"), y=alt.Y("class:N", title=None), color="split:N",
        tooltip=["class", "split", "crops"]).properties(height=330), use_container_width=True)
    per_class_train = stats[stats.split == "train"].groupby("class").size()
    st.caption(f"Training crops per class: {per_class_train.min()}–{per_class_train.max()} "
               f"(ratio {per_class_train.max() / per_class_train.min():.3f}). The classes are balanced, "
               "so no class weighting is used.")

    samples = ARTIFACTS / "plots" / "audit_crop_samples.png"
    if samples.exists():
        st.subheader("Representative crops (one row per class, ids 0–10 top to bottom)")
        st.image(str(samples), use_container_width=True)

    def eta_squared(feature):
        """Share of a feature's variance explained by the class (0 = unrelated, 1 = fully determined)."""
        group_means = stats.groupby("class")[feature].transform("mean")
        return ((group_means - stats[feature].mean()) ** 2).sum() / ((stats[feature] - stats[feature].mean()) ** 2).sum()

    def boxplot(feature, title, log=False):
        scale = alt.Scale(type="log") if log else alt.Scale(zero=False)
        return alt.Chart(stats).mark_boxplot(outliers=False).encode(
            x=alt.X(f"{feature}:Q", title=title, scale=scale), y=alt.Y("class:N", title=None),
            color=alt.Color("class:N", legend=None)).properties(height=330)

    st.subheader("Feature 1 — Brightness (mean grey level)")
    st.altair_chart(boxplot("brightness", "mean brightness (0–255)"), use_container_width=True)
    by_split = stats.groupby("split")["brightness"].median()
    st.markdown(
        f"- Brightness ranges from {stats.brightness.min():.0f} to {stats.brightness.max():.0f} "
        f"(median {stats.brightness.median():.0f}).\n"
        f"- Class explains only **{eta_squared('brightness'):.0%}** of its variance: the wide boxes overlap, so "
        "brightness mostly reflects the microscope, lamp and camera, not the parasite.\n"
        f"- Median brightness: train {by_split['train']:.0f}, val {by_split['val']:.0f}, test {by_split['test']:.0f} — "
        "the held-out test photos are somewhat darker.\n"
        "- **Implication:** a model that keyed on brightness would be unreliable across microscopes; "
        "training uses random brightness/contrast augmentation to make it robust to illumination.")

    st.subheader("Feature 2 — Crop size and aspect ratio")
    st.altair_chart(boxplot("width", "crop side length in pixels (log scale)", log=True), use_container_width=True)
    st.markdown(
        f"- Crops are square by construction (aspect ratio {stats.aspect_ratio.min():.2f}–{stats.aspect_ratio.max():.2f}), "
        f"side {stats.width.min()}–{stats.width.max()} px; {(stats.width < 224).mean():.0%} are smaller than the "
        "224 px model input and get upsampled.\n"
        f"- Size is strongly class-dependent (class explains **{eta_squared('width'):.0%}** of the variance), "
        "matching real differences in egg size, but it also depends on magnification and camera resolution.\n"
        "- **Implication:** squaring before resizing preserves egg shape (no stretching). Because every crop is "
        "resized to 224×224, absolute size is *not* available to the model; it must rely on shape and texture.")

    st.subheader("Feature 3 — Texture/focus (Laplacian variance) and colour saturation")
    left, right = st.columns(2)
    left.altair_chart(boxplot("sharpness", "Laplacian variance (log scale)", log=True), use_container_width=True)
    right.altair_chart(boxplot("saturation", "mean HSV saturation (0–255)"), use_container_width=True)
    sharp_split = stats.groupby("split")["sharpness"].median()
    st.markdown(
        f"- Laplacian variance measures edge strength: low values mean blurred/out-of-focus crops. It spans "
        f"{stats.sharpness.min():.0f}–{stats.sharpness.max():.0f}, and class explains only "
        f"**{eta_squared('sharpness'):.1%}** of it — focus quality is a property of the capture, not of the species.\n"
        f"- Median sharpness: train {sharp_split['train']:.0f}, test {sharp_split['test']:.0f}: the test set is "
        "noticeably sharper, a distribution difference to keep in mind when reading test results.\n"
        f"- Saturation is moderately class-related (**{eta_squared('saturation'):.0%}** of variance), consistent "
        "with some eggs being naturally yellow-brown (bile-stained) and others nearly colourless.\n"
        "- **Limitation:** both are single-number summaries; they ignore where in the image the texture is "
        "(egg vs. background debris).")


def page_upload():
    st.title("Upload labelled training images")
    st.write("Images are validated (type, size, decodable, known label, not a duplicate, not a protected test "
             "image) and stored **inside the SQLite database** for the next retraining run.")
    classes = load_classes()
    mode = st.radio("Labelling method", ["One class for all selected images", "ZIP with one folder per class"])
    if mode.startswith("One"):
        choice = st.selectbox("Class label", classes, format_func=lambda c: f"{c['class_id']:02d} · {c['display_name']}")
        files = st.file_uploader("Images", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"], accept_multiple_files=True)
        data = {"label": str(choice["class_id"])}
    else:
        st.caption("Folder names must be class folders such as `00_ascaris_lumbricoides/` "
                   "(see the list below). The whole ZIP is rejected if any folder is unknown.")
        st.code("\n".join(c["folder"] for c in classes))
        files = st.file_uploader("ZIP archive", type=["zip"], accept_multiple_files=False)
        files = [files] if files else []
        data = {}
    if files and st.button("Upload", type="primary"):
        payload = [("files", (f.name, f.getvalue())) for f in files]
        response = requests.post(f"{API}/upload", files=payload, data=data, headers=admin_headers(), timeout=300)
        if not response.ok:
            show_error(response)
        else:
            result = response.json()
            st.success(f"Saved {result['saved_to_database']} image(s) to the database.")
            if result["accepted"]:
                st.dataframe(pd.DataFrame(result["accepted"]), use_container_width=True)
            if result["rejected"]:
                st.error(f"{len(result['rejected'])} file(s) rejected")
                st.dataframe(pd.DataFrame(result["rejected"]), use_container_width=True)
    records = api_get("/uploads")
    if records:
        counts = records["counts"]
        st.subheader("Database records")
        c1, c2, c3 = st.columns(3)
        c1.metric("Validated images available for retraining", counts["pending"])
        c2.metric("Already used in retraining", counts["used"])
        c3.metric("Total stored", counts["total"])
        if records["records"]:
            st.dataframe(pd.DataFrame(records["records"]), use_container_width=True, hide_index=True)


def page_train():
    st.title("Training and retraining")
    st.markdown(
        "**Initial training** builds a model from the full prepared dataset (GPU recommended; run "
        "`python -m src.train --model both --select` or the notebook). "
        "**Retraining** continues training the *currently active* saved model with the newly uploaded images "
        "plus a replay sample of original training images, then evaluates the candidate on the validation set "
        "and activates it only if its macro-F1 is not worse than the active model's (within the configured tolerance).")
    status = api_get("/retrain/status")
    if status is None:
        st.error("API unreachable.")
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Active model", status["active_model"] or "-")
    c2.metric("New validated images", status["pending_new_images"])
    c3.metric("Automatic trigger threshold", status["trigger_threshold"])
    if status["retraining_recommended"]:
        st.warning("Threshold reached: retraining is recommended.")
    if st.button("Retrain Model", type="primary", disabled=status["job_running"]):
        response = requests.post(f"{API}/retrain", headers=admin_headers(), timeout=30)
        if response.ok:
            st.session_state["job_id"] = response.json()["job_id"]
            st.success(f"Retraining job {st.session_state['job_id']} started.")
        else:
            show_error(response)
    job_id = st.session_state.get("job_id") or (status["latest_job"] or {}).get("id")
    if job_id:
        job = api_get(f"/retrain/status/{job_id}")
        if job:
            st.subheader(f"Job {job_id}: {job['status']}")
            st.write(f"Progress: {job['progress']}")
            if job["status"] in ("queued", "running"):
                st.button("Refresh status")
            if job.get("promotion_decision"):
                (st.success if job["promoted"] else st.warning)(job["promotion_decision"])
            if job.get("candidate_metrics"):
                table = pd.DataFrame({"active model (before)": job["active_metrics"],
                                      "candidate (retrained)": job["candidate_metrics"]})
                st.dataframe(table, use_container_width=True)
            if job.get("error"):
                st.error(job["error"])
    st.subheader("Previous jobs")
    jobs = api_get("/retrain/jobs") or []
    if jobs:
        st.dataframe(pd.DataFrame(jobs)[["id", "status", "trigger", "created_at", "finished_at", "n_new_images",
                                         "base_version", "candidate_version", "promoted"]],
                     use_container_width=True, hide_index=True)
    st.subheader("Model versions")
    versions = api_get("/model/versions") or []
    if versions:
        rows = [{"version": v["version"], "active": v["active"], "kind": v["kind"], "parent": v["parent_version"],
                 "created": v["created_at"], "val accuracy": (v["validation_metrics"] or {}).get("accuracy"),
                 "val macro F1": (v["validation_metrics"] or {}).get("macro_f1"),
                 "test macro F1": (v.get("test_metrics") or {}).get("macro_f1"), "size MB": v["model_size_mb"]}
                for v in versions]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        target = st.selectbox("Activate a version", [v["version"] for v in versions])
        c1, c2 = st.columns(2)
        if c1.button("Activate selected version"):
            r = requests.post(f"{API}/model/activate/{target}", headers=admin_headers(), timeout=60)
            st.success(r.json()) if r.ok else show_error(r)
        if c2.button("Roll back to previous version"):
            r = requests.post(f"{API}/model/rollback", headers=admin_headers(), timeout=60)
            st.success(r.json()) if r.ok else show_error(r)


def page_monitoring():
    st.title("Monitoring")
    health = api_get("/health")
    metrics = api_get("/metrics")
    if not health or not metrics:
        st.error(f"API unreachable at {API}")
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("API health", health["status"])
    c2.metric("Model version", health["model_version"] or "-")
    c3.metric("Uptime (this process)", format_duration(health["service_uptime_seconds"]))
    c4.metric("Served by", health["instance"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Prediction requests", metrics["prediction_requests_total"])
    c2.metric("Successful", metrics["prediction_requests_successful"])
    c3.metric("Failed", metrics["prediction_requests_failed"])
    recent = metrics["recent_latency_ms"]
    c4.metric("Recent p95 latency", f"{recent['p95']:.0f} ms" if recent["p95"] else "-")
    if metrics["recent_requests"]:
        frame = pd.DataFrame(metrics["recent_requests"])
        st.subheader("Recent prediction activity")
        st.altair_chart(alt.Chart(frame.reset_index()).mark_line(point=True).encode(
            x=alt.X("timestamp:T"), y=alt.Y("latency_ms:Q", title="latency (ms)"),
            tooltip=["timestamp", "latency_ms", "status_code", "model_version"]).properties(height=250),
            use_container_width=True)
        st.dataframe(frame, use_container_width=True, hide_index=True)
    st.subheader("Model evaluation results (labelled data)")
    st.caption("Runtime monitoring above has no ground truth. Predictive performance below comes from "
               "labelled held-out data, not from production traffic.")
    for name, label in [("final_test_metrics.json", "Held-out test set (offline evaluation)"),
                        ("production_evaluation.json", "Deployed API evaluated on labelled test crops")]:
        path = ARTIFACTS / "metrics" / name
        if path.exists():
            data = json.loads(path.read_text())
            st.markdown(f"**{label}**")
            st.json({k: data[k] for k in data if k not in ("per_class", "confusion_matrix")}, expanded=False)
    matrix = ARTIFACTS / "plots" / "final_test_confusion_matrix.png"
    if matrix.exists():
        st.image(str(matrix), caption="Confusion matrix — held-out test set")


PAGES = {"Dashboard": page_dashboard, "Predict": page_predict, "Dataset insights": page_insights,
         "Upload training data": page_upload, "Train & retrain": page_train, "Monitoring": page_monitoring}

with st.sidebar:
    st.header("OvaScope")
    page = st.radio("Page", list(PAGES))
    st.text_input("Admin API key", type="password", key="admin_key",
                  help="Needed for upload, retraining and model activation when OVASCOPE_ADMIN_KEY is set on the API.")
    st.caption(f"API: {API}")

PAGES[page]()
