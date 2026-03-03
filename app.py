import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from io import BytesIO
from datetime import datetime

# PDF (ReportLab)
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader

st.set_page_config(page_title="Student Performance Intelligence System", layout="wide")


# -----------------------------
# Load assets
# -----------------------------
@st.cache_resource
def load_assets():
    clf = pickle.load(open("grade_classifier.pkl", "rb"))
    reg = pickle.load(open("cgpa_regressor.pkl", "rb"))
    kmeans = pickle.load(open("kmeans_model.pkl", "rb"))
    cluster_scaler = pickle.load(open("cluster_scaler.pkl", "rb"))
    model_features = pickle.load(open("model_features.pkl", "rb"))
    cluster_features = pickle.load(open("cluster_features.pkl", "rb"))
    population_stats = pickle.load(open("population_stats.pkl", "rb"))
    cluster_artifact = pickle.load(open("cluster_profile.pkl", "rb"))
    return clf, reg, kmeans, cluster_scaler, model_features, cluster_features, population_stats, cluster_artifact

clf, reg, kmeans, cluster_scaler, model_features, cluster_features, population_stats, cluster_artifact = load_assets()

cluster_profile_df = cluster_artifact["profile"]
cluster_labels = cluster_artifact["labels"]


# -----------------------------
# Display names
# -----------------------------
DISPLAY_NAMES = {
    "attendance": "Attendance (%)",
    "study_hours": "Study hours/day",
    "coding_hours": "Coding hours/week",
    "sleep_hours": "Sleep hours/day",
    "social_media_hours": "Social media hours/day",
    "stress": "Stress (1–5)",
    "motivation": "Motivation (1–5)",
    "backlogs": "Backlogs",
    "study_efficiency": "Study efficiency (study/(social+1))",
    "sleep_balance": "Sleep balance |sleep-7|",
    "academic_consistency": "Academic consistency (attendance*study)",
    "stress_index": "Stress index (stress/(sleep+1))"
}


# -----------------------------
# Helpers: benchmarking and plots
# -----------------------------
def pop_band(feature: str, value: float):
    med = float(population_stats["feature_median"][feature])
    q25 = float(population_stats["feature_q25"][feature])
    q75 = float(population_stats["feature_q75"][feature])

    if value < q25:
        band = "Below typical"
    elif value > q75:
        band = "Above typical"
    else:
        band = "Typical"

    if value > med:
        rel = "above median"
    elif value < med:
        rel = "below median"
    else:
        rel = "at median"

    return {"median": med, "q25": q25, "q75": q75, "band": band, "relative": rel}

def fig_to_png_bytes(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

def plot_benchmark_bullet(feature: str, user_value: float):
    b = pop_band(feature, user_value)

    xmin = min(b["q25"], b["median"], user_value)
    xmax = max(b["q75"], b["median"], user_value)
    if xmin == xmax:
        xmin -= 1
        xmax += 1

    fig, ax = plt.subplots(figsize=(6.2, 1.2))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Typical band (Q25-Q75)
    ax.hlines(0.5, b["q25"], b["q75"], linewidth=10)
    # Median marker
    ax.vlines(b["median"], 0.25, 0.75, linewidth=2)
    # User marker
    ax.vlines(user_value, 0.15, 0.85, linewidth=3)

    ax.text(b["q25"], 0.95, "Q25", ha="center", va="bottom", fontsize=9)
    ax.text(b["median"], 0.95, "Median", ha="center", va="bottom", fontsize=9)
    ax.text(b["q75"], 0.95, "Q75", ha="center", va="bottom", fontsize=9)
    ax.text(user_value, 0.05, "You", ha="center", va="bottom", fontsize=9)

    title = f"{DISPLAY_NAMES.get(feature, feature)} — {b['band']} ({b['relative']})"
    return fig, title, b

def normalize_by_q25q75(values: dict, feats: list):
    out = []
    for f in feats:
        q25 = float(population_stats["feature_q25"][f])
        q75 = float(population_stats["feature_q75"][f])
        v = float(values[f])
        denom = (q75 - q25) if (q75 - q25) != 0 else 1.0
        norm = (v - q25) / denom
        out.append(float(np.clip(norm, 0, 1)))
    return np.array(out)

def plot_radar(user_vals: dict, cluster_vals: dict, feats: list, title: str):
    labels = [DISPLAY_NAMES.get(f, f) for f in feats]
    user_norm = normalize_by_q25q75(user_vals, feats)
    cluster_norm = normalize_by_q25q75(cluster_vals, feats)

    N = len(labels)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    user_series = user_norm.tolist() + user_norm[:1].tolist()
    clus_series = cluster_norm.tolist() + cluster_norm[:1].tolist()

    fig = plt.figure(figsize=(6.4, 6.4))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=9)
    ax.set_ylim(0, 1)

    ax.plot(angles, user_series, linewidth=2, label="You")
    ax.fill(angles, user_series, alpha=0.12)

    ax.plot(angles, clus_series, linewidth=2, label="Cluster avg")
    ax.fill(angles, clus_series, alpha=0.08)

    ax.set_title(title, fontsize=12, pad=18)
    ax.legend(loc="upper right", bbox_to_anchor=(1.22, 1.12))
    return fig

def improvement_suggestions(inputs: dict):
    tips = []

    sm = inputs["social_media_hours"]
    sm_med = float(population_stats["feature_median"]["social_media_hours"])
    if sm > sm_med + 0.5:
        tips.append(("Reduce social media hours", "You are above the dataset median; reducing distraction typically improves study efficiency."))

    sh = inputs["study_hours"]
    sh_med = float(population_stats["feature_median"]["study_hours"])
    if sh < sh_med - 0.5:
        tips.append(("Increase focused study time", "You are below the dataset median; increasing consistent study hours usually improves performance."))

    sb = inputs["sleep_balance"]
    sb_med = float(population_stats["feature_median"]["sleep_balance"])
    if sb > sb_med + 0.3:
        tips.append(("Stabilize sleep around 7 hours", "Your sleep deviates more from 7 hours than typical; improving sleep regularity can reduce stress index."))

    si = inputs["stress_index"]
    si_med = float(population_stats["feature_median"]["stress_index"])
    if si > si_med + 0.1:
        tips.append(("Lower stress-to-sleep ratio", "Your stress index is above median; reducing stress and improving recovery can help."))

    att = inputs["attendance"]
    att_med = float(population_stats["feature_median"]["attendance"])
    if att < att_med - 3:
        tips.append(("Improve attendance consistency", "Your attendance is below median; attendance supports consistency and outcomes."))

    bl = inputs["backlogs"]
    if bl >= 1:
        tips.append(("Prioritize clearing backlogs", "Backlogs are a strong negative driver; clearing them is high-impact."))

    return tips[:4]

def get_cluster_profile(cluster_id: int):
    row = cluster_profile_df[cluster_profile_df["cluster"] == int(cluster_id)]
    if row.empty:
        return None
    return row.iloc[0].to_dict()

def safe_predict_proba(model, X_df):
    """
    Returns (labels, probs) or (None, None) if not supported.
    Works for Pipeline objects too.
    """
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_df)[0]
        labels = model.classes_ if hasattr(model, "classes_") else np.arange(len(probs))
        return labels, probs
    return None, None

def classify_probabilities(X_input: pd.DataFrame):
    # Works only if classifier supports predict_proba
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(X_input)[0]
        # class labels
        if hasattr(clf, "classes_"):
            labels = clf.classes_
        else:
            labels = np.arange(len(probs))
        return pd.DataFrame({"Class": labels, "Probability": probs}).sort_values("Probability", ascending=False)
    return None


# -----------------------------
# PDF generator
# -----------------------------
def make_pdf_report(
    name: str,
    current_cgpa: float,
    predicted_cgpa: float,
    predicted_grade: str,
    cluster_id: int,
    cluster_name: str,
    inputs: dict,
    tips: list,
    prob_df: pd.DataFrame | None,
    figs: list[tuple[str, bytes]]
) -> bytes:
    """
    figs: list of tuples -> (title, png_bytes)
    """
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4

    def header(title):
        c.setFont("Helvetica-Bold", 14)
        c.drawString(2*cm, height - 2.0*cm, title)
        c.setFont("Helvetica", 9)
        c.drawString(2*cm, height - 2.6*cm, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        c.line(2*cm, height - 2.75*cm, width - 2*cm, height - 2.75*cm)

    def wrap_text(text, x, y, max_width, line_height=12, font="Helvetica", size=10):
        c.setFont(font, size)
        words = text.split()
        line = ""
        cur_y = y
        for w in words:
            test = (line + " " + w).strip()
            if c.stringWidth(test, font, size) <= max_width:
                line = test
            else:
                c.drawString(x, cur_y, line)
                cur_y -= line_height
                line = w
        if line:
            c.drawString(x, cur_y, line)
            cur_y -= line_height
        return cur_y

    header("Student Performance Intelligence Report")

    y = height - 3.6*cm
    c.setFont("Helvetica-Bold", 11)
    c.drawString(2*cm, y, "Identity and summary")
    y -= 0.55*cm

    c.setFont("Helvetica", 10)
    summary = (
        f"Name: {name} | Current CGPA: {current_cgpa:.2f} | "
        f"Predicted CGPA: {predicted_cgpa:.2f} | Predicted Grade: {predicted_grade} | "
        f"Behavioral Segment: {cluster_name} (ID: {cluster_id})"
    )
    y = wrap_text(summary, 2*cm, y, width - 4*cm, line_height=12, size=10)
    y -= 0.2*cm

    # Gap
    gap = predicted_cgpa - current_cgpa
    c.setFont("Helvetica-Bold", 10)
    c.drawString(2*cm, y, "Performance gap interpretation")
    y -= 0.5*cm
    c.setFont("Helvetica", 10)
    if gap > 0.2:
        msg = f"Model indicates potential improvement of approximately {gap:.2f} CGPA points relative to the current CGPA."
    elif gap < -0.2:
        msg = f"Model indicates potential risk of decline of approximately {abs(gap):.2f} CGPA points relative to the current CGPA."
    else:
        msg = "Model indicates stability: predicted performance is close to the current CGPA."
    y = wrap_text(msg, 2*cm, y, width - 4*cm, line_height=12, size=10)
    y -= 0.2*cm

    # Inputs table (compact)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(2*cm, y, "Inputs (including engineered features)")
    y -= 0.45*cm

    c.setFont("Helvetica", 9)
    items = [(DISPLAY_NAMES.get(k, k), float(v)) for k, v in inputs.items()]
    # print in two columns
    col_x = [2*cm, 10.5*cm]
    col_y = y
    for i, (k, v) in enumerate(items):
        x = col_x[i % 2]
        if i % 2 == 0 and i > 0:
            col_y -= 0.42*cm
        c.drawString(x, col_y, f"{k}: {v:.3f}")
        if col_y < 3*cm:
            c.showPage()
            header("Student Performance Intelligence Report (continued)")
            col_y = height - 3.2*cm
    y = col_y - 0.8*cm

    # Recommendations
    if y < 6*cm:
        c.showPage()
        header("Student Performance Intelligence Report (continued)")
        y = height - 3.2*cm

    c.setFont("Helvetica-Bold", 10)
    c.drawString(2*cm, y, "Personalized recommendations")
    y -= 0.5*cm
    c.setFont("Helvetica", 10)
    if not tips:
        y = wrap_text("Inputs are close to typical values. Focus on consistency and maintaining current routines.", 2*cm, y, width - 4*cm)
    else:
        for idx, (t, why) in enumerate(tips, 1):
            y = wrap_text(f"{idx}. {t}: {why}", 2*cm, y, width - 4*cm)
            if y < 3.2*cm:
                c.showPage()
                header("Student Performance Intelligence Report (continued)")
                y = height - 3.2*cm

    # Probabilities
    if prob_df is not None:
        if y < 6*cm:
            c.showPage()
            header("Student Performance Intelligence Report (continued)")
            y = height - 3.2*cm

        c.setFont("Helvetica-Bold", 10)
        c.drawString(2*cm, y, "Grade class probabilities (classifier output)")
        y -= 0.5*cm
        c.setFont("Helvetica", 10)

        # Simple list
        for _, r in prob_df.iterrows():
            y = wrap_text(f"- {r['Class']}: {float(r['Probability']):.3f}", 2*cm, y, width - 4*cm)
            if y < 3.2*cm:
                c.showPage()
                header("Student Performance Intelligence Report (continued)")
                y = height - 3.2*cm

    # Charts pages
    for chart_title, png_bytes in figs:
        c.showPage()
        header("Charts and benchmarking")

        c.setFont("Helvetica-Bold", 12)
        c.drawString(2*cm, height - 3.4*cm, chart_title)

        img = ImageReader(BytesIO(png_bytes))
        # Fit image nicely
        img_w = width - 4*cm
        img_h = height - 6.0*cm
        c.drawImage(img, 2*cm, 2.5*cm, width=img_w, height=img_h, preserveAspectRatio=True, anchor="c")

    c.save()
    buf.seek(0)
    return buf.getvalue()


# -----------------------------
# UI
# -----------------------------
st.title("Student Performance Intelligence & Risk Prediction System")
st.caption("Personalized prediction, segmentation, benchmarking, and a downloadable report.")


# Sidebar
with st.sidebar:
    st.header("Student profile")
    name = st.text_input("Name", "Student")
    current_cgpa = st.number_input("Current CGPA", 0.0, 10.0, 8.0, step=0.01)

    st.header("Academic & lifestyle inputs")
    attendance = st.slider("Attendance (%)", 0, 100, 75)
    study_hours = st.slider("Study hours/day", 0, 12, 4)
    coding_hours = st.slider("Coding hours/week", 0, 60, 8)
    sleep_hours = st.slider("Sleep hours/day", 0, 12, 7)
    social_media_hours = st.slider("Social media hours/day", 0, 12, 3)
    stress = st.slider("Stress (1–5)", 1, 5, 3)
    motivation = st.slider("Motivation (1–5)", 1, 5, 3)
    backlogs = st.slider("Backlogs", 0, 10, 0)

    st.header("What-if simulator")
    st.caption("Optional: simulate improvements without changing your base inputs.")
    delta_study = st.slider("Change study hours/day", -3, 3, 0)
    delta_social = st.slider("Change social media hours/day", -3, 3, 0)
    delta_sleep = st.slider("Change sleep hours/day", -3, 3, 0)
    delta_att = st.slider("Change attendance (%)", -15, 15, 0)


# Engineered features
study_efficiency = study_hours / (social_media_hours + 1)
sleep_balance = abs(sleep_hours - 7)
academic_consistency = attendance * study_hours
stress_index = stress / (sleep_hours + 1)

inputs = {
    "attendance": float(attendance),
    "study_hours": float(study_hours),
    "coding_hours": float(coding_hours),
    "sleep_hours": float(sleep_hours),
    "social_media_hours": float(social_media_hours),
    "stress": float(stress),
    "motivation": float(motivation),
    "backlogs": float(backlogs),
    "study_efficiency": float(study_efficiency),
    "sleep_balance": float(sleep_balance),
    "academic_consistency": float(academic_consistency),
    "stress_index": float(stress_index),
}

X_input = pd.DataFrame([inputs])[model_features]

cluster_inputs = {
    "attendance": float(attendance),
    "study_hours": float(study_hours),
    "sleep_hours": float(sleep_hours),
    "social_media_hours": float(social_media_hours),
    "stress": float(stress),
    "motivation": float(motivation),
    "backlogs": float(backlogs),
}
X_cluster_input = pd.DataFrame([cluster_inputs])[cluster_features]


# Predictions
predicted_cgpa = float(reg.predict(X_input)[0])
predicted_grade = clf.predict(X_input)[0]
prob_df = classify_probabilities(X_input)

Xc_scaled = cluster_scaler.transform(X_cluster_input)
cluster_id = int(kmeans.predict(Xc_scaled)[0])
cluster_name = cluster_labels.get(cluster_id, f"Cluster {cluster_id}")
cluster_profile = get_cluster_profile(cluster_id)


# What-if scenario (apply deltas with bounds)
sim_att = float(np.clip(attendance + delta_att, 0, 100))
sim_study = float(np.clip(study_hours + delta_study, 0, 12))
sim_social = float(np.clip(social_media_hours + delta_social, 0, 12))
sim_sleep = float(np.clip(sleep_hours + delta_sleep, 0, 12))

sim_study_eff = sim_study / (sim_social + 1)
sim_sleep_bal = abs(sim_sleep - 7)
sim_acad_cons = sim_att * sim_study
sim_stress_idx = float(stress) / (sim_sleep + 1)

sim_inputs = dict(inputs)
sim_inputs.update({
    "attendance": sim_att,
    "study_hours": sim_study,
    "sleep_hours": sim_sleep,
    "social_media_hours": sim_social,
    "study_efficiency": sim_study_eff,
    "sleep_balance": sim_sleep_bal,
    "academic_consistency": sim_acad_cons,
    "stress_index": sim_stress_idx
})

X_sim = pd.DataFrame([sim_inputs])[model_features]
pred_cgpa_sim = float(reg.predict(X_sim)[0])
pred_grade_sim = clf.predict(X_sim)[0]
prob_df_sim = classify_probabilities(X_sim)


tips = improvement_suggestions(inputs)
gap = predicted_cgpa - float(current_cgpa)


# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Personalized results",
    "Benchmarking and charts",
    "What-if analysis",
    "Download report (PDF)"
])


with tab1:
    st.subheader(f"Personalized analysis for {name}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted CGPA", f"{predicted_cgpa:.2f}")
    c2.metric("Predicted grade category", str(predicted_grade))
    c3.metric("Behavioral segment", str(cluster_name))

    st.markdown("### Performance gap (current vs predicted)")
    if gap > 0.2:
        st.success(f"Model indicates potential improvement of approximately {gap:.2f} CGPA points compared with your current CGPA.")
    elif gap < -0.2:
        st.warning(f"Model indicates potential risk of decline of approximately {abs(gap):.2f} CGPA points compared with your current CGPA.")
    else:
        st.info("Model indicates stability: predicted CGPA is close to your current CGPA.")

    st.markdown("### Personalized recommendations")
    if tips:
        for i, (t, why) in enumerate(tips, 1):
            st.write(f"{i}. {t} — {why}")
    else:
        st.write("Your inputs are close to typical values. Focus on consistency and maintaining current routine.")

    if prob_df is not None:
        st.markdown("### Grade probability distribution (classifier)")
        st.dataframe(prob_df.reset_index(drop=True), use_container_width=True)


with tab2:
    st.subheader("Population benchmarking (you vs typical student)")
    focus_features = ["attendance", "study_hours", "sleep_hours", "social_media_hours", "stress_index", "backlogs"]

    chart_bytes = []
    for f in focus_features:
        fig, title, b = plot_benchmark_bullet(f, float(inputs[f]))
        st.markdown(f"**{title}** (Typical range: Q25={b['q25']:.2f} to Q75={b['q75']:.2f})")
        st.pyplot(fig)
        chart_bytes.append((title, fig_to_png_bytes(fig)))

    st.subheader("Radar comparison (you vs your cluster average)")
    if cluster_profile is not None:
        radar_feats = ["attendance", "study_hours", "sleep_hours", "social_media_hours", "stress", "motivation", "backlogs"]
        cluster_vals = {f: float(cluster_profile[f]) for f in radar_feats}
        fig_radar = plot_radar(inputs, cluster_vals, radar_feats, "Behavioral profile: you vs cluster mean (normalized)")
        st.pyplot(fig_radar)
        chart_bytes.append(("Radar: you vs cluster mean", fig_to_png_bytes(fig_radar)))
    else:
        st.info("Cluster profile not found for this cluster id.")


with tab3:
    st.subheader("What-if analysis (scenario simulation)")

    st.write(
        "This section simulates small changes to key inputs and recomputes predictions. "
        "Use it to demonstrate sensitivity of the model to behavior changes."
    )

    # Create simulated values
    sim_att = float(np.clip(attendance + delta_att, 0, 100))
    sim_study = float(np.clip(study_hours + delta_study, 0, 12))
    sim_social = float(np.clip(social_media_hours + delta_social, 0, 12))
    sim_sleep = float(np.clip(sleep_hours + delta_sleep, 0, 12))

    # Engineered features for simulation (must match training)
    sim_study_eff = sim_study / (sim_social + 1)
    sim_sleep_bal = abs(sim_sleep - 7)
    sim_acad_cons = sim_att * sim_study
    sim_stress_idx = float(stress) / (sim_sleep + 1)

    sim_inputs = dict(inputs)
    sim_inputs.update({
        "attendance": sim_att,
        "study_hours": sim_study,
        "sleep_hours": sim_sleep,
        "social_media_hours": sim_social,
        "study_efficiency": sim_study_eff,
        "sleep_balance": sim_sleep_bal,
        "academic_consistency": sim_acad_cons,
        "stress_index": sim_stress_idx
    })

    X_sim = pd.DataFrame([sim_inputs])[model_features]

    # Base vs simulated predictions
    base_cgpa = float(predicted_cgpa)
    sim_cgpa = float(reg.predict(X_sim)[0])

    base_grade = str(predicted_grade)
    sim_grade = str(clf.predict(X_sim)[0])

    st.markdown("### Scenario summary (what changed)")
    change_df = pd.DataFrame({
        "Feature": ["Attendance", "Study hours/day", "Sleep hours/day", "Social media hours/day"],
        "Base": [attendance, study_hours, sleep_hours, social_media_hours],
        "Simulated": [sim_att, sim_study, sim_sleep, sim_social],
        "Delta": [sim_att-attendance, sim_study-study_hours, sim_sleep-sleep_hours, sim_social-social_media_hours]
    })
    st.dataframe(change_df, use_container_width=True)

    st.markdown("### Prediction impact")
    c1, c2, c3 = st.columns(3)
    c1.metric("Base predicted CGPA", f"{base_cgpa:.2f}")
    c2.metric("Simulated predicted CGPA", f"{sim_cgpa:.2f}")
    c3.metric("Delta (Sim - Base)", f"{(sim_cgpa - base_cgpa):+.2f}")

    st.write(f"Base grade category: {base_grade}")
    st.write(f"Simulated grade category: {sim_grade}")

    # Probability shift (if supported)
    labels_base, probs_base = safe_predict_proba(clf, X_input)
labels_sim, probs_sim = safe_predict_proba(clf, X_sim)

if probs_base is not None and probs_sim is not None:
    base_df = pd.DataFrame({"Class": labels_base, "Base": probs_base})
    sim_df  = pd.DataFrame({"Class": labels_sim, "Simulated": probs_sim})

    merged = pd.merge(base_df, sim_df, on="Class", how="outer").fillna(0.0)

    # Force numeric
    merged["Base"] = pd.to_numeric(merged["Base"], errors="coerce").fillna(0.0)
    merged["Simulated"] = pd.to_numeric(merged["Simulated"], errors="coerce").fillna(0.0)

    merged["Delta"] = merged["Simulated"] - merged["Base"]
    merged = merged.sort_values("Simulated", ascending=False).reset_index(drop=True)

    st.markdown("### Grade probability shift (base vs simulated)")
    st.dataframe(merged, use_container_width=True)

    # Diagnostics so it can’t be “blank” without explanation
    dmin, dmax = float(merged["Delta"].min()), float(merged["Delta"].max())
    st.caption(f"Delta range: min={dmin:+.6f}, max={dmax:+.6f}")

    fig, ax = plt.subplots(figsize=(8.0, 3.2))
    ax.bar(merged["Class"].astype(str), merged["Delta"].astype(float))
    ax.axhline(0, linewidth=1)

    # Auto-scale y so small deltas are visible
    max_abs = max(abs(dmin), abs(dmax))
    if max_abs < 1e-6:
        # Changes are basically zero; show a message and a default range
        ax.set_ylim(-0.01, 0.01)
        st.info("Probabilities did not change meaningfully in this simulation (deltas ~ 0). Try larger changes.")
    else:
        pad = max_abs * 1.2
        ax.set_ylim(-pad, pad)

    ax.set_title("Probability change per class (Simulated - Base)")
    ax.set_ylabel("Delta probability")
    ax.set_xlabel("Class")

    st.pyplot(fig, clear_figure=True)
else:
    st.info("Probability chart not available because this classifier does not support predict_proba.")

with tab4:
    st.subheader("Download a full PDF report")

    st.write(
        "Generates a structured PDF report containing your inputs, model outputs, recommendations, "
        "probability breakdown (if available), and charts."
    )

    # Build charts for PDF (rebuild minimal set so it always exists)
    pdf_figs = []
    # bullets
    for f in ["attendance", "study_hours", "sleep_hours", "social_media_hours", "stress_index", "backlogs"]:
        fig, title, _ = plot_benchmark_bullet(f, float(inputs[f]))
        pdf_figs.append((title, fig_to_png_bytes(fig)))

    # radar
    if cluster_profile is not None:
        radar_feats = ["attendance", "study_hours", "sleep_hours", "social_media_hours", "stress", "motivation", "backlogs"]
        cluster_vals = {f: float(cluster_profile[f]) for f in radar_feats}
        fig_radar = plot_radar(inputs, cluster_vals, radar_feats, "Behavioral profile: you vs cluster mean (normalized)")
        pdf_figs.append(("Radar: you vs cluster mean", fig_to_png_bytes(fig_radar)))

    if st.button("Generate PDF report"):
        pdf_bytes = make_pdf_report(
            name=name,
            current_cgpa=float(current_cgpa),
            predicted_cgpa=predicted_cgpa,
            predicted_grade=str(predicted_grade),
            cluster_id=cluster_id,
            cluster_name=str(cluster_name),
            inputs=inputs,
            tips=tips,
            prob_df=prob_df,
            figs=pdf_figs
        )

        st.download_button(
            label="Download PDF report",
            data=pdf_bytes,
            file_name=f"{name.replace(' ', '_')}_student_report.pdf",
            mime="application/pdf"
        )

    st.markdown("---")
    