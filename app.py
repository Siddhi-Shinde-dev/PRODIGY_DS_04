import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Twitter Sentiment Analyser",
    page_icon="🐦",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background-color: #0f1117; }

    .metric-card {
        background: #1a1d27;
        border: 1px solid #2a2d3e;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
    }
    .metric-card .label {
        font-size: 13px;
        color: #8b8fa8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 6px;
    }
    .metric-card .value {
        font-size: 32px;
        font-weight: 700;
        color: #ffffff;
    }
    .metric-card .sub {
        font-size: 12px;
        color: #555870;
        margin-top: 4px;
    }

    .sentiment-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
    }

    .section-title {
        font-size: 18px;
        font-weight: 700;
        color: #e2e4f0;
        margin-bottom: 4px;
    }
    .section-sub {
        font-size: 13px;
        color: #6b6f87;
        margin-bottom: 20px;
    }

    div[data-testid="stSelectbox"] label,
    div[data-testid="stTextInput"] label {
        color: #9ba0bb !important;
        font-size: 13px !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .stDataFrame { border-radius: 10px; overflow: hidden; }

    hr { border-color: #2a2d3e; margin: 28px 0; }
</style>
""", unsafe_allow_html=True)

# ── Colour palette ────────────────────────────────────────────────────────────
COLORS = {
    "Positive":   "#22c55e",
    "Neutral":    "#3b82f6",
    "Negative":   "#ef4444",
    "Irrelevant": "#a855f7",
}
BG = "#1a1d27"
TEXT = "#e2e4f0"
GRID = "#2a2d3e"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor":   BG,
    "axes.edgecolor":   GRID,
    "axes.labelcolor":  TEXT,
    "xtick.color":      TEXT,
    "ytick.color":      TEXT,
    "text.color":       TEXT,
    "grid.color":       GRID,
    "grid.linewidth":   0.5,
})

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data(path="twitter_training.csv"):
    col_names = ["ID", "Entity", "Sentiments", "Content"]
    df = pd.read_csv(path, names=col_names)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df["Sentiments"] = df["Sentiments"].str.strip()
    df["Entity"]     = df["Entity"].str.strip()
    return df

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## 🐦 Twitter Sentiment Analyser")
st.markdown('<p style="color:#6b6f87;margin-top:-10px;">Explore brand sentiment patterns across 74 k+ tweets</p>', unsafe_allow_html=True)
st.markdown("---")

# ── File upload or default ────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload `twitter_training.csv`  (or leave empty to use default)", type="csv")

try:
    if uploaded:
        col_names = ["ID", "Entity", "Sentiments", "Content"]
        df = pd.read_csv(uploaded, names=col_names)
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)
        df["Sentiments"] = df["Sentiments"].str.strip()
        df["Entity"]     = df["Entity"].str.strip()
    else:
        df = load_data()
except Exception as e:
    st.error(f"Could not load data: {e}")
    st.stop()

# ── KPIs ──────────────────────────────────────────────────────────────────────
total      = len(df)
entities   = df["Entity"].nunique()
top_sent   = df["Sentiments"].value_counts().idxmax()
top_brand  = df["Entity"].value_counts().idxmax()

c1, c2, c3, c4 = st.columns(4)
for col, label, val, sub in [
    (c1, "Total Tweets",    f"{total:,}",    "after cleaning"),
    (c2, "Brands / Entities", f"{entities}",  "unique entities"),
    (c3, "Top Sentiment",   top_sent,        "most frequent class"),
    (c4, "Most Tweeted Brand", top_brand,    "by tweet volume"),
]:
    col.markdown(f"""
    <div class="metric-card">
        <div class="label">{label}</div>
        <div class="value">{val}</div>
        <div class="sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("---")

# ── Tab layout ────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🏷️ Brand Analysis", "🔍 Live Analyser", "📄 Raw Data"])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — Overview
# ════════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-title">Sentiment Distribution</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Proportion of each sentiment class across all tweets</div>', unsafe_allow_html=True)

    sentiment_counts = df["Sentiments"].value_counts()

    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        bars = ax.bar(
            sentiment_counts.index,
            sentiment_counts.values,
            color=[COLORS.get(s, "#888") for s in sentiment_counts.index],
            width=0.55,
            zorder=3,
        )
        ax.yaxis.grid(True, zorder=0)
        ax.set_xlabel("Sentiment", labelpad=8)
        ax.set_ylabel("Tweet Count", labelpad=8)
        ax.set_title("All Sentiments", pad=12, fontweight="bold")
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 200,
                    f"{h:,}", ha="center", va="bottom", fontsize=10, color=TEXT)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_right:
        fig2, ax2 = plt.subplots(figsize=(4.5, 4.5))
        wedges, texts, autotexts = ax2.pie(
            sentiment_counts.values,
            labels=sentiment_counts.index,
            autopct="%1.1f%%",
            colors=[COLORS.get(s, "#888") for s in sentiment_counts.index],
            startangle=140,
            wedgeprops=dict(width=0.6),
        )
        for t in texts:    t.set_color(TEXT)
        for t in autotexts: t.set_color("#0f1117"); t.set_fontsize(10)
        ax2.set_title("Share by Class", pad=12, fontweight="bold")
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close()

    st.markdown("---")
    st.markdown('<div class="section-title">Top 10 Most Tweeted Brands</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Overall tweet volume per entity</div>', unsafe_allow_html=True)

    top10 = df["Entity"].value_counts().head(10)
    fig3, ax3 = plt.subplots(figsize=(9, 3.5))
    ax3.barh(top10.index[::-1], top10.values[::-1], color="#3b82f6", height=0.6, zorder=3)
    ax3.xaxis.grid(True, zorder=0)
    ax3.set_xlabel("Tweet Count", labelpad=8)
    ax3.set_title("Top 10 Brands by Volume", pad=12, fontweight="bold")
    for i, (val, name) in enumerate(zip(top10.values[::-1], top10.index[::-1])):
        ax3.text(val + 50, i, f"{val:,}", va="center", fontsize=9, color=TEXT)
    fig3.tight_layout()
    st.pyplot(fig3)
    plt.close()

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — Brand Analysis
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">Brand-Level Sentiment Breakdown</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Select one or more brands to compare sentiment distributions</div>', unsafe_allow_html=True)

    all_entities = sorted(df["Entity"].unique())
    selected = st.multiselect("Choose brands", all_entities, default=all_entities[:3])

    if not selected:
        st.info("Pick at least one brand from the list above.")
    else:
        brand_df = df[df["Entity"].isin(selected)]

        # Stacked bar
        pivot = (
            brand_df.groupby(["Entity", "Sentiments"])
            .size()
            .unstack(fill_value=0)
        )
        sent_order = [s for s in ["Positive", "Neutral", "Negative", "Irrelevant"] if s in pivot.columns]
        pivot = pivot[sent_order]

        fig4, ax4 = plt.subplots(figsize=(max(7, len(selected) * 1.4), 4.5))
        bottom = pd.Series([0] * len(pivot), index=pivot.index)
        for sent in sent_order:
            ax4.bar(pivot.index, pivot[sent], bottom=bottom,
                    label=sent, color=COLORS[sent], width=0.55, zorder=3)
            bottom += pivot[sent]
        ax4.yaxis.grid(True, zorder=0)
        ax4.set_ylabel("Tweet Count", labelpad=8)
        ax4.set_title("Sentiment by Brand (Stacked)", pad=12, fontweight="bold")
        ax4.tick_params(axis="x", rotation=25)
        patches = [mpatches.Patch(color=COLORS[s], label=s) for s in sent_order]
        ax4.legend(handles=patches, loc="upper right", framealpha=0.2)
        fig4.tight_layout()
        st.pyplot(fig4)
        plt.close()

        # Per-brand pie charts
        if len(selected) <= 6:
            st.markdown("#### Sentiment share per brand")
            cols = st.columns(min(len(selected), 3))
            for idx, brand in enumerate(selected):
                bdata = brand_df[brand_df["Entity"] == brand]["Sentiments"].value_counts()
                with cols[idx % 3]:
                    fig5, ax5 = plt.subplots(figsize=(3.5, 3.5))
                    ax5.pie(bdata.values,
                            labels=bdata.index,
                            autopct="%1.0f%%",
                            colors=[COLORS.get(s, "#888") for s in bdata.index],
                            startangle=140,
                            wedgeprops=dict(width=0.55))
                    ax5.set_title(brand, fontsize=11, fontweight="bold", pad=8)
                    fig5.tight_layout()
                    st.pyplot(fig5)
                    plt.close()

# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — Live Analyser
# ════════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">Live Tweet Sentiment</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Type any text to analyse its sentiment using TextBlob</div>', unsafe_allow_html=True)

    user_text = st.text_area("Enter tweet / text here", height=120,
                             placeholder="e.g. The new Xbox update is absolutely brilliant!")

    if st.button("Analyse Sentiment", width='stretch'):
        if user_text.strip():
            blob = TextBlob(user_text)
            polarity    = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity

            if polarity > 0.1:
                label, color = "Positive", COLORS["Positive"]
            elif polarity < -0.1:
                label, color = "Negative", COLORS["Negative"]
            else:
                label, color = "Neutral", COLORS["Neutral"]

            r1, r2, r3 = st.columns(3)
            r1.markdown(f"""<div class="metric-card">
                <div class="label">Sentiment</div>
                <div class="value" style="color:{color}">{label}</div>
            </div>""", unsafe_allow_html=True)
            r2.markdown(f"""<div class="metric-card">
                <div class="label">Polarity</div>
                <div class="value">{polarity:.3f}</div>
                <div class="sub">-1 (negative) → +1 (positive)</div>
            </div>""", unsafe_allow_html=True)
            r3.markdown(f"""<div class="metric-card">
                <div class="label">Subjectivity</div>
                <div class="value">{subjectivity:.3f}</div>
                <div class="sub">0 (objective) → 1 (subjective)</div>
            </div>""", unsafe_allow_html=True)

            # Polarity gauge bar
            st.markdown("<br>", unsafe_allow_html=True)
            fig6, ax6 = plt.subplots(figsize=(7, 1.2))
            ax6.barh(["Polarity"], [1], color=GRID, height=0.4)
            ax6.barh(["Polarity"], [polarity], color=color, height=0.4)
            ax6.set_xlim(-1, 1)
            ax6.axvline(0, color=TEXT, linewidth=0.8, linestyle="--")
            ax6.set_title(f"Polarity score: {polarity:.3f}", fontsize=11, pad=8)
            ax6.yaxis.set_visible(False)
            fig6.tight_layout()
            st.pyplot(fig6)
            plt.close()
        else:
            st.warning("Please enter some text first.")

    st.markdown("---")
    st.markdown("**How it works:** TextBlob computes a *polarity* score (−1 to +1) and a *subjectivity* score (0 to 1) from lexical patterns. Polarity > 0.1 → Positive, < −0.1 → Negative, else Neutral.")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — Raw Data
# ════════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">Dataset Preview</div>', unsafe_allow_html=True)

    col_f1, col_f2 = st.columns(2)
    filter_entity    = col_f1.selectbox("Filter by Brand", ["All"] + sorted(df["Entity"].unique()))
    filter_sentiment = col_f2.selectbox("Filter by Sentiment", ["All"] + sorted(df["Sentiments"].unique()))

    view = df.copy()
    if filter_entity    != "All": view = view[view["Entity"]     == filter_entity]
    if filter_sentiment != "All": view = view[view["Sentiments"] == filter_sentiment]

    st.markdown(f"**{len(view):,} rows** matching filters")
    st.dataframe(view[["Entity", "Sentiments", "Content"]].reset_index(drop=True), width='stretch', height=420)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<p style="text-align:center;color:#555870;font-size:12px;">Built by Siddhi Shinde · PRODIGY_DS_04 · Twitter Sentiment Analysis</p>', unsafe_allow_html=True)