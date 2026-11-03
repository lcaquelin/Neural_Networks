"""Museum guide: recognise the style and the genre of a painting from a photo (Streamlit app).

Run from the project root:
    uv run streamlit run streamlit_app.py

The models are the ResNet18 of `06_01_resnet_style` and `06_02_resnet_genre`; their weights must first be
saved by the last cell of each notebook (data/models/resnet18_style_app.pt and resnet18_genre_app.pt).
The app shows the three most likely styles bis of the photo with their descriptions
(module `museum_guide.style_bis`), and its three most likely genres. Probabilities are not shown:
the most likely comes first.
"""

import streamlit as st
from PIL import Image

from museum_guide.predict import (
    GENRE_CHECKPOINT,
    STYLE_CHECKPOINT,
    load_model,
    predict_top_genres,
    predict_top_style_bis,
)
from museum_guide.style_bis import STYLE_BIS

TOP_K = 3
ROMAN = ["I", "II", "III"]  # ranks written in Roman numerals, as in a museum catalogue

# Readable names for the styles bis and the styles (the keys of the code contain underscores)
SPECIAL_NAMES = {
    "Baroque_Rococo": "Baroque & Rococo",
    "Impressionism_PostImpressionism": "Impressionism & Post-Impressionism",
    "Expressionism_Fauvism": "Expressionism & Fauvism",
    "Post_Impressionism": "Post-Impressionism",
    "Mannerism_Late_Renaissance": "Mannerism (Late Renaissance)",
    "Naive_Art_Primitivism": "Naive Art & Primitivism",
    "Action_painting": "Action Painting",
    "Ukiyo_e": "Ukiyo-e",
}


def readable(name):
    """Return the name to display for a style bis or a style."""
    return SPECIAL_NAMES.get(name, name.replace("_", " "))


def readable_genre(name):
    """Return the name to display for a genre, e.g. "genre_painting" -> "Genre painting"."""
    return name.replace("_", " ").capitalize()


# --- Page and visual identity: ivory paper, black ink, gold accents, serif type ---
st.set_page_config(page_title="The Museum Guide", page_icon="🖼️", layout="wide")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;500;600&family=EB+Garamond:ital,wght@0,400;0,500;1,400&display=swap');

    :root {
        --ivory: #F6F1E7;
        --paper: #FCFAF5;
        --ink: #1F1B16;
        --muted: #6E6457;
        --gold: #A8854E;
        --line: #DDD1BC;
    }
    .stApp { background: var(--ivory); color: var(--ink); }
    [data-testid="stHeader"] { background: transparent; }
    html, body, [class*="css"], .stMarkdown, p, li, label { font-family: 'EB Garamond', Georgia, serif; color: var(--ink); }
    h1, h2, h3 { font-family: 'Cormorant Garamond', Georgia, serif !important; color: var(--ink) !important; font-weight: 500 !important; }

    .museum-header { text-align: center; padding: 1.2rem 0 0.4rem 0; }
    .museum-kicker { font-family: 'EB Garamond', serif; letter-spacing: 0.35em; text-transform: uppercase;
                     font-size: 0.8rem; color: var(--gold); }
    .museum-title { font-family: 'Cormorant Garamond', serif; font-size: 3rem; font-weight: 500; margin: 0.2rem 0; }
    .museum-subtitle { font-family: 'EB Garamond', serif; font-style: italic; color: var(--muted); font-size: 1.15rem; }
    .gold-rule { width: 120px; height: 1px; background: var(--gold); margin: 1rem auto 1.6rem auto; }

    .notice { background: var(--paper); border: 1px solid var(--gold); border-left: 4px solid var(--gold);
              padding: 1rem 1.3rem; margin: 0.4rem 0 1.6rem 0; font-size: 1.02rem; line-height: 1.5; }
    .notice-title { font-family: 'Cormorant Garamond', serif; font-size: 1.25rem; font-weight: 600;
                    letter-spacing: 0.06em; margin-bottom: 0.4rem; }

    .genres { text-align: center; font-size: 1.15rem; line-height: 1.5; margin-top: 0.8rem; }
    .genres-label { font-variant: small-caps; letter-spacing: 0.08em; color: var(--muted); }
    .intro { font-size: 1.2rem; line-height: 1.55; margin: 0.2rem 0 1.2rem 0; }
    .card { background: var(--paper); border: 1px solid var(--line); padding: 1.3rem 1.5rem; margin-bottom: 1.1rem; }
    .card-rank { font-family: 'Cormorant Garamond', serif; color: var(--gold); font-size: 1.1rem; letter-spacing: 0.2em; }
    .card-name { font-family: 'Cormorant Garamond', serif; font-size: 1.9rem; font-weight: 500; margin: 0.1rem 0 0.7rem 0; }
    .card-text { font-size: 1.08rem; line-height: 1.55; }
    .sub-title { font-variant: small-caps; letter-spacing: 0.08em; color: var(--muted); margin: 0.9rem 0 0.3rem 0; }
    .sub-style { border-top: 1px solid var(--line); padding: 0.55rem 0 0.2rem 0; font-size: 1.02rem; line-height: 1.5; }
    .sub-style b { font-family: 'Cormorant Garamond', serif; font-size: 1.15rem; font-weight: 600; }

    /* Serif type everywhere (Streamlit sets its own sans-serif font otherwise) */
    .stApp p, .stApp li, .stApp label, .stApp button, .notice, .intro, .card, .genres {
        font-family: 'EB Garamond', Georgia, serif !important;
    }
    [data-testid="stAppDeployButton"] { display: none; }

    /* The two buttons ("upload" = the button of the file uploader): same size, thin gold frame, capitals */
    .stButton button, [data-testid="stFileUploaderDropzone"] button {
        width: 100%; min-height: 3.2rem; background: var(--paper); color: var(--ink);
        border: 1px solid var(--gold); border-radius: 0;
    }
    .stButton button p, [data-testid="stFileUploaderDropzone"] button::after {
        font-size: 1rem; letter-spacing: 0.18em; text-transform: uppercase;
    }
    .stButton button:hover, [data-testid="stFileUploaderDropzone"] button:hover {
        background: var(--ink); color: var(--paper); border-color: var(--ink);
    }
    .stButton button:disabled, .stButton button:disabled:hover {
        background: transparent; color: var(--muted); border-color: var(--line); cursor: not-allowed;
    }
    .stButton button:disabled p { color: var(--muted); }
    /* The file uploader is reduced to a single button that opens the file explorer */
    [data-testid="stFileUploaderDropzone"] { background: transparent; border: none; padding: 0; }
    [data-testid="stFileUploaderDropzone"] > span, [data-testid="stFileUploaderDropzone"] > div { width: 100%; }
    [data-testid="stFileUploaderDropzoneInstructions"], [data-testid="stFileChips"] > div:first-child,
    [data-testid="stFileUploaderDropzone"] button > div { display: none; }
    [data-testid="stFileUploaderDropzone"] button::after { content: "Import a photo"; }
    /* Once a photo is loaded, the same button lets the visitor choose another one */
    [data-testid="stFileChips"] { display: block; width: 100%; padding: 0; }
    [data-testid="stFileChips"] button::after { content: "Import another photo"; }

    /* The photo, framed like a painting */
    [data-testid="stImage"] img { border: 12px solid var(--paper); outline: 1px solid var(--gold);
                                  box-shadow: 0 6px 18px rgba(31, 27, 22, 0.15); border-radius: 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="museum-header">
        <div class="museum-kicker">Museum of Fine Arts</div>
        <div class="museum-title">The Museum Guide</div>
        <div class="museum-subtitle">Import a photo of a painting and discover its style</div>
        <div class="gold-rule"></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Word of caution, always visible ---
st.markdown(
    """
    <div class="notice">
        <div class="notice-title">Before you begin</div>
        Art is not only a matter of colours, shapes or a painter's manner: a style or a genre also grows out
        of ideas, a context and intentions, whose subtleties artificial intelligence may not fully grasp.<br>
        Although tested and relatively reliable, the analysis of the painting and its results are generated
        by artificial intelligence and may contain errors. Feel free to ask the museum staff for more
        information.
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_models():
    """Load the two models once, then reuse them for every photo (Streamlit cache)."""
    return load_model(STYLE_CHECKPOINT), load_model(GENRE_CHECKPOINT)


missing = [path for path in (STYLE_CHECKPOINT, GENRE_CHECKPOINT) if not path.exists()]
if missing:
    st.error(
        "Model weights not found: " + ", ".join(str(path) for path in missing) + ". Run the notebooks "
        "06_01_resnet_style and 06_02_resnet_genre, including their last cell, which saves the weights for this app."
    )
    st.stop()

# --- The two buttons: import a photo (opens the file explorer) and take a photo (not available yet) ---
col_import, col_camera = st.columns(2)
with col_import:
    uploaded = st.file_uploader("Import a photo", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
with col_camera:
    st.button("Take a photo", disabled=True, help="Coming soon", use_container_width=True)

if uploaded is not None:
    image = Image.open(uploaded)
    (style_model, style_names, style_size), (genre_model, genre_names, genre_size) = get_models()
    top_styles = predict_top_style_bis(style_model, style_names, image, style_size, k=TOP_K)
    top_genres = predict_top_genres(genre_model, genre_names, image, genre_size, k=TOP_K)

    col_image, col_results = st.columns([2, 3], gap="large")
    with col_image:
        st.image(image, caption="Your photo")
        genres = " or ".join(f"<b>{readable_genre(name)}</b>" for name in top_genres)
        st.markdown(
            f"<div class='genres'><span class='genres-label'>Likely genres:</span> {genres}</div>",
            unsafe_allow_html=True,
        )

    with col_results:
        names = [f"<b>{readable(name)}</b>" for name in top_styles]
        st.markdown(
            f"<div class='intro'>The three most likely styles of this painting are: "
            f"{names[0]}, {names[1]} and {names[2]}. Here is a description of each:</div>",
            unsafe_allow_html=True,
        )

        for rank, name in zip(ROMAN, top_styles):
            group = STYLE_BIS[name]
            # Sub-styles are listed only when the style bis gathers several styles
            sub_styles = ""
            if len(group["styles"]) > 1:
                items = "".join(
                    f"<div class='sub-style'><b>{readable(style)}</b>. {text}</div>"
                    for style, text in group["styles"].items()
                )
                sub_styles = f"<div class='sub-title'>The styles it brings together</div>{items}"
            st.markdown(
                f"""
                <div class="card">
                    <div class="card-rank">{rank}</div>
                    <div class="card-name">{readable(name)}</div>
                    <div class="card-text">{group["description"]}</div>
                    {sub_styles}
                </div>
                """,
                unsafe_allow_html=True,
            )
