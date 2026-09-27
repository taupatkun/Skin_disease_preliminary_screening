import streamlit as st
from fastai.learner import load_learner
from fastai.vision.core import PILImage
import pathlib
import sys
import torch
import fastai
import torchvision
import PIL
import os
import traceback
import hashlib
import pickle
import types
import gdown

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Derma - AI Skin Disease Screening",
    page_icon="🩺",
    layout="centered"
)
# =========================================================
# TITLE
# =========================================================
st.title("🩺 Derma")
st.subheader(
    "AI-Based Preliminary Skin Disease Screening"
)
st.write(
    "By Ruengsit Matachaiyasit & Teerapat Sittichottithikun"
)
st.write(
    "Upload an image of a skin condition to receive "
    "an AI-based preliminary screening result."
)
st.warning(
    "⚠️ This result is an AI-based preliminary "
    "screening and is NOT a medical diagnosis. "
    "The model may produce incorrect results. "
    "Please consult a qualified healthcare "
    "professional for proper evaluation."
)
# =========================================================
# DISEASE INFORMATION
# =========================================================
disease_info = {
    "BA- cellulitis": {
        "name": "Cellulitis",
        "description":
            "A bacterial infection affecting the deeper layers of the skin.",
        "causes": [
            "Usually caused by bacteria entering through a break in the skin.",
            "Cuts, wounds, insect bites, or other skin injuries can provide an entry point for bacteria."
        ],
        "what_to_do": [
            "Keep the affected area clean.",
            "Avoid touching or scratching the affected area.",
            "Seek medical evaluation, especially if redness or swelling is spreading.",
            "Follow treatment recommended by a healthcare professional."
        ]
    },
    "BA-impetigo": {
        "name": "Impetigo",
        "description":
            "A contagious bacterial skin infection that commonly affects the surface of the skin.",
        "causes": [
            "Usually caused by Staphylococcus or Streptococcus bacteria.",
            "Can spread through direct contact with an infected person.",
            "Can also spread through contaminated personal items."
        ],
        "what_to_do": [
            "Keep the affected area clean.",
            "Avoid scratching or touching the affected area.",
            "Avoid sharing towels, clothing, or other personal items.",
            "Seek medical advice for appropriate treatment."
        ]
    },
    "FU-nail-fungus": {
        "name": "Nail Fungus",
        "description":
            "A fungal infection affecting the fingernails or toenails.",
        "causes": [
            "Caused by a fungal infection.",
            "Can spread through contact with contaminated surfaces or objects.",
            "Warm and moist environments can promote fungal growth."
        ],
        "what_to_do": [
            "Keep nails clean and dry.",
            "Avoid sharing nail clippers or other personal items.",
            "Keep feet dry if toenails are affected.",
            "Seek medical advice for appropriate treatment."
        ]
    },
    "FU-ringworm": {
        "name": "Ringworm",
        "description":
            "A fungal infection that can cause circular or ring-shaped skin lesions.",
        "causes": [
            "Caused by a fungal infection.",
            "Can spread through contact with infected people or animals.",
            "Can also spread through contaminated clothing, towels, or other objects."
        ],
        "what_to_do": [
            "Keep the affected area clean and dry.",
            "Avoid scratching the affected area.",
            "Avoid sharing towels, clothing, or personal items.",
            "Seek medical advice if the condition does not improve."
        ]
    },
    "FU-athlete-foot": {
        "name": "Athlete's Foot",
        "description":
            "A fungal infection that commonly affects the skin of the feet.",
        "causes": [
            "Caused by a fungal infection.",
            "Warm and moist environments can promote fungal growth.",
            "Can spread through contaminated floors, shoes, socks, or towels."
        ],
        "what_to_do": [
            "Keep the feet clean and dry.",
            "Change socks regularly.",
            "Avoid sharing towels, socks, or footwear.",
            "Seek medical advice if symptoms persist or worsen."
        ]
    },
    "PA-cutaneous-larva-migrans": {
        "name": "Cutaneous Larva Migrans",
        "description":
            "A skin condition caused by larvae that migrate through the skin.",
        "causes": [
            "Usually associated with contact with soil or sand contaminated with animal hookworm larvae.",
            "The larvae can enter the skin through direct contact with contaminated ground."
        ],
        "what_to_do": [
            "Avoid further contact with potentially contaminated soil or sand.",
            "Keep the affected skin clean.",
            "Seek medical evaluation for appropriate treatment."
        ]
    },
    "VI-shingles": {
        "name": "Shingles",
        "description":
            "A viral infection that can cause a painful skin rash.",
        "causes": [
            "Caused by reactivation of the varicella-zoster virus.",
            "The virus can remain inactive in the body after a previous chickenpox infection."
        ],
        "what_to_do": [
            "Seek medical evaluation promptly.",
            "Avoid scratching or touching the rash.",
            "Avoid close contact with people who may be vulnerable to varicella infection.",
            "Follow medical advice and prescribed treatment."
        ]
    },
    "VI-chickenpox": {
        "name": "Chickenpox",
        "description":
            "A contagious viral infection that commonly causes an itchy rash and fluid-filled spots.",
        "causes": [
            "Caused by the varicella-zoster virus.",
            "Can spread through respiratory droplets and close contact."
        ],
        "what_to_do": [
            "Avoid close contact with others while contagious.",
            "Avoid scratching the rash.",
            "Keep the affected skin clean.",
            "Seek medical advice when necessary."
        ]
    }
}
# =========================================================
# LOAD MODEL
# =========================================================
# The model is hosted on Google Drive (GitHub/Git LFS only stored a
# tiny pointer file). Paste the file ID from the share link here:
# https://drive.google.com/file/d/<FILE_ID>/view
GDRIVE_FILE_ID = "1xkTs2TQ3QsNw7-p5MpE-m7p1UJ56mHHV"

# New file name so an old/stale Skin_disease.pkl from the repo is never used
MODEL_PATH = "Skin_disease (2).pkl"

# Set to False once everything works to hide the debug panel
SHOW_DEBUG = True

# Packages that can affect how a fastai model is unpickled
KEY_PACKAGES = [
    "fastai", "fastcore", "fasttransform", "plum-dispatch",
    "torch", "torchvision", "numpy", "pandas", "scipy",
    "scikit-learn", "matplotlib", "pillow", "packaging",
    "pyyaml", "requests", "spacy", "fastprogress", "fastdownload",
    "dill", "cloudpickle", "streamlit", "gdown",
]


def package_versions():
    from importlib import metadata
    versions = {"python": sys.version}
    for name in KEY_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


@st.cache_resource
def download_model():
    """Download the model from Google Drive and return (size, md5)."""
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)

    gdown.download(
        id=GDRIVE_FILE_ID,
        output=MODEL_PATH,
        quiet=False,
    )

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            "Model download failed. Check that the Google Drive file "
            "is shared as 'Anyone with the link' and the file ID is correct."
        )

    size = os.path.getsize(MODEL_PATH)
    if size < 1024 * 1024:
        raise RuntimeError(
            f"Downloaded file is only {size} bytes - too small to be the "
            "model. Check the Google Drive sharing setting "
            "('Anyone with the link')."
        )

    with open(MODEL_PATH, "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()

    return size, md5


@st.cache_resource
def load_model():
    return load_learner(MODEL_PATH, cpu=True)


def trace_failed_load(path):
    """
    Re-run the load with a tracing unpickler that records which
    classes/functions were being rebuilt. The LAST entries in the
    trace show the object that made the load fail.
    """
    class TracingUnpickler(pickle.Unpickler):
        trace = []

        def find_class(self, module, name):
            TracingUnpickler.trace.append(f"{module}.{name}")
            return super().find_class(module, name)

    # torch.load reads pickle_module.__name__, so this must be a real module
    tracer = types.ModuleType("tracer_pickle")
    for k, v in vars(pickle).items():
        if not k.startswith("__"):
            setattr(tracer, k, v)
    tracer.Unpickler = TracingUnpickler

    try:
        load_learner(path, cpu=True, pickle_module=tracer)
        error_text = "Trace run finished without an error."
    except Exception:
        error_text = traceback.format_exc()

    return TracingUnpickler.trace[-40:], error_text


def list_pickle_globals(path):
    """
    Read the model file WITHOUT loading it and list every class/function
    the pickle refers to (module.name). Works even when loading fails.
    """
    import zipfile
    import pickletools

    with zipfile.ZipFile(path) as z:
        pkl_name = next(n for n in z.namelist() if n.endswith("data.pkl"))
        data = z.read(pkl_name)

    found = set()
    strings = []
    for op, arg, _pos in pickletools.genops(data):
        if op.name in ("SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8", "UNICODE"):
            strings.append(arg)
        elif op.name == "GLOBAL":
            found.add(str(arg).replace(" ", "."))
        elif op.name == "STACK_GLOBAL" and len(strings) >= 2:
            found.add(f"{strings[-2]}.{strings[-1]}")
    return sorted(found)


# ---- 1. Show environment info BEFORE loading (visible even on failure) ----
if SHOW_DEBUG:
    with st.expander("🛠 Debug info (environment & model file)", expanded=True):
        st.write("**Package versions on Streamlit**")
        st.json(package_versions())

# ---- 2. Download ----
try:
    model_size, model_md5 = download_model()
except Exception:
    st.error("Error downloading model:")
    st.code(traceback.format_exc())
    st.stop()

if SHOW_DEBUG:
    with st.expander("🛠 Model file info", expanded=True):
        st.write({
            "model_file_size_bytes": model_size,
            "model_file_md5": model_md5,
        })

# ---- 3. Load (with tracing diagnostics if it fails) ----
try:
    model = load_model()
except Exception:
    st.error("Error loading model:")
    st.code(traceback.format_exc())

    st.subheader("🔬 Diagnostics")

    # A: every class/function referenced inside the model file
    try:
        globals_list = list_pickle_globals(MODEL_PATH)
        st.write(f"**A. Classes/functions referenced in the model file "
                 f"({len(globals_list)}):**")
        st.code("\n".join(globals_list))
    except Exception:
        st.write("Could not read the model file's contents:")
        st.code(traceback.format_exc())

    # B: what the unpickler was rebuilding when it failed
    try:
        last_globals, trace_error = trace_failed_load(MODEL_PATH)
        st.write("**B. Last classes/functions resolved before the failure "
                 "(failing object is at or near the bottom):**")
        st.code("\n".join(last_globals) if last_globals else "(none)")
        st.code(trace_error)
    except Exception:
        st.write("Tracing itself failed:")
        st.code(traceback.format_exc())
    st.stop()

# =========================================================
# IMAGE UPLOADER
# =========================================================
uploaded_file = st.file_uploader(
    "Upload a skin image",
    type=["jpg", "jpeg", "png"]
)
# =========================================================
# IMAGE PROCESSING
# =========================================================
if uploaded_file is not None:
    # Convert uploaded image into FastAI-compatible image
    image = PILImage.create(uploaded_file)
    # =====================================================
    # DISPLAY IMAGE
    # =====================================================
    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )
    # =====================================================
    # ANALYZE BUTTON
    # =====================================================
    if st.button("🔍 Analyze Image"):
        with st.spinner("Analyzing image..."):
            try:
                # =================================================
                # FASTAI PREDICTION
                # =================================================
                pred, pred_idx, probs = model.predict(image)
                # Convert tensor index to Python integer
                pred_idx = pred_idx.item()
                # Convert model confidence to percentage
                confidence = float(
                    probs[pred_idx]
                ) * 100
                # Convert prediction to string
                pred_key = str(pred).strip()
                # Find disease information
                info = disease_info.get(
                    pred_key
                )
                # =================================================
                # RESULT
                # =================================================
                st.divider()
                st.subheader(
                    "🔎 Screening Result"
                )
                # -------------------------------------------------
                # MAIN PREDICTION
                # -------------------------------------------------
                if info:
                    display_name = info["name"]
                else:
                    display_name = pred_key
                st.write(
                    f"### Prediction: **{display_name}**"
                )
                st.write(
                    f"**Model Confidence: {confidence:.2f}%**"
                )
                # =================================================
                # OTHER MODEL PREDICTIONS
                # =================================================
                st.subheader(
                    "🔄 Other Model Predictions"
                )
                # Get class names from FastAI
                class_names = model.dls.vocab
                # Sort probabilities from highest to lowest
                sorted_indices = probs.argsort(
                    descending=True
                )
                # Number of alternative predictions
                num_alternatives = 3
                shown = 0
                for idx in sorted_indices:
                    idx = int(idx)
                    # Skip the main prediction
                    if idx == pred_idx:
                        continue
                    # Get class name
                    alternative_key = str(
                        class_names[idx]
                    ).strip()
                    # Get confidence
                    alternative_confidence = float(
                        probs[idx]
                    ) * 100
                    # Find disease information
                    alternative_info = disease_info.get(
                        alternative_key
                    )
                    # Get readable disease name
                    if alternative_info:
                        alternative_name = (
                            alternative_info["name"]
                        )
                    else:
                        alternative_name = alternative_key
                    # Display alternative prediction
                    st.write(
                        f"**{shown + 1}. "
                        f"{alternative_name} — "
                        f"{alternative_confidence:.2f}%**"
                    )
                    # Progress bar
                    st.progress(
                        min(
                            alternative_confidence / 100,
                            1.0
                        )
                    )
                    shown += 1
                    # Only show top 3 alternatives
                    if shown >= num_alternatives:
                        break
                # =================================================
                # DISEASE INFORMATION
                # =================================================
                if info:
                    st.divider()
                    # -------------------------------------------------
                    # DESCRIPTION
                    # -------------------------------------------------
                    st.subheader(
                        "📖 About this condition"
                    )
                    st.write(
                        info["description"]
                    )
                    # -------------------------------------------------
                    # POSSIBLE CAUSES
                    # -------------------------------------------------
                    st.subheader(
                        "🧬 Possible causes"
                    )
                    for cause in info["causes"]:
                        st.write(
                            f"- {cause}"
                        )
                    # -------------------------------------------------
                    # WHAT TO DO
                    # -------------------------------------------------
                    st.subheader(
                        "💡 What you can do"
                    )
                    for action in info["what_to_do"]:
                        st.write(
                            f"- {action}"
                        )
                else:
                    st.info(
                        "Additional information for this "
                        "prediction is not currently available."
                    )
                # =================================================
                # DISCLAIMER
                # =================================================
                st.divider()
                st.warning(
                    "⚠️ This result is an AI-based preliminary "
                    "screening and is NOT a medical diagnosis. "
                    "The model may produce incorrect results. "
                    "Please consult a qualified healthcare "
                    "professional for proper evaluation."
                )
            except Exception as e:
                st.error(
                    "An error occurred while analyzing the image."
                )
                st.exception(e)
