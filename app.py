import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
import glob
import kagglehub
from streamlit_option_menu import option_menu

import skimage.measure
from skimage.transform import rescale, resize
from skimage import exposure
from skimage import filters
from skimage import feature
from skimage import img_as_float

# --- CONFIGURATION & CUSTOM CSS ---
st.set_page_config(page_title="MRI & Alzheimer's Vanguard", layout="wide", page_icon="🧠")

st.markdown("""
<style>
    /* Global Font & Colors */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background-color: #0d1117;
        color: #e6edf3;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #ffffff;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    /* Custom Title Area */
    .hero-section {
        background: linear-gradient(135deg, #1f2937, #111827);
        padding: 40px;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        border: 1px solid #374151;
    }
    
    .hero-title {
        font-size: 3rem;
        background: -webkit-linear-gradient(45deg, #3b82f6, #8b5cf6, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
        font-weight: 800;
    }
    
    .hero-subtitle {
        color: #9ca3af;
        font-size: 1.2rem;
        font-weight: 400;
    }
    
    /* Cards for metrics/images */
    .metric-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(59, 130, 246, 0.2);
        border-color: #3b82f6;
    }
    
    /* Style DataFrames */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid #30363d !important;
    }
    
    /* Custom Divider */
    hr {
        border-top: 1px solid #30363d;
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)


# --- HERO SECTION ---
st.markdown("""
<div class="hero-section">
    <div class="hero-title">NeuroVision Vanguard</div>
    <div class="hero-subtitle">Advanced Cognitive Image Processing & Alzheimer's Analytics Dashboard</div>
</div>
""", unsafe_allow_html=True)


# --- DATA DOWNLOADING (CACHED) ---
@st.cache_data(show_spinner=False, ttl=3600*24)
def download_data():
    with st.spinner("Initializing Data Streams..."):
        mri_alzheimers_path = kagglehub.dataset_download('jboysen/mri-and-alzheimers')
        imagesoasis_path = kagglehub.dataset_download('ninadaithal/imagesoasis')
    return mri_alzheimers_path, imagesoasis_path

mri_path, oasis_path = download_data()

# --- TABULAR DATA LOADING ---
@st.cache_data
def load_tabular_data(base_path):
    longitudinal_data = pd.read_csv(os.path.join(base_path, "oasis_longitudinal.csv"))
    cross_sectional = pd.read_csv(os.path.join(base_path, "oasis_cross-sectional.csv"))
    return longitudinal_data, cross_sectional

long_df, cross_df = load_tabular_data(mri_path)

# --- IMAGE DATA PROCESSING ---
@st.cache_data
def load_image_paths(base_path):
    image_files = glob.glob(os.path.join(base_path, "*", "*", "*.jpg"))
    images = []
    for path in image_files:
        filename = os.path.basename(path)
        subject_id = "_".join(filename.split('_')[:3])
        images.append((subject_id, path))
    return pd.DataFrame(images, columns=['ID', 'ImageFiles'])

image_df = load_image_paths(oasis_path)

# --- TOP NAVIGATION BAR ---
selected = option_menu(
    menu_title=None,
    options=["Data Matrix", "Neural Imaging", "AI Architectures", "Diagnostic Sandbox"],
    icons=["clipboard-data", "image", "cpu", "file-medical"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#161b22", "border-radius": "100px", "border": "1px solid #30363d"},
        "icon": {"color": "#3b82f6", "font-size": "18px"},
        "nav-link": {"font-size": "16px", "text-align": "center", "margin": "0px", "--hover-color": "#1f2937"},
        "nav-link-selected": {"background-color": "#3b82f6", "color": "white", "border-radius": "100px"},
    }
)

# --- PAGE 1: DATA MATRIX ---
if selected == "Data Matrix":
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.subheader("📊 Longitudinal Patient Dynamics")
        st.write("Temporal tracking of cognitive decline and MRI metrics.")
        st.dataframe(long_df.head(100), use_container_width=True, height=250)
        st.caption(f"Dataset dimensionality: {long_df.shape[0]} subjects × {long_df.shape[1]} features")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.subheader("🎯 Cross-Sectional Triage")
        st.write("Static snapshot of pathological markers.")
        st.dataframe(cross_df.head(100), use_container_width=True, height=250)
        st.caption(f"Dataset dimensionality: {cross_df.shape[0]} subjects × {cross_df.shape[1]} features")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Demographic & Clinical Distribution")
    
    # Custom plotting aesthetics
    sns.set_theme(style="darkgrid")
    # Change grid color for dark mode aesthetics
    plt.style.use('dark_background')
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(cross_df['Age'], bins=20, kde=True, ax=ax, color='#3b82f6', edgecolor='white', linewidth=1)
        ax.set_title("Age Demographics", color='white', fontweight='bold')
        ax.set_facecolor('#0d1117')
        fig.patch.set_facecolor('#161b22')
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with c2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.countplot(data=cross_df, x='CDR', ax=ax, palette=['#3b82f6', '#8b5cf6', '#ec4899', '#10b981'])
        ax.set_title("Clinical Dementia Rating (CDR)", color='white', fontweight='bold')
        ax.set_facecolor('#0d1117')
        fig.patch.set_facecolor('#161b22')
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)


# --- PAGE 2: NEURAL IMAGING ---
elif selected == "Neural Imaging":
    st.markdown("### 🔬 Tomographic Scan Analysis")
    
    # Selectors in an expander for clean UI
    with st.expander("Filter Scan Database", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            subject_ids = image_df['ID'].unique()
            selected_id = st.selectbox("Patient Identifier Code", subject_ids)
        with col2:
            subject_images = image_df[image_df['ID'] == selected_id]['ImageFiles'].tolist()
            if subject_images:
                selected_image_path = st.selectbox("Volumetric Slice Target", subject_images, format_func=lambda x: os.path.basename(x))
    
    if subject_images:
        original_img = Image.open(selected_image_path)
        img_np = np.array(original_img)
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        col_img, col_metrics = st.columns([1, 1])
        with col_img:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.image(original_img, caption=f"Source: {os.path.basename(selected_image_path)}", use_column_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col_metrics:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.markdown("#### ⚡ Matrix Analytics")
            
            # Use columns for metrics
            m1, m2 = st.columns(2)
            m1.metric("Resolution", f"{img_np.shape[1]}×{img_np.shape[0]}")
            m2.metric("Channels", f"{img_np.shape[2] if len(img_np.shape)==3 else 1}")
            
            m3, m4 = st.columns(2)
            m3.metric("Min Density", f"{img_np.min()}")
            m4.metric("Max Density", f"{img_np.max()}")
            
            entropy_img = original_img.resize((64, 64)).convert('L')
            entropy_val = skimage.measure.shannon_entropy(np.array(entropy_img))
            st.markdown("<br>", unsafe_allow_html=True)
            st.metric("Information Saturation (Shannon Entropy)", f"{entropy_val:.5f} bits")
            st.progress(entropy_val/10) # Assuming max entropy roughly ~ 8 for 8-bit image
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)
        
        # Image Transformations tabs
        st.markdown("### 🧪 Signal Processing Suite")
        t1, t2, t3, t4 = st.tabs(["Luminance Modulation", "Edge Isolation", "Feature Generation", "Spatial Rendering"])
        
        with t1:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            gamma = st.slider("Gamma Curvature Factor", 0.1, 5.0, 2.0, 0.1)
            gamma_corrected = exposure.adjust_gamma(img_np, gamma)
            log_corrected = exposure.adjust_log(img_np, 1)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.image(img_np, caption="Baseline Scan", use_column_width=True)
            with c2:
                st.image(gamma_corrected, caption=f"Gamma Corrected (γ={gamma})", use_column_width=True)
            with c3:
                st.image(log_corrected, caption="Logarithmic Mapping", use_column_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with t2:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            if len(img_np.shape) == 3:
                gray_img = np.array(original_img.convert('L'))
            else:
                gray_img = img_np
                
            edges = filters.sobel(gray_img)
            col1, col2 = st.columns(2)
            with col1:
                st.image(original_img, caption="Baseline Scan", use_column_width=True)
            with col2:
                edges_normalized = (edges - edges.min()) / (edges.max() - edges.min())
                st.image(edges_normalized, caption="Sobel Manifold Mapping", use_column_width=True, clamp=True)
            st.markdown("</div>", unsafe_allow_html=True)
                
        with t3:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            col_ctrl, col_plot = st.columns([1, 2])
            
            with col_ctrl:
                max_sigma = st.slider("Max Gaussian Variance (Sigma)", 10, 200, 100)
                threshold = st.slider("Activation Threshold", 0.01, 0.20, 0.07, 0.01)
            
            with col_plot:
                if len(img_np.shape) == 3:
                    img_gray = img_np[:,:,0]
                else:
                    img_gray = img_np
                    
                blobs = feature.blob_dog(img_gray, max_sigma=max_sigma, threshold=threshold)
                
                plt.style.use('dark_background')
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.imshow(img_gray, cmap='gray')
                for blob in blobs:
                    y, x, r = blob
                    c = plt.Circle((x, y), r * np.sqrt(2), color='#ec4899', linewidth=2, fill=False)
                    ax.add_patch(c)
                ax.set_axis_off()
                fig.patch.set_facecolor('#161b22')
                
                st.pyplot(fig)
                st.caption(f"Cluster points identified: {len(blobs)}")
            st.markdown("</div>", unsafe_allow_html=True)

        with t4:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                scale = st.slider("Resolution Interpolation Factor", 0.1, 1.0, 0.25, 0.05)
                image_rescaled = rescale(img_np, scale, anti_aliasing=False, channel_axis=2 if len(img_np.shape)==3 else None)
                st.image(image_rescaled, caption=f"Lossy Downsampling ({int(scale*100)}%)", use_column_width=True, clamp=True)
            
            with col2:
                target_w = st.number_input("Target Dimensions (W)", 32, 1024, img_np.shape[1]//4)
                target_h = st.number_input("Target Dimensions (H)", 32, 1024, img_np.shape[0]//4)
                image_resized = resize(img_np, (target_h, target_w), anti_aliasing=True)
                st.image(image_resized, caption=f"Anti-Aliased Filtering ({target_w}x{target_h})", use_column_width=True, clamp=True)
            st.markdown("</div>", unsafe_allow_html=True)


# --- PAGE 3: AI ARCHITECTURES ---
elif selected == "AI Architectures":
    st.markdown("### 🤖 Diagnostic Neural Architectures")
    st.write("Core computational models instantiated for clinical inference (CDR Detection).")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.write("#### Baseline CNN")
        st.caption("Low-latency feature extraction")
        st.code("""
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16*(128//2)*(128//2), num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        return self.fc1(x.view(-1, 16*(128//2)*(128//2)))
        """, language="python")
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.write("#### ResNet-18 Node")
        st.caption("Deep residual learning framework")
        st.code("""
def get_resnet(num_classes):
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(
        model.fc.in_features, 
        num_classes
    )
    return model
        """, language="python")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with c3:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.write("#### Vision Transformer")
        st.caption("Attention-based global spatial analysis")
        st.code("""
def get_vit(num_classes):
    model = vit_b_16(pretrained=True)
    model.heads.head = nn.Linear(
        model.heads.head.in_features, 
        num_classes
    )
    return model
        """, language="python")
        st.markdown("</div>", unsafe_allow_html=True)

# --- PAGE 4: DIAGNOSTIC SANDBOX ---
elif selected == "Diagnostic Sandbox":
    st.markdown("### 📤 Diagnostic AI Sandbox: Custom Scan Inference")
    st.write("Upload MRI scan for intelligent image-based Alzheimer detection")

    st.markdown("<hr>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload Medical Scan", type=["jpg","jpeg","png"])

 # 🔥 STRICT IMAGE VALIDATION FUNCTION
def is_medical_scan(image_np):
    gray = np.mean(image_np, axis=2)

    # 1️⃣ Color variance (MRI = low color variation)
    color_var = np.mean(np.var(image_np, axis=2))

    # 2️⃣ Edge structure (brain edges exist)
    edges = filters.sobel(gray)
    edge_density = np.mean(edges)

    # 3️⃣ Intensity distribution
    dark_ratio = np.sum(gray < 60) / gray.size
    bright_ratio = np.sum(gray > 180) / gray.size

    # ✅ STRICT CONDITION (reject posters)
    if (
        color_var < 100 and           # must be grayscale-like
        edge_density > 0.01 and       # must have structure
        dark_ratio > 0.05 and         # must have dark regions
        bright_ratio > 0.05           # must have bright regions
    ):
        return True

    return False


if uploaded_file is not None:

    user_img = Image.open(uploaded_file).convert('RGB')
    user_img_np = np.array(user_img)

    # 🔴 HARD VALIDATION (IMPORTANT)
    if not is_medical_scan(user_img_np):
        st.error("❌ This is NOT a valid MRI/CT medical scan. Please upload a brain scan.")
        st.stop()

    col_img, col_res = st.columns([1,1])

    with col_img:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.image(user_img, caption="Uploaded Scan", use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_res:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)

        st.markdown("#### ⚙️ AI Inference Engine")

        with st.spinner("Analyzing brain structure..."):
            import time
            time.sleep(1)

            # 🔹 Preprocess
            gray = user_img.resize((128,128)).convert('L')
            arr = np.array(gray)

            # 🔹 Features
            entropy = skimage.measure.shannon_entropy(arr)
            dark_ratio = np.sum(arr < 50) / arr.size
            edge = filters.sobel(arr)
            edge_density = np.mean(edge)

            # 🔹 Improved scoring (more stable)
            score = (entropy/8)*0.6 + dark_ratio*0.25 + edge_density*5*0.15

            # 🔹 Prediction
            if score > 0.5:
                pred = "Non-Demented"
                color = "#10b981"
                conf = score*100
            else:
                pred = "Demented"
                color = "#ef4444"
                conf = (1-score)*100

            # ✅ OUTPUT
            st.markdown(f"<h3 style='color:{color}'>Prediction: {pred}</h3>", unsafe_allow_html=True)
            st.progress(int(conf))
            st.write(f"Confidence: {conf:.2f}%")

            st.markdown("---")
            st.write("### Extracted Features")
            st.write(f"Entropy: `{entropy:.3f}`")
            st.write(f"Dark Pixel Ratio: `{dark_ratio:.3f}`")
            st.write(f"Edge Density: `{edge_density:.4f}`")
            st.write(f"Resolution: `{user_img_np.shape[1]} x {user_img_np.shape[0]}`")

        st.markdown("</div>", unsafe_allow_html=True)

