import streamlit as st
import os
import io
from PIL import Image, ImageFilter # Import ImageFilter for dummy heatmap
from datetime import datetime
import json
import base64
import google.generativeai as genai
import asyncio
import uuid
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import mark_boundaries, slic
from skimage.color import label2rgb
from skimage.measure import regionprops, regionprops_table
from skimage.filters import sobel
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as ReportLabImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib import colors

# Configure Gemini API
YOUR_GEMINI_API_KEY = "AIzaSyDhMC_PEi-3ueM7a6jVc1qZhxTQSfQd7ZU"
genai.configure(api_key=YOUR_GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-2.0-flash')

# Import helper modules (assuming these are in utils_simple.py)
from utils_simple import (
    process_file, generate_heatmap, save_analysis,
    get_latest_analyses, generate_report, search_pubmed, # generate_report is not used directly here, but kept for context
    generate_statistics_report, extract_findings_and_keywords
)

# Import chat system (assuming this is in chat_system.py)
from chat_system import render_chat_interface, create_manual_chat_room

# Streamlit config
st.set_page_config(
    page_title="Medical Image Analysis Platform",
    page_icon="🏥",
    layout="wide"
)

# Session state defaults
defaults = {
    "gemini_key_configured": True,
    "file_data": None,
    "analysis_results": None,
    "file_name": None,
    "file_type": None,
    "uploaded_file": None,
    "segmentation_params": {
        "n_segments": 100,
        "compactness": 10,
        "sigma": 1.0,
        "threshold": 0.05
    },
    # New industry-level session state additions
    "patient_id": "",
    "patient_age": "",
    "patient_gender": "Prefer not to say",
    "referring_physician": "",
    "user_role": "Radiologist", # Default user role
    "audit_log": [], # To store in-app actions
    "report_sections": { # For granular report customization
        "radiological_analysis": True,
        "key_findings": True,
        "xai_explanation": True,
        "xai_image": True,
        "medical_literature": True,
        "quantitative_analysis": True, # New section for quantitative analysis
    },
    "quantitative_analysis_results": None, # Store quantitative results
}
for key, val in defaults.items():
    st.session_state.setdefault(key, val)

# --- Helper for Audit Log ---
def add_to_audit_log(action, details=""):
    """Adds an entry to the in-app audit log."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.audit_log.append(f"[{timestamp}] {st.session_state.user_role}: {action} - {details}")

# Header
st.title("🏥 Advanced Medical Diagnosis Platform")
st.markdown("Upload medical images for AI-powered analysis, generate comprehensive reports, and collaborate securely.")

# Sidebar
with st.sidebar:
    st.header("Configuration & Tools")

    # Simulated User Role Selection
    st.subheader("User Profile")
    st.session_state.user_role = st.selectbox(
        "Select Your Role",
        ["Radiologist", "Referring Physician", "Administrator", "Resident"],
        index=["Radiologist", "Referring Physician", "Administrator", "Resident"].index(st.session_state.user_role),
        on_change=lambda: add_to_audit_log(f"Role changed to {st.session_state.user_role}")
    )
    st.markdown(f"**Current Role:** `{st.session_state.user_role}`")
    st.markdown("---") # Separator

    st.subheader("Analysis Options")
    enable_xai = st.checkbox("Enable Explainable AI", value=True, key="enable_xai_sidebar")
    include_references = st.checkbox("Include Medical References", value=True, key="include_references_sidebar")
    
    if enable_xai:
        st.subheader("Segmentation Parameters (XAI)")
        st.session_state.segmentation_params["n_segments"] = st.slider(
            "Number of segments", 10, 500, st.session_state.segmentation_params["n_segments"], key="n_segments_slider"
        )
        st.session_state.segmentation_params["compactness"] = st.slider(
            "Compactness", 1, 30, st.session_state.segmentation_params["compactness"], key="compactness_slider"
        )
        st.session_state.segmentation_params["sigma"] = st.slider(
            "Smoothing (sigma)", 0.1, 5.0, st.session_state.segmentation_params["sigma"], key="sigma_slider"
        )
        st.session_state.segmentation_params["threshold"] = st.slider(
            "Boundary threshold", 0.01, 0.5, st.session_state.segmentation_params["threshold"], key="threshold_slider"
        )
    st.markdown("---") # Separator

    st.subheader("Recent Analyses (Worklist)")
    recent_analyses = get_latest_analyses(limit=5) # Assuming get_latest_analyses can fetch more details now
    if recent_analyses:
        for analysis in recent_analyses:
            filename = analysis.get('filename', 'Unknown')
            date_str = analysis.get('date', '')[:10]
            patient_id_display = analysis.get('patient_details', {}).get('patient_id', 'N/A')
            st.caption(f"**{filename}**")
            st.markdown(f"*{date_str}* | Patient ID: `{patient_id_display}`")
            st.markdown("---")
    else:
        st.info("No recent analyses.")
    
    st.markdown("---") # Separator
    st.subheader("Platform Statistics")
    if st.button("Generate Comprehensive Statistics Report", key="generate_stats_report_sidebar"):
        stats_report = generate_statistics_report()
        if stats_report:
            b64_pdf = base64.b64encode(stats_report.read()).decode()
            href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="platform_statistics_report.pdf">Download Statistics Report</a>'
            st.markdown(href, unsafe_allow_html=True)
        add_to_audit_log("Generated platform statistics report")

# Gemini-based image analysis
def analyze_image_gemini(image_data, enable_xai=True):
    buffered = io.BytesIO()
    image_data.save(buffered, format="PNG")
    encoded_image = base64.b64encode(buffered.getvalue()).decode()
    prompt = """
    Provide a comprehensive and detailed medical analysis of this image, aiming for 100% insight.
    Include:
    1. Exhaustive description of all key findings, even subtle ones.
    2. A thorough list of possible diagnoses, considering all relevant possibilities.
    3. Detailed recommendations for clinical correlation, further investigations, or follow-up actions.

    Format your response with clear "Radiological Analysis" and "Impression" sections, ensuring maximum detail and insight.
    """
    try:
        response = gemini_model.generate_content(
            [prompt, {"mime_type": "image/png", "data": base64.b64decode(encoded_image)}],
            generation_config=genai.types.GenerationConfig(max_output_tokens=1500)
        )
        analysis = response.text
        findings, keywords = extract_findings_and_keywords(analysis)
        return {
            "id": str(uuid.uuid4()),
            "analysis": analysis,
            "findings": findings,
            "keywords": keywords,
            "date": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "id": str(uuid.uuid4()),
            "analysis": f"Error analyzing image: {str(e)}",
            "findings": [],
            "keywords": [],
            "date": datetime.now().isoformat()
        }

# --- New Feature: Automated Quantitative Analysis (Dummy Implementation) ---
def perform_quantitative_analysis(image_array):
    """
    Performs dummy quantitative analysis on the image array.
    In a real scenario, this would involve advanced image processing.
    """
    st.write("Performing quantitative analysis...")
    # Simulate some quantitative measurements
    height, width = image_array.shape[:2]
    average_pixel_intensity = np.mean(image_array)
    std_dev_pixel_intensity = np.std(image_array)

    # Simulate finding a "lesion" and measuring it
    # This is a very basic simulation; real analysis would involve segmentation
    # and precise measurement of actual pathologies.
    lesion_detected = False
    lesion_size_cm = None
    lesion_density_hu = None

    if average_pixel_intensity > 100 and average_pixel_intensity < 150: # Arbitrary threshold for a "finding"
        lesion_detected = True
        lesion_size_cm = round(np.random.uniform(0.5, 5.0), 2) # Simulate size
        lesion_density_hu = round(np.random.uniform(20, 80), 1) # Simulate density (Hounsfield Units)

    results = {
        "image_dimensions": f"{width}x{height} pixels",
        "average_intensity": f"{average_pixel_intensity:.2f}",
        "std_dev_intensity": f"{std_dev_pixel_intensity:.2f}",
        "lesion_detected": lesion_detected,
        "lesion_size_cm": lesion_size_cm,
        "lesion_density_hu": lesion_density_hu
    }
    add_to_audit_log("Quantitative analysis performed")
    return results

# Async wrapper for image analysis
async def perform_analysis():
    if st.session_state.uploaded_file and st.session_state.gemini_key_configured:
        with st.spinner("Analyzing image..."):
            analysis_results = analyze_image_gemini(
                st.session_state.file_data["data"],
                enable_xai=enable_xai
            )
            
            # Add patient details to analysis results before saving
            analysis_results["patient_details"] = {
                "patient_id": st.session_state.patient_id,
                "age": st.session_state.patient_age,
                "gender": st.session_state.patient_gender,
                "referring_physician": st.session_state.referring_physician
            }

            # Perform quantitative analysis
            if st.session_state.file_data.get("array") is not None:
                st.session_state.quantitative_analysis_results = perform_quantitative_analysis(st.session_state.file_data["array"])
                analysis_results["quantitative_analysis"] = st.session_state.quantitative_analysis_results # Store in analysis results
            else:
                st.session_state.quantitative_analysis_results = None
                analysis_results["quantitative_analysis"] = None

            analysis_results = save_analysis(analysis_results, filename=st.session_state.uploaded_file.name)
            st.session_state.analysis_results = analysis_results
            st.session_state.findings = analysis_results.get("findings", [])
            add_to_audit_log("Image analysis performed", f"File: {st.session_state.uploaded_file.name}, Patient ID: {st.session_state.patient_id}")
            st.rerun()

# Enhanced segmentation overlay with advanced XAI explanations, region properties, saliency maps, and anomaly detection
def advanced_visualize(image_array):
    try:
        params = st.session_state.segmentation_params
        segments = slic(
            image_array, 
            n_segments=params["n_segments"],
            compactness=params["compactness"],
            sigma=params["sigma"],
            start_label=1
        )
        
        # Determine if image is grayscale or color
        if len(image_array.shape) == 2:
            gray = image_array
            intensity_image = gray
        else:
            gray = np.mean(image_array, axis=2)
            intensity_image = gray  # Use grayscale for intensity calculations
        
        # Compute region properties
        props = regionprops_table(
            segments, 
            intensity_image=intensity_image,
            properties=['label', 'area', 'mean_intensity', 'perimeter', 'solidity', 'eccentricity', 'equivalent_diameter']
        )
        df = pd.DataFrame(props)
        
        # Identify potential abnormal regions (e.g., outliers in intensity, size, shape)
        # Low intensity quartiles might indicate denser tissues or lesions in certain modalities
        low_intensity_threshold = df['mean_intensity'].quantile(0.25)
        high_intensity_threshold = df['mean_intensity'].quantile(0.75)
        large_area_threshold = df['area'].quantile(0.9)
        abnormal_regions = df[
            (df['mean_intensity'] < low_intensity_threshold) | 
            (df['mean_intensity'] > high_intensity_threshold) |
            (df['area'] > large_area_threshold) |
            (df['eccentricity'] > 0.8)  # Highly elongated regions might indicate vessels or fractures
        ]
        
        # Create multi-panel visualization
        fig = plt.figure(figsize=(24, 12))
        gs = fig.add_gridspec(2, 3)
        
        # Original image
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(image_array)
        ax1.set_title("Original Image", fontsize=14)
        ax1.axis('off')
        
        # Segmentation overlay with boundaries
        overlay = mark_boundaries(
            image_array, 
            segments, 
            color=(1, 0, 0),
            mode='thick',
            outline_color=(1, 0, 0)
        )
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(overlay)
        ax2.set_title("AI Segmentation Overlay", fontsize=14)
        ax2.axis('off')
        
        # Colored regions with labels
        colored_regions = label2rgb(
            segments, 
            image_array, 
            kind='avg',
            bg_label=0
        )
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(colored_regions)
        ax3.set_title("Region Analysis with Labels", fontsize=14)
        ax3.axis('off')
        
        # Add labels to regions
        regions = regionprops(segments)
        for region in regions:
            if region.area > params["n_segments"] * 0.05:  # Only label larger regions for clarity
                y, x = region.centroid
                ax3.text(x, y, str(region.label), color='white', fontsize=8, ha='center', va='center')
        
        # Saliency map (edge-based importance)
        saliency_map = sobel(gray)
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.imshow(saliency_map, cmap='hot')
        ax4.set_title("Saliency Map (Edge Detection)", fontsize=14)
        ax4.axis('off')
        
        # Intensity histogram
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.hist(intensity_image.ravel(), bins=256, color='blue', alpha=0.7)
        ax5.set_title("Pixel Intensity Histogram", fontsize=14)
        ax5.set_xlabel("Intensity Value")
        ax5.set_ylabel("Frequency")
        
        # Abnormal regions highlight
        abnormal_mask = np.zeros_like(segments)
        for label in abnormal_regions['label']:
            abnormal_mask[segments == label] = 1
        abnormal_overlay = mark_boundaries(image_array, abnormal_mask, color=(0, 1, 0), mode='thick')
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.imshow(abnormal_overlay)
        ax6.set_title("Highlighted Potential Abnormal Regions", fontsize=14)
        ax6.axis('off')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Display region properties table
        st.subheader("Detailed Region Properties")
        st.dataframe(df.style.highlight_max(color='lightgreen', axis=0))
        
        # Display abnormal regions
        if not abnormal_regions.empty:
            st.subheader("Potential Abnormal Regions (Outliers in Intensity/Size/Shape)")
            st.dataframe(abnormal_regions.style.background_gradient(cmap='OrRd'))
            abnormal_summary = f"""
            - Number of Abnormal Regions: {len(abnormal_regions)}
            - Average Area of Abnormals: {abnormal_regions['area'].mean():.2f} pixels
            - Average Intensity: {abnormal_regions['mean_intensity'].mean():.2f}
            - Average Eccentricity: {abnormal_regions['eccentricity'].mean():.2f} (0=circle, 1=line)
            """
        else:
            abnormal_summary = "- No significant abnormal regions detected based on current thresholds."
        
        # Generate comprehensive XAI explanation
        xai_explanation = f"""
        **Advanced Explainable AI (XAI) Segmentation Analysis:**
        
        - **Segments Created:** {len(np.unique(segments))}
        - **Algorithm:** Enhanced SLIC (Simple Linear Iterative Clustering) with Region Property Analysis and Anomaly Detection
        - **Parameters:**
          - Compactness: {params["compactness"]} (higher values create more regular-shaped segments)
          - Smoothing (sigma): {params["sigma"]} pixels
          - Boundary Threshold: {params["threshold"]}
          
        **Region Statistics Summary:**
        - Total Regions: {len(df)}
        - Average Area: {df['area'].mean():.2f} pixels (Std: {df['area'].std():.2f})
        - Average Perimeter: {df['perimeter'].mean():.2f} pixels
        - Average Solidity: {df['solidity'].mean():.2f} (1=compact, lower=irregular)
        - Average Eccentricity: {df['eccentricity'].mean():.2f}
        - Average Equivalent Diameter: {df['equivalent_diameter'].mean():.2f} pixels
        
        **Saliency Analysis:**
        The saliency map highlights edges and transitions using Sobel filters, indicating areas of high contrast which may correspond to anatomical boundaries or pathological features.
        
        **Intensity Distribution:**
        The histogram shows the overall pixel intensity distribution. Multimodal distributions may indicate different tissue types.
        
        **Anomaly Detection:**
        {abnormal_summary}
        These regions are flagged based on statistical outliers (e.g., intensity quartiles, large areas, high eccentricity). In medical context:
        - Low-intensity regions may represent denser tissues (e.g., bones, calcifications) or hypodense lesions.
        - High-intensity regions could indicate fat, air, or hyperdense areas (e.g., hemorrhages).
        - Irregular shapes (high eccentricity/low solidity) might suggest pathological growths or artifacts.
        
        **Clinical Interpretation:**
        This advanced segmentation provides interpretable insights into the AI's focus areas. Boundaries between segments often align with tissue interfaces. Flagged abnormal regions should be correlated with the radiological findings for potential pathologies. The labeled regions allow precise reference to specific areas in discussions or reports.
        For further customization, adjust segmentation parameters in the sidebar.
        """
        
        st.markdown(xai_explanation)
        return xai_explanation, fig
        
    except Exception as e:
        st.error(f"Error in advanced segmentation: {str(e)}")
        return None, None

# Generate PDF report with XAI content
def generate_report_with_xai(
    analysis_results,
    xai_content=None,
    xai_image=None,
    include_references=True,
    report_sections_to_include=None # New parameter for granular control
):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Default sections if not provided
    if report_sections_to_include is None:
        report_sections_to_include = st.session_state.report_sections

    # Patient Details from analysis_results
    patient_details = analysis_results.get("patient_details", {})
    patient_id = patient_details.get("patient_id", "N/A")
    patient_age = patient_details.get("age", "N/A")
    patient_gender = patient_details.get("gender", "N/A")
    referring_physician = patient_details.get("referring_physician", "") # Get referring physician

    # ---------- Header Section ----------
    hospital_name = "AI Medical Diagnostics Center"
    
    story.append(Paragraph(f"<b>{hospital_name}</b>", styles["Title"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph("<b>Medical Imaging Report</b>", styles["Heading2"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph(f"<b>Patient ID:</b> {patient_id}", styles["Normal"]))
    story.append(Paragraph(f"<b>Age:</b> {patient_age}", styles["Normal"]))
    story.append(Paragraph(f"<b>Gender:</b> {patient_gender}", styles["Normal"]))
    story.append(Paragraph(f"<b>Referring Physician:</b> {referring_physician}", styles["Normal"]))
    story.append(Paragraph(f"<b>Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
    story.append(Paragraph(f"<b>File:</b> {analysis_results.get('filename', 'Unknown')}", styles["Normal"]))
    story.append(Spacer(1, 12))

    # ---------- Radiological Analysis ----------
    if report_sections_to_include.get("radiological_analysis", True):
        story.append(Paragraph("<b>Radiological Analysis:</b>", styles["Heading2"]))
        analysis_text = analysis_results.get("analysis", "No analysis available.")
        for line in analysis_text.split("\n"):
            story.append(Paragraph(line.strip(), styles["Normal"]))
        story.append(Spacer(1, 12))

    # ---------- Key Findings ----------
    findings = analysis_results.get("findings", [])
    if report_sections_to_include.get("key_findings", True) and findings:
        story.append(Paragraph("<b>Impression / Key Findings:</b>", styles["Heading2"]))
        for f in findings:
            story.append(Paragraph(f"• {f}", styles["Normal"]))
        story.append(Spacer(1, 12))

    # ---------- Quantitative Analysis ----------
    quantitative_results = analysis_results.get("quantitative_analysis", {})
    if report_sections_to_include.get("quantitative_analysis", True) and quantitative_results:
        story.append(Paragraph("<b>Automated Quantitative Analysis:</b>", styles["Heading2"]))
        story.append(Paragraph(f"• Image Dimensions: {quantitative_results.get('image_dimensions', 'N/A')}", styles["Normal"]))
        story.append(Paragraph(f"• Average Pixel Intensity: {quantitative_results.get('average_intensity', 'N/A')}", styles["Normal"]))
        story.append(Paragraph(f"• Standard Deviation of Pixel Intensity: {quantitative_results.get('std_dev_intensity', 'N/A')}", styles["Normal"]))
        
        if quantitative_results.get("lesion_detected"):
            story.append(Paragraph(f"• **Potential Lesion Detected:** Yes", styles["Normal"]))
            story.append(Paragraph(f"  - Simulated Size: {quantitative_results.get('lesion_size_cm', 'N/A')} cm", styles["Normal"]))
            story.append(Paragraph(f"  - Simulated Density (HU): {quantitative_results.get('lesion_density_hu', 'N/A')}", styles["Normal"]))
        else:
            story.append(Paragraph(f"• Potential Lesion Detected: No (based on initial scan)", styles["Normal"]))

        story.append(Spacer(1, 12))

    # ---------- XAI Explanation ----------
    if report_sections_to_include.get("xai_explanation", True) and xai_content:
        story.append(Paragraph("<b>Explainable AI Analysis (XAI):</b>", styles["Heading2"]))
        for line in xai_content.split("\n"):
            story.append(Paragraph(line.strip(), styles["Normal"]))
        story.append(Spacer(1, 12))

    # ---------- XAI Image ----------
    if report_sections_to_include.get("xai_image", True) and xai_image:
        story.append(Paragraph("<b>Segmentation Visualization:</b>", styles["Heading3"]))
        img_buffer = io.BytesIO()
        xai_image.savefig(img_buffer, format="png", dpi=300, bbox_inches="tight")
        img_buffer.seek(0)
        story.append(ReportLabImage(img_buffer, width=6 * inch, height=3 * inch))
        story.append(Spacer(1, 12))

    # ---------- Medical Literature ----------
    if report_sections_to_include.get("medical_literature", True) and include_references and analysis_results.get("keywords"):
        story.append(Paragraph("<b>Relevant Medical Literature:</b>", styles["Heading2"]))
        references = search_pubmed(analysis_results["keywords"], max_results=3)
        for ref in references:
            ref_text = f"{ref['title']}. {ref['journal']}, {ref['year']} (PMID: {ref['id']})"
            story.append(Paragraph(ref_text, styles["Normal"]))
            story.append(Spacer(1, 6))

    # ---------- Footer ----------
    story.append(Spacer(1, 24))
    story.append(Paragraph("________________________", styles["Normal"]))
    
    # Display Referring Physician if available and selected
    if referring_physician:
        story.append(Paragraph(f"{referring_physician} (Referring Physician)", styles["Normal"]))
        story.append(Spacer(1, 6)) # Add a small space

    # Display Consulting Radiologist
    story.append(Paragraph(f"Dr. Anonymous (Consulting Radiologist)", styles["Normal"])) 
    story.append(Spacer(1, 12))
    story.append(Paragraph("Report generated by AI Medical Diagnostics Center", styles["Normal"]))

    doc.build(story)
    buffer.seek(0)
    return buffer


# Tabs
with st.spinner("Loading modules..."):
    tab1, tab2, tab3, tab4 = st.tabs(["Image Upload & Analysis", "Collaboration", "Reports", "Audit Log"])

# ------------------- TAB 1: Image Upload & Analysis -------------------
with tab1:
    st.header("Upload Medical Image for Analysis")

    # Patient Demographics Input
    st.subheader("Patient Demographics")
    col_pid, col_age, col_gender = st.columns(3)
    with col_pid:
        st.session_state.patient_id = st.text_input("Patient ID", value=st.session_state.patient_id, key="patient_id_input")
    with col_age:
        st.session_state.patient_age = st.text_input("Age", value=st.session_state.patient_age, key="patient_age_input")
    with col_gender:
        st.session_state.patient_gender = st.selectbox(
            "Gender", ["Male", "Female", "Prefer not to say"], 
            index=["Male", "Female", "Prefer not to say"].index(st.session_state.patient_gender),
            key="patient_gender_select"
        )
    st.session_state.referring_physician = st.text_input("Referring Physician", value=st.session_state.referring_physician, key="referring_physician_input")

    uploaded_now = st.file_uploader("Upload a medical image", type=["jpg", "jpeg", "png", "dcm", "nii", "nii.gz"], key="image_uploader")
    
    if uploaded_now is not None:
        if st.session_state.uploaded_file != uploaded_now: # Only process if a new file is uploaded
            st.session_state.uploaded_file = uploaded_now
            add_to_audit_log("Image uploaded", f"File: {uploaded_now.name}")
            # Clear previous analysis results when a new file is uploaded
            st.session_state.analysis_results = None 
            st.session_state.findings = []
            st.session_state.quantitative_analysis_results = None # Clear previous quantitative results
            st.rerun() # Rerun to display the new image and clear old results

    uploaded_file = st.session_state.uploaded_file

    if uploaded_file:
        try:
            file_data = process_file(uploaded_file)
            if file_data:
                st.session_state.file_data = file_data
                st.session_state.file_name = uploaded_file.name
                st.session_state.file_type = file_data["type"]
                st.image(file_data["data"], caption=f"Uploaded {file_data['type']} image", use_container_width=True)

                if st.button("Analyze Image", key="analyze_image_button"):
                    if not st.session_state.patient_id:
                        st.error("Please enter a Patient ID before analyzing the image.")
                    else:
                        asyncio.run(perform_analysis())

                if st.session_state.analysis_results:
                    st.subheader("Analysis Results")
                    st.markdown(st.session_state.analysis_results["analysis"])
                    if st.session_state.analysis_results.get("findings"):
                        st.subheader("Key Findings")
                        for idx, finding in enumerate(st.session_state.analysis_results["findings"], 1):
                            st.markdown(f"{idx}. {finding}")
                    if st.session_state.analysis_results.get("keywords"):
                        st.subheader("Keywords")
                        st.markdown(f"*{', '.join(st.session_state.analysis_results['keywords'])}*")
                    
                    # --- Display Quantitative Analysis Results ---
                    if st.session_state.quantitative_analysis_results:
                        st.subheader("Automated Quantitative Analysis")
                        q_results = st.session_state.quantitative_analysis_results
                        st.markdown(f"**Image Dimensions:** `{q_results.get('image_dimensions', 'N/A')}`")
                        st.markdown(f"**Average Pixel Intensity:** `{q_results.get('average_intensity', 'N/A')}`")
                        st.markdown(f"**Standard Deviation of Pixel Intensity:** `{q_results.get('std_dev_intensity', 'N/A')}`")
                        if q_results.get("lesion_detected"):
                            st.markdown(f"**Potential Lesion Detected:** `Yes`")
                            st.markdown(f"  - **Simulated Size:** `{q_results.get('lesion_size_cm', 'N/A')} cm`")
                            st.markdown(f"  - **Simulated Density (HU):** `{q_results.get('lesion_density_hu', 'N/A')}`")
                        else:
                            st.markdown(f"**Potential Lesion Detected:** `No (based on initial scan)`")
                    
                    # AI Feedback Mechanism
                    st.subheader("Provide Feedback on AI Analysis")
                    col_feedback1, col_feedback2, col_feedback3 = st.columns(3)
                    with col_feedback1:
                        if st.button("👍 Accurate", key="feedback_accurate"):
                            add_to_audit_log("AI Feedback", f"Analysis of {st.session_state.file_name} marked as Accurate.")
                            st.success("Thank you for your feedback!")
                    with col_feedback2:
                        if st.button("🤔 Needs Review", key="feedback_needs_review"):
                            add_to_audit_log("AI Feedback", f"Analysis of {st.session_state.file_name} marked as Needs Review.")
                            st.warning("Feedback recorded. A specialist will review.")
                    with col_feedback3:
                        if st.button("👎 Incorrect", key="feedback_incorrect"):
                            add_to_audit_log("AI Feedback", f"Analysis of {st.session_state.file_name} marked as Incorrect.")
                            st.error("Feedback recorded. This will help improve the model.")

                    # XAI Visualization
                    xai_content = None
                    xai_fig = None
                    
                    if enable_xai and file_data.get("array") is not None:
                        st.subheader("Explainable AI Visualization")
                        overlay, heatmap = generate_heatmap(file_data["array"])
                        col_basic, col_adv = st.columns([2, 1])
                        with col_basic:
                            st.image(overlay, caption="Heatmap Overlay", use_container_width=True)
                            st.image(heatmap, caption="Raw Heatmap", use_container_width=True)
                        with col_adv:
                            if st.checkbox("Show Advanced Segmentation", key="show_adv_segmentation"):
                                xai_content, xai_fig = advanced_visualize(file_data["array"])

                    if include_references and st.session_state.analysis_results.get("keywords"):
                        st.subheader("Relevant Medical Literature")
                        references = search_pubmed(st.session_state.analysis_results["keywords"], max_results=3)
                        for ref in references:
                            st.markdown(f"- **{ref['title']}** \n{ref['journal']}, {ref['year']} (PMID: {ref['id']})")

                    st.subheader("Report Generation")
                    st.markdown("Select sections to include in the PDF report:")
                    col_report_options = st.columns(len(st.session_state.report_sections))
                    for i, (section, default_val) in enumerate(st.session_state.report_sections.items()):
                        with col_report_options[i]:
                            st.session_state.report_sections[section] = st.checkbox(
                                section.replace('_', ' ').title(), 
                                value=default_val, 
                                key=f"report_section_{section}"
                            )

                    pdf_buffer = generate_report_with_xai(
                        st.session_state.analysis_results,
                        xai_content=xai_content,
                        xai_image=xai_fig,
                        include_references=include_references,
                        report_sections_to_include=st.session_state.report_sections
                    )
                    b64_pdf = base64.b64encode(pdf_buffer.read()).decode()
                    href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="medical_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf">Download PDF Report</a>'
                    st.markdown(href, unsafe_allow_html=True)
                    add_to_audit_log("PDF Report generated", f"File: {st.session_state.file_name}, Patient ID: {st.session_state.patient_id}")


                    st.subheader("Collaborate")
                    col1, col2 = st.columns(2)

                    with col1:
                        if st.button("Start Case Discussion", key="start_case_discussion_button"):
                            findings = st.session_state.analysis_results.get("findings", [])
                            if findings:
                                case_description = findings[0]
                            else:
                                case_description = st.session_state.uploaded_file.name
                            
                            # Include patient ID in case description for clarity in chat
                            case_description = f"Patient ID: {st.session_state.patient_id} - {case_description}"

                            created_case_id = create_manual_chat_room(st.session_state.user_name, case_description)
                            st.session_state.current_case_id = created_case_id
                            add_to_audit_log("Case discussion started", f"Case ID: {created_case_id}, Description: {case_description}")
                            st.rerun()

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            add_to_audit_log("Error processing file", f"Details: {str(e)}")

# ------------------- TAB 2: Collaboration -------------------
with tab2:
    render_chat_interface() # This function is from chat_system.py

# ------------------- TAB 3: Reports -------------------
with tab3:
    st.subheader("Medical Reports & Analytics History")
    st.markdown("### Analysis History")
    recent_analyses = get_latest_analyses(limit=10)
    if recent_analyses:
        for idx, analysis in enumerate(recent_analyses, 1):
            patient_id_display = analysis.get('patient_details', {}).get('patient_id', 'N/A')
            with st.expander(f"{idx}. {analysis.get('filename', 'Unknown')} (Patient ID: {patient_id_display}) - {analysis.get('date', '')[:10]}"):
                st.markdown(analysis.get("analysis", "No analysis available"))
                if analysis.get("findings"):
                    st.markdown("**Key Findings:**")
                    for i, f in enumerate(analysis["findings"], 1):
                        st.markdown(f"{i}. {f}")
                # Display quantitative analysis if available from history
                if analysis.get("quantitative_analysis"):
                    st.markdown("**Automated Quantitative Analysis:**")
                    q_results = analysis["quantitative_analysis"]
                    st.markdown(f"• Image Dimensions: {q_results.get('image_dimensions', 'N/A')}")
                    st.markdown(f"• Average Pixel Intensity: {q_results.get('average_intensity', 'N/A')}")
                    st.markdown(f"• Standard Deviation of Pixel Intensity: {q_results.get('std_dev_intensity', 'N/A')}")
                    if q_results.get("lesion_detected"):
                        st.markdown(f"• **Potential Lesion Detected:** Yes")
                        st.markdown(f"  - Simulated Size: {q_results.get('lesion_size_cm', 'N/A')} cm")
                        st.markdown(f"  - Simulated Density (HU): {q_results.get('lesion_density_hu', 'N/A')}")
                    else:
                        st.markdown(f"• Potential Lesion Detected: No (based on initial scan)")

                col_report_gen, col_view_details = st.columns(2)
                with col_report_gen:
                    if st.button(f"Generate Report #{idx}", key=f"gen_report_{idx}"):
                        # Re-generate with current XAI settings and references
                        # Note: xai_content and xai_image won't be available from saved analysis directly
                        # For a full production system, these would be stored or re-generated on demand.
                        # For this demo, we'll generate a basic report from saved data.
                        pdf_buffer = generate_report_with_xai(
                            analysis, 
                            include_references=include_references,
                            report_sections_to_include=st.session_state.report_sections # Use current selections for historical reports
                        )
                        b64_pdf = base64.b64encode(pdf_buffer.read()).decode()
                        report_name = f"report_{analysis.get('id', 'unknown')[:8]}.pdf"
                        href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="{report_name}">Download Report</a>'
                        st.markdown(href, unsafe_allow_html=True)
                        add_to_audit_log("Report downloaded from history", f"Analysis ID: {analysis.get('id')[:8]}, File: {analysis.get('filename')}")
    else:
        st.info("No previous analyses found. Upload and analyze an image to get started.")

    st.markdown("### Platform Statistics")
    if st.button("Generate Comprehensive Statistics", key="generate_comprehensive_stats"):
        stats_report = generate_statistics_report()
        if stats_report:
            b64_pdf = base64.b64encode(stats_report.read()).decode()
            href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="comprehensive_statistics.pdf">Download PDF</a>'
            st.markdown(href, unsafe_allow_html=True)
        add_to_audit_log("Generated comprehensive statistics report")

# ------------------- TAB 4: Audit Log -------------------
with tab4:
    st.subheader("Platform Audit Log")
    st.markdown("This log tracks key actions performed within this session.")
    if st.session_state.audit_log:
        for entry in reversed(st.session_state.audit_log): # Show most recent first
            st.code(entry)
    else:
        st.info("No actions recorded yet for this session.")

