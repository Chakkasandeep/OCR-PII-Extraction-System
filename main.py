import streamlit as st
import cv2
import numpy as np
from google.cloud import vision
from google.oauth2 import service_account
import re
import json
import spacy
from PIL import Image
import io
import os
from dotenv import load_dotenv

# ========== LOAD ENVIRONMENT VARIABLES ==========
load_dotenv()

def get_credentials():
    try:
        credentials_dict = {
            "type": st.secrets["GCP_TYPE"],
            "project_id": st.secrets["GCP_PROJECT_ID"],
            "private_key_id": st.secrets["GCP_PRIVATE_KEY_ID"],
            "private_key": st.secrets["GCP_PRIVATE_KEY"].replace('\\n', '\n'),
            "client_email": st.secrets["GCP_CLIENT_EMAIL"],
            "client_id": st.secrets["GCP_CLIENT_ID"],
            "auth_uri": st.secrets["GCP_AUTH_URI"],
            "token_uri": st.secrets["GCP_TOKEN_URI"],
            "auth_provider_x509_cert_url": st.secrets["GCP_AUTH_PROVIDER_CERT_URL"],
            "client_x509_cert_url": st.secrets["GCP_CLIENT_CERT_URL"],
            "universe_domain": st.secrets["GCP_UNIVERSE_DOMAIN"]
        }

        credentials = service_account.Credentials.from_service_account_info(credentials_dict)
        return credentials

    except KeyError as e:
        st.error(f"❌ Missing secret: {e}")
        st.stop()

    except Exception as e:
        st.error(f"❌ Credential loading error: {str(e)}")
        st.stop()

# ========== 1. IMAGE PREPROCESSING ==========
def preprocess_image(image):
    """Apply all preprocessing steps for better OCR accuracy"""
    img = np.array(image)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Noise removal (bilateral filter preserves edges)
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Adaptive thresholding (better for handwriting)
    thresh = cv2.adaptiveThreshold(
        denoised, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    # Deskew (correct rotation)
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) > 0:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        
        (h, w) = thresh.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(thresh, M, (w, h), 
                                 flags=cv2.INTER_CUBIC, 
                                 borderMode=cv2.BORDER_REPLICATE)
    else:
        rotated = thresh
    
    # Sharpen
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(rotated, -1, kernel)
    
    return sharpened

# ========== 2. GOOGLE CLOUD VISION OCR ==========
def perform_ocr(image_bytes, credentials):
    """Extract text using Google Cloud Vision API"""
    try:
        client = vision.ImageAnnotatorClient(credentials=credentials)
        
        image = vision.Image(content=image_bytes)
        
        # Use document_text_detection for better handwriting support
        response = client.document_text_detection(image=image)
        
        if response.error.message:
            raise Exception(f"API Error: {response.error.message}")
        
        # Extract full text
        full_text = response.full_text_annotation.text if response.full_text_annotation else ""
        
        # Extract word-level bounding boxes for redaction
        words_data = []
        if response.full_text_annotation:
            for page in response.full_text_annotation.pages:
                for block in page.blocks:
                    for paragraph in block.paragraphs:
                        for word in paragraph.words:
                            word_text = ''.join([symbol.text for symbol in word.symbols])
                            vertices = [(v.x, v.y) for v in word.bounding_box.vertices]
                            words_data.append({
                                'text': word_text,
                                'box': vertices
                            })
        
        return full_text, words_data
    
    except Exception as e:
        st.error(f"OCR Error: {str(e)}")
        return "", []

# ========== 3. TEXT CLEANING ==========
def clean_text(text):
    """Clean extracted text"""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s@.,:/()-]', '', text)
    return text.strip()

# ========== 4. PII DETECTION ==========
def detect_pii(text):
    """Detect all PII using hybrid approach"""
    pii_data = {
        'phones': [],
        'emails': [],
        'names': [],
        'addresses': [],
        'dates': [],
        'ids': [],
        'medical_keywords': []
    }
    
    # REGEX PATTERNS
    phone_pattern = r'\b\d{10}\b|\b\d{5}\s?\d{5}\b|\b\+91\s?\d{10}\b'
    phones = re.findall(phone_pattern, text)
    pii_data['phones'] = list(set(phones))
    
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, text)
    pii_data['emails'] = list(set(emails))
    
    date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b'
    dates = re.findall(date_pattern, text)
    pii_data['dates'] = list(set(dates))
    
    aadhar_pattern = r'\b\d{4}\s?\d{4}\s?\d{4}\b'
    ids = re.findall(aadhar_pattern, text)
    pii_data['ids'] = list(set(ids))
    
    # KEYWORD-BASED EXTRACTION
    keywords = {
        'name': [r'Name\s*:?\s*([A-Za-z\s]+)', r'Patient\s*:?\s*([A-Za-z\s]+)'],
        'address': [r'Address\s*:?\s*([^\n]+)', r'Addr\s*:?\s*([^\n]+)'],
        'age': [r'Age\s*:?\s*(\d+)', r'(\d+)\s*years?'],
        'doctor': [r'Dr\.?\s*([A-Za-z\s]+)', r'Doctor\s*:?\s*([A-Za-z\s]+)']
    }
    
    for key, patterns in keywords.items():
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                if key == 'name' or key == 'doctor':
                    pii_data['names'].extend(matches)
                elif key == 'address':
                    pii_data['addresses'].extend(matches)
    
    # SPACY NER
    try:
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                pii_data['names'].append(ent.text)
            elif ent.label_ in ["GPE", "LOC"]:
                pii_data['addresses'].append(ent.text)
            elif ent.label_ == "DATE":
                pii_data['dates'].append(ent.text)
    except Exception as e:
        st.warning(f"⚠️ spaCy NER not available: {str(e)}")
    
    # Medical keywords
    medical_terms = ['patient', 'doctor', 'dr', 'hospital', 'clinic', 'diagnosis', 
                     'prescription', 'medicine', 'treatment', 'bp', 'pulse']
    found_terms = [term for term in medical_terms if term.lower() in text.lower()]
    pii_data['medical_keywords'] = found_terms
    
    # Clean duplicates
    for key in pii_data:
        pii_data[key] = list(set([str(x).strip() for x in pii_data[key] if x]))
    
    return pii_data

# ========== 5. REDACTION ==========
def redact_image(image, words_data, pii_data):
    """Black out PII in the image"""
    img = np.array(image)
    
    all_pii = []
    for category in ['phones', 'emails', 'names', 'ids']:
        all_pii.extend(pii_data.get(category, []))
    
    for word_info in words_data:
        word = word_info['text']
        for pii in all_pii:
            if word in str(pii):
                box = word_info['box']
                points = np.array(box, dtype=np.int32)
                cv2.fillPoly(img, [points], (0, 0, 0))
    
    return Image.fromarray(img)

# ========== STREAMLIT UI ==========
st.set_page_config(page_title="OCR + PII Extraction", page_icon="🔍", layout="wide")

st.title("🔍 Medical Document OCR + PII Extraction")
st.markdown("### 📄 Upload handwritten medical documents for automatic text extraction and PII detection")

# Load credentials
credentials = get_credentials()

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    show_preprocessed = st.checkbox("Show preprocessed image", value=True)
    enable_redaction = st.checkbox("Enable PII redaction", value=True)
    st.markdown("---")
    st.markdown("**🔐 Features:**")
    st.markdown("✅ Google Vision OCR")
    st.markdown("✅ Image preprocessing")
    st.markdown("✅ PII detection (Regex + NER)")
    st.markdown("✅ Redaction support")
    st.markdown("---")
    st.success("✅ Credentials loaded!")

# Main content
uploaded_file = st.file_uploader("📤 Upload a handwritten document (JPEG/PNG)", 
                                  type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 Original Image")
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)
    
    with col2:
        if show_preprocessed:
            st.subheader("🔧 Preprocessed Image")
            preprocessed = preprocess_image(image)
            st.image(preprocessed, use_container_width=True, channels="GRAY")
    
    if st.button("🚀 Extract Text & Detect PII", type="primary"):
        with st.spinner("Processing..."):
            preprocessed_img = preprocess_image(image)
            _, buffer = cv2.imencode('.jpg', preprocessed_img)
            image_bytes = buffer.tobytes()
            
            extracted_text, words_data = perform_ocr(image_bytes, credentials)
            
            if extracted_text:
                cleaned_text = clean_text(extracted_text)
                pii_results = detect_pii(cleaned_text)
                
                st.success("✅ Processing Complete!")
                
                tab1, tab2, tab3, tab4 = st.tabs(["📝 Extracted Text", "🔍 PII Data", "📊 Summary", "🖼️ Redacted"])
                
                with tab1:
                    st.subheader("Extracted Text")
                    st.text_area("Full Text", cleaned_text, height=300)
                    st.download_button(
                        "💾 Download Text",
                        cleaned_text,
                        file_name="extracted_text.txt"
                    )
                
                with tab2:
                    st.subheader("Detected PII")
                    
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        st.markdown("**📞 Phone Numbers**")
                        st.write(pii_results['phones'] if pii_results['phones'] else "None detected")
                        
                        st.markdown("**📧 Email Addresses**")
                        st.write(pii_results['emails'] if pii_results['emails'] else "None detected")
                        
                        st.markdown("**👤 Names**")
                        st.write(pii_results['names'] if pii_results['names'] else "None detected")
                    
                    with col_b:
                        st.markdown("**🏠 Addresses/Locations**")
                        st.write(pii_results['addresses'] if pii_results['addresses'] else "None detected")
                        
                        st.markdown("**📅 Dates**")
                        st.write(pii_results['dates'] if pii_results['dates'] else "None detected")
                        
                        st.markdown("**🆔 ID Numbers**")
                        st.write(pii_results['ids'] if pii_results['ids'] else "None detected")
                    
                    json_data = json.dumps(pii_results, indent=2)
                    st.download_button(
                        "💾 Download PII Data (JSON)",
                        json_data,
                        file_name="pii_data.json",
                        mime="application/json"
                    )
                
                with tab3:
                    st.subheader("📊 Detection Summary")
                    
                    total_pii = sum(len(v) for v in pii_results.values() if isinstance(v, list))
                    
                    m1, m2, m3, m4 = st.columns(4)
                    
                    with m1:
                        st.metric("Total PII Found", total_pii)
                    with m2:
                        st.metric("Phone Numbers", len(pii_results['phones']))
                    with m3:
                        st.metric("Names", len(pii_results['names']))
                    with m4:
                        st.metric("Dates", len(pii_results['dates']))
                    
                    if pii_results['medical_keywords']:
                        st.info(f"🏥 Medical context: {', '.join(pii_results['medical_keywords'])}")
                
                with tab4:
                    if enable_redaction:
                        st.subheader("🖼️ Redacted Image")
                        redacted_img = redact_image(image, words_data, pii_results)
                        st.image(redacted_img, use_container_width=True)
                        
                        buf = io.BytesIO()
                        redacted_img.save(buf, format='PNG')
                        st.download_button(
                            "💾 Download Redacted Image",
                            buf.getvalue(),
                            file_name="redacted_image.png",
                            mime="image/png"
                        )
                    else:
                        st.info("Enable redaction in sidebar")
            
            else:
                st.error("❌ Failed to extract text. Try another image.")

else:
    st.info("👆 Upload an image to get started")
    
    with st.expander("📋 What this system does"):
        st.markdown("""
        **OCR + PII extraction pipeline:**
        
        1. **Preprocesses** handwritten documents (denoising, thresholding, deskewing)
        2. **Extracts text** using Google Cloud Vision API (85-95% accuracy)
        3. **Detects PII** using Regex + spaCy NER + medical keywords
        4. **Redacts sensitive information** from images
        5. **Exports results** as text, JSON, and redacted image
        
        Perfect for medical records, clinic notes, and handwritten forms!
        """)
