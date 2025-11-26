from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os, json, re, uuid
from datetime import datetime
import pdfplumber, pytesseract
from pdf2image import convert_from_path

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max
app.config['INQUIRIES_FOLDER'] = 'inquiries'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['INQUIRIES_FOLDER'], exist_ok=True)


# === Helper: Extract Arabic text from PDF ===
def extract_text_from_pdf(pdf_path):
    texts = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                if len(text) < 40:  # Fallback to OCR
                    images = convert_from_path(pdf_path, first_page=page.page_number, last_page=page.page_number)
                    text = pytesseract.image_to_string(images[0], lang="ara+eng")
                texts.append(text)
    except Exception as e:
        print(f"Error extracting PDF: {e}")
    return "\n\n".join(texts)

# === New Route: Submit Tax Inquiry ===
@app.route("/submit-inquiry", methods=["POST"])
def submit_inquiry():
    try:
        # جمع البيانات
        data = {
            'regNumber': request.form.get('regNumber'),
            'fullName': request.form.get('fullName'),
            'mobile': request.form.get('mobile'),
            'email': request.form.get('email'),
            'taxCategory': request.form.get('taxCategory'),
            'inquiryType': request.form.get('inquiryType'),
            'inquiryText': request.form.get('inquiryText'),
            'refNumber': request.form.get('refNumber'),
            'timestamp': datetime.now().isoformat()
        }

        # توليد رقم تتبع
        tracking_number = f"TF-{datetime.now().strftime('%Y%m')}-{str(uuid.uuid4())[:8].upper()}"
        data['trackingNumber'] = tracking_number

        # حفظ الملفات
        files_info = []
        
        # الملف الرئيسي
        if 'mainFile' in request.files:
            main_file = request.files['mainFile']
            if main_file.filename:
                filename = secure_filename(f"{tracking_number}_main_{main_file.filename}")
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                main_file.save(file_path)
                
                # استخراج النص إذا كان PDF
                extracted_text = ""
                if filename.endswith('.pdf'):
                    extracted_text = extract_text_from_pdf(file_path)
                
                files_info.append({
                    'type': 'main',
                    'filename': filename,
                    'path': file_path,
                    'extracted_text': extracted_text[:500]  # أول 500 حرف
                })

        # الملفات الإضافية
        for key in request.files:
            if key.startswith('additionalFile'):
                file = request.files[key]
                if file.filename:
                    filename = secure_filename(f"{tracking_number}_add_{file.filename}")
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(file_path)
                    files_info.append({
                        'type': 'additional',
                        'filename': filename,
                        'path': file_path
                    })

        data['files'] = files_info

        # حفظ الطلب في JSON
        inquiry_file = f"inquiries/{tracking_number}.json"
        with open(inquiry_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        # TODO: هنا نضيف AI Classification و Workflow Routing

        return jsonify({
            'success': True,
            'trackingNumber': tracking_number,
            'message': 'تم استلام طلبك بنجاح! سيتم الرد خلال 24-48 ساعة',
            'estimatedResponse': '24-48 ساعة'
        }), 200

    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500


# === Old upload route (keep for compatibility) ===
@app.route("/upload", methods=["POST"])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "لم يتم رفع أي ملف"}), 400

    file = request.files['file']
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    text = extract_text_from_pdf(file_path)
    
    return jsonify({
        "message": "✅ تم استخراج النص من الملف بنجاح!",
        "preview": text[:500]
    })


if __name__ == "__main__":
    app.run(debug=True)