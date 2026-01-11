"""
Demo Instruction Processor API

Processes PDF or Word documents containing product demo instructions.
Extracts text using OCR, detects images, generates image descriptions using GPT-4o,
and outputs structured step-by-step instructions in JSON format.
"""

import os
import sys
import json
import base64
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv()

try:
    import fitz  # PyMuPDF
except ImportError:
    print("PyMuPDF not found. Please install it using:")
    print("  pip install pymupdf")
    sys.exit(1)

try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    print("python-docx not found. Word document support will be disabled.")
    print("To enable, install: pip install python-docx")
    DOCX_AVAILABLE = False

try:
    from PIL import Image
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    print("OCR dependencies not found. OCR will be disabled.")
    print("To enable OCR, install: pip install pillow pytesseract")
    OCR_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    print("OpenAI not found. Image description will be disabled.")
    print("To enable, install: pip install openai")
    OPENAI_AVAILABLE = False

# Configure Tesseract path for Windows
TESSERACT_FOUND = False
if OCR_AVAILABLE and sys.platform == 'win32':
    import shutil
    tesseract_path = shutil.which('tesseract')
    if tesseract_path:
        TESSERACT_FOUND = True
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
    else:
        common_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            r'C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'.format(os.getenv('USERNAME')),
        ]
        for path in common_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                TESSERACT_FOUND = True
                break
elif OCR_AVAILABLE:
    TESSERACT_FOUND = True

# Initialize OpenAI client
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_client = None
if OPENAI_AVAILABLE and openai_api_key:
    openai_client = OpenAI(api_key=openai_api_key)
elif OPENAI_AVAILABLE:
    print("⚠️  WARNING: OPENAI_API_KEY not found in environment. Image description will be disabled.")

# Initialize FastAPI
app = FastAPI(
    title="Demo Instruction Processor API",
    description="Process PDF/Word documents to generate structured demo instructions",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def extract_text_with_ocr(page_image: Image.Image) -> str:
    """Extract text from an image using OCR."""
    if not OCR_AVAILABLE or not TESSERACT_FOUND:
        return ""
    
    try:
        ocr_text = pytesseract.image_to_string(page_image, lang='eng')
        return ocr_text.strip()
    except Exception as e:
        print(f"OCR error: {str(e)}")
        return ""


def get_images_from_pdf_page(page) -> List[Image.Image]:
    """Extract images from a PDF page."""
    images = []
    try:
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            try:
                xref = img[0]
                base_image = page.parent.extract_image(xref)
                image_bytes = base_image["image"]
                image = Image.open(BytesIO(image_bytes))
                images.append(image)
            except Exception as e:
                print(f"  ⚠ Error extracting image {img_index}: {str(e)}")
                continue
    except Exception as e:
        print(f"  ⚠ Error getting images from page: {str(e)}")
    
    return images


def get_images_from_docx_paragraph(paragraph) -> List[Image.Image]:
    """Extract images from a Word document paragraph."""
    images = []
    try:
        for run in paragraph.runs:
            if run._element.xpath('.//a:blip'):
                # Image found in run
                for rel in run.part.rels.values():
                    if "image" in rel.target_ref:
                        try:
                            image_bytes = rel.target_part.blob
                            image = Image.open(BytesIO(image_bytes))
                            images.append(image)
                        except Exception as e:
                            print(f"  ⚠ Error extracting image: {str(e)}")
                            continue
    except Exception as e:
        print(f"  ⚠ Error getting images from paragraph: {str(e)}")
    
    return images


def encode_image_to_base64(image: Image.Image) -> str:
    """Encode PIL Image to base64 string."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


def describe_image_with_gpt4o(image: Image.Image, page_text: str) -> str:
    """Use GPT-4o to describe an image in the context of the page."""
    if not openai_client:
        return "Image description unavailable (OpenAI not configured)"
    
    try:
        # Encode image to base64
        base64_image = encode_image_to_base64(image)
        
        # Create prompt
        prompt = f"""Analyze this screenshot/image from a product demo instruction document. 
The page contains the following text content:
{page_text[:1000]}

Describe what you see in this image in detail, focusing on:
1. UI elements visible (buttons, fields, menus, etc.)
2. Text labels and their positions
3. Visual indicators (colors, icons, etc.)
4. The overall context and what action the user should take

Provide a clear, actionable description that can be used to create step-by-step instructions."""

        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500,
            temperature=0.3
        )
        
        description = response.choices[0].message.content.strip()
        return description
        
    except Exception as e:
        print(f"  ⚠ Error describing image with GPT-4o: {str(e)}")
        return f"Image description unavailable: {str(e)}"


def process_page_with_ai(page_text: str, image_descriptions: List[str], existing_steps: List[Dict], next_step_number: int = 1) -> List[Dict]:
    """Process a single page with GPT-4o to extract steps."""
    if not openai_client:
        raise HTTPException(
            status_code=500,
            detail="OpenAI API key not configured. Please set OPENAI_API_KEY in environment."
        )
    
    # Build context about existing steps
    existing_context = ""
    if existing_steps:
        existing_context = f"\n\nPreviously extracted steps (for reference, do not duplicate):\n"
        for step in existing_steps[-3:]:  # Only show last 3 steps for context
            existing_context += f"Step {step['step']}: {step['title']} - {step['description']}\n"
    
    # Always specify the starting step number (even if it's 1)
    existing_context += f"\n\nCRITICAL: You MUST start numbering new steps from {next_step_number}. Ignore any step numbers mentioned in the document content - use {next_step_number} as the first step number for the steps you extract from this page."
    
    # Build image context
    image_context = ""
    if image_descriptions:
        image_context = "\n\nImage descriptions from this page:\n"
        for idx, desc in enumerate(image_descriptions, 1):
            image_context += f"Image {idx}: {desc}\n"
    
    # Create prompt
    system_prompt = """You are an expert at analyzing product demo instruction documents. 
Your task is to extract step-by-step instructions from the provided page content.

IMPORTANT RULES:
1. Extract ALL steps from the current page (there may be multiple steps per page)
2. Each step should have: step number, title, description, and prompt
3. The prompt should be detailed and actionable, describing exactly what the user should do
4. If images are present, incorporate the image descriptions into the step instructions
5. Connect image descriptions with the text content to create comprehensive steps
6. Return ONLY valid JSON, no additional text
7. CRITICAL: Use the step number provided in the instructions - IGNORE any step numbers mentioned in the document content itself
8. Be specific about UI elements, buttons, fields, and actions"""

    user_prompt = f"""Extract step-by-step instructions from this page content:

{page_text}

{image_context}

{existing_context}

Return a JSON object with a "steps" key containing an array of step objects. Each step object must have:
- step: integer (continue numbering from existing steps)
- title: string (brief title)
- description: string (what this step is about)
- prompt: string (detailed instruction on what to do, incorporating image details if available)

Example format:
{{
  "steps": [
    {{
      "step": 1,
      "title": "Starting a project",
      "description": "Click on Start your project",
      "prompt": "Click on Start your project which is present as a green button"
    }},
    {{
      "step": 2,
      "title": "Credentials Input",
      "description": "Enter the email and password you have",
      "prompt": "Click on the button which says email and enter the email 'mathewmarsha...'"
    }}
  ]
}}

Return ONLY the JSON object with the "steps" key, nothing else."""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=2000,
            response_format={"type": "json_object"}
        )
        
        # Parse response
        result_text = response.choices[0].message.content.strip()
        
        # Parse JSON response
        try:
            parsed = json.loads(result_text)
            if isinstance(parsed, dict) and "steps" in parsed:
                return parsed["steps"]
            elif isinstance(parsed, list):
                return parsed
            elif isinstance(parsed, dict):
                # If it's a dict, try to find array values
                for key, value in parsed.items():
                    if isinstance(value, list):
                        return value
                return []
            else:
                return []
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            if "```json" in result_text:
                start = result_text.find("```json") + 7
                end = result_text.find("```", start)
                result_text = result_text[start:end].strip()
            elif "```" in result_text:
                start = result_text.find("```") + 3
                end = result_text.find("```", start)
                result_text = result_text[start:end].strip()
            
            parsed = json.loads(result_text)
            if isinstance(parsed, dict) and "steps" in parsed:
                return parsed["steps"]
            elif isinstance(parsed, list):
                return parsed
            return []
            
    except Exception as e:
        print(f"  ⚠ Error processing page with AI: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing page with AI: {str(e)}"
        )


def process_pdf_document(pdf_bytes: bytes, filename: str) -> List[Dict]:
    """Process a PDF document page by page."""
    all_steps = []
    
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pdf_name = Path(filename).stem
        
        print(f"\n📄 Processing PDF: {filename} ({len(doc)} pages)")
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            print(f"\n  📖 Processing page {page_num + 1}/{len(doc)}")
            
            # Get page as image for OCR
            mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            page_image = Image.open(BytesIO(img_data))
            
            # Extract text using OCR (always use OCR)
            page_text = extract_text_with_ocr(page_image)
            if not page_text:
                # Fallback to regular text extraction
                page_text = page.get_text()
            
            print(f"    ✓ Extracted {len(page_text)} characters of text")
            
            # Get images from page
            images = get_images_from_pdf_page(page)
            print(f"    ✓ Found {len(images)} image(s)")
            
            # Describe images with GPT-4o
            image_descriptions = []
            for img_idx, image in enumerate(images):
                print(f"    🔍 Describing image {img_idx + 1}...")
                description = describe_image_with_gpt4o(image, page_text)
                image_descriptions.append(description)
                print(f"      ✓ Description generated")
            
            # Process page with AI to extract steps
            if page_text.strip() or image_descriptions:
                print(f"    🤖 Processing page content with GPT-4o...")
                next_step = max([step["step"] for step in all_steps], default=0) + 1
                page_steps = process_page_with_ai(page_text, image_descriptions, all_steps, next_step)
                
                # Validate and normalize step numbers
                for idx, step in enumerate(page_steps):
                    if "step" not in step or "title" not in step or "description" not in step or "prompt" not in step:
                        print(f"    ⚠ Skipping invalid step: {step}")
                        continue
                    
                    # Normalize step number to be sequential (ignore what AI returned)
                    step["step"] = next_step + idx
                
                all_steps.extend(page_steps)
                print(f"    ✓ Extracted {len(page_steps)} step(s) from this page")
            else:
                print(f"    ⚠ Page has no text or images, skipping")
        
        doc.close()
        return all_steps
        
    except Exception as e:
        raise Exception(f"Error processing PDF {filename}: {str(e)}")


def process_docx_document(docx_bytes: bytes, filename: str) -> List[Dict]:
    """Process a Word document page by page (paragraph by paragraph)."""
    all_steps = []
    
    try:
        doc = Document(BytesIO(docx_bytes))
        print(f"\n📄 Processing Word document: {filename}")
        
        # Process document paragraph by paragraph
        current_page_text = ""
        current_page_images = []
        page_num = 1
        
        for para_idx, paragraph in enumerate(doc.paragraphs):
            para_text = paragraph.text.strip()
            
            # Get images from paragraph
            para_images = get_images_from_docx_paragraph(paragraph)
            
            # Accumulate text and images
            if para_text:
                current_page_text += para_text + "\n\n"
            
            if para_images:
                current_page_images.extend(para_images)
            
            # Process when we have substantial content or hit a page break
            # (Word doesn't have explicit pages, so we process in chunks)
            if len(current_page_text) > 500 or (para_idx > 0 and para_idx % 10 == 0):
                if current_page_text.strip() or current_page_images:
                    print(f"\n  📖 Processing section {page_num}")
                    print(f"    ✓ Extracted {len(current_page_text)} characters of text")
                    print(f"    ✓ Found {len(current_page_images)} image(s)")
                    
                    # Describe images
                    image_descriptions = []
                    for img_idx, image in enumerate(current_page_images):
                        print(f"    🔍 Describing image {img_idx + 1}...")
                        description = describe_image_with_gpt4o(image, current_page_text)
                        image_descriptions.append(description)
                        print(f"      ✓ Description generated")
                    
                    # Process with AI
                    print(f"    🤖 Processing section content with GPT-4o...")
                    next_step = max([step["step"] for step in all_steps], default=0) + 1
                    page_steps = process_page_with_ai(current_page_text, image_descriptions, all_steps, next_step)
                    
                    # Validate and normalize step numbers
                    for idx, step in enumerate(page_steps):
                        if "step" not in step or "title" not in step or "description" not in step or "prompt" not in step:
                            print(f"    ⚠ Skipping invalid step: {step}")
                            continue
                        
                        # Normalize step number to be sequential (ignore what AI returned)
                        step["step"] = next_step + idx
                    
                    all_steps.extend(page_steps)
                    print(f"    ✓ Extracted {len(page_steps)} step(s) from this section")
                
                # Reset for next section
                current_page_text = ""
                current_page_images = []
                page_num += 1
        
        # Process remaining content
        if current_page_text.strip() or current_page_images:
            print(f"\n  📖 Processing final section {page_num}")
            image_descriptions = []
            for img_idx, image in enumerate(current_page_images):
                print(f"    🔍 Describing image {img_idx + 1}...")
                description = describe_image_with_gpt4o(image, current_page_text)
                image_descriptions.append(description)
            
            print(f"    🤖 Processing section content with GPT-4o...")
            next_step = max([step["step"] for step in all_steps], default=0) + 1
            page_steps = process_page_with_ai(current_page_text, image_descriptions, all_steps, next_step)
            
            # Validate and normalize step numbers
            for idx, step in enumerate(page_steps):
                if "step" not in step or "title" not in step or "description" not in step or "prompt" not in step:
                    print(f"    ⚠ Skipping invalid step: {step}")
                    continue
                
                # Normalize step number to be sequential (ignore what AI returned)
                step["step"] = next_step + idx
            
            all_steps.extend(page_steps)
            print(f"    ✓ Extracted {len(page_steps)} step(s) from this section")
        
        return all_steps
        
    except Exception as e:
        raise Exception(f"Error processing Word document {filename}: {str(e)}")


@app.post("/process-demo-instructions")
async def process_demo_instructions(
    file: UploadFile = File(...)
):
    """
    Process a PDF or Word document containing demo instructions.
    
    Extracts text using OCR, detects images, generates image descriptions,
    and outputs structured step-by-step instructions in JSON format.
    
    Args:
        file: PDF or Word document file
    
    Returns:
        JSON response with structured steps
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    # Check file type
    filename_lower = file.filename.lower()
    is_pdf = filename_lower.endswith('.pdf')
    is_docx = filename_lower.endswith('.docx') or filename_lower.endswith('.doc')
    
    if not (is_pdf or is_docx):
        raise HTTPException(
            status_code=400,
            detail="File must be a PDF (.pdf) or Word document (.docx, .doc)"
        )
    
    if is_docx and not DOCX_AVAILABLE:
        raise HTTPException(
            status_code=400,
            detail="Word document support not available. Please install python-docx."
        )
    
    if not OCR_AVAILABLE or not TESSERACT_FOUND:
        raise HTTPException(
            status_code=400,
            detail="OCR is required but not available. Please install pillow, pytesseract, and Tesseract OCR."
        )
    
    if not openai_client:
        raise HTTPException(
            status_code=500,
            detail="OpenAI API key not configured. Please set OPENAI_API_KEY in environment."
        )
    
    try:
        # Read file contents
        contents = await file.read()
        
        # Process document
        if is_pdf:
            steps = process_pdf_document(contents, file.filename)
        else:
            steps = process_docx_document(contents, file.filename)
        
        print(f"\n✅ Processing complete: {len(steps)} total steps extracted")
        
        return JSONResponse(content={
            "status": "success",
            "filename": file.filename,
            "total_steps": len(steps),
            "steps": steps
        })
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error processing file: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing file: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "ocr_available": OCR_AVAILABLE and TESSERACT_FOUND,
        "openai_available": openai_client is not None,
        "docx_available": DOCX_AVAILABLE
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Demo Instruction Processor API",
        "version": "2.0.0",
        "endpoints": {
            "POST /process-demo-instructions": "Process PDF/Word document to generate structured demo steps",
            "GET /health": "Health check"
        },
        "ocr_available": OCR_AVAILABLE and TESSERACT_FOUND,
        "openai_available": openai_client is not None,
        "docx_available": DOCX_AVAILABLE
    }


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", "8080"))
    print(f"🚀 Starting Demo Instruction Processor API on port {port}")
    print(f"📄 OCR Available: {OCR_AVAILABLE and TESSERACT_FOUND}")
    print(f"🤖 OpenAI Available: {openai_client is not None}")
    print(f"📝 Word Doc Support: {DOCX_AVAILABLE}")
    
    uvicorn.run(app, host="0.0.0.0", port=port)
