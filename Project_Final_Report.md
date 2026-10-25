# AI Face Mask Detection & Compliance Monitoring System
## Final Project Technical & Execution Report

---

## 1. Project Overview & Objectives

### What is the Project?
The **AI Face Mask Detection & Compliance Monitoring System** is an end-to-end computer vision solution designed to detect human faces in digital images and video feeds, and accurately classify their mask compliance status into three distinct categories:

1. 🟢 **Wearing Mask (`Class 0`):** Full compliance where the face mask completely covers the nose, mouth, and chin.
2. 🟡 **Wore Mask Incorrectly (`Class 1`):** Partial compliance where a mask is present but worn improperly (e.g., pulled below the nostrils or resting under the chin).
3. 🔴 **No Mask (`Class 2`):** Zero compliance where the face is completely bare and exposed.

### Purpose & Real-World Utility
Designed for deployment in public venues, healthcare facilities, schools, transit terminals, and commercial offices, this system provides automated compliance audits with sub-35ms latency on standard consumer laptop CPUs without requiring specialized GPU accelerators.

---

## 2. Project Architecture & Included Components

The project repository has been organized into a lean, modular structure containing only the assets required for execution, evaluation, and deployment:

```
Face Mask Detection/
├── app.py                      # Flask Web Server & API backend
├── requirements.txt            # Python environment dependencies
├── README.md                   # Project documentation & user guide
├── Project_Final_Report.md     # Comprehensive technical & execution report
│
├── models/
│   └── best.pt                 # Optimized 10-epoch YOLOv8n trained weights (6.2 MB)
│
├── templates/
│   └── index.html              # Clean, modern single-page front-end interface
│
├── static/
│   ├── style.css               # Glassmorphic dark-mode CSS design system
│   └── app.js                  # Client-side drag-and-drop & asynchronous API logic
│
├── scripts/
│   ├── infer.py                # Command-line single & batch image inference pipeline
│   ├── evaluate.py             # Standalone model benchmark & validation script
│   └── train.py                # PyTorch / Ultralytics model training pipeline
│
└── dataset/                    # Standardized YOLO formatted dataset
    ├── data.yaml               # Dataset configuration & class definitions
    ├── train/ (images, labels) # 2,776 training images & annotations
    ├── valid/ (images, labels) # 805 validation images & annotations
    └── test/  (images, labels) # 75 held-out test images & annotations
```

---

## 3. Technology Stack & Component Mapping

| Technology / Library | Version / Spec | Where & How It Is Used |
|---|---|---|
| **Ultralytics YOLOv8** | `v8.4.163` | **Core Detection Model:** Implements YOLOv8 Nano (`yolov8n`), an anchor-free object detection network with C2f feature extractors and decoupled classification/regression heads. Used in `models/best.pt`, `app.py`, `scripts/infer.py`, and `scripts/evaluate.py`. |
| **PyTorch (`torch` & `torchvision`)** | `2.11.0+cpu` | **Deep Learning Framework:** Manages neural network weights, forward tensor passes, non-maximum suppression (NMS), and CPU hardware optimizations. |
| **OpenCV (`cv2`)** | `4.11.0+` | **Computer Vision & Image Processing:** Decodes raw binary image uploads, handles color conversion, resizes frames, renders styled corner-bracket bounding boxes, and overlays summary dashboard banners. |
| **NumPy** | `1.26.0+` | **Data Transformation:** Converts binary multipart form upload streams into contiguous memory byte arrays (`np.frombuffer`) for OpenCV decoding. |
| **Flask** | `2.2.3` | **Web Application Server:** Hosts the WSGI application (`app.py`), renders HTML templates, and exposes the `/predict` multipart REST API for single-image and batch-folder uploads. |
| **HTML5 & Semantic Web** | Native | **UI Layout:** Implements drag-and-drop target zones, native folder selection (`webkitdirectory`), interactive range sliders, and image galleries in `templates/index.html`. |
| **Vanilla CSS3** | Custom System | **Design System:** Dark-mode glassmorphic styling (`static/style.css`), glowing status badges, responsive card layouts, and micro-animations with zero external CSS frameworks (no bloated Tailwind/Bootstrap). |
| **JavaScript (ES6+)** | Native Vanilla | **Client Interactivity:** Manages file drag-and-drop events, dynamic slider inputs, asynchronous `fetch` calls, Base64 image rendering, and modal zooms in `static/app.js`. |
| **PyYAML** | `6.0+` | **Configuration Parser:** Parses `dataset/data.yaml` to dynamically resolve class label mappings and data splits. |

---

## 4. Model Performance & Evaluation Summary

Evaluated on the full validation dataset split (**805 images, 1,252 face annotations**) at 416px resolution:

| Metric | Result | Operational Meaning |
|---|:---:|---|
| **Overall mAP@50** | **72.6%** | High overall detection and localization reliability |
| **Overall mAP@50-95** | **50.6%** | Strict localization quality across high overlap thresholds |
| **Overall Precision** | **79.5%** | Nearly 8 out of 10 detections are true positive faces |
| **Overall Recall** | **65.5%** | Captures nearly 2 out of 3 faces in diverse crowd scenes |
| **`Wearing Mask` mAP@50** | **75.1%** | Accurately tracks surgical, cloth, and N95 masks |
| **`Wore Mask Incorrectly` mAP@50** | **50.2%** | **Surged +18.6%** from prototype; recall doubled to 40.3% |
| **`No Mask` mAP@50** | **92.5%** | Extremely high reliability for detecting bare faces |
| **CPU Latency** | **~25–35 ms** | **30–40 FPS throughput** on standard 12th Gen Intel Core i5 |

---

## 5. How to Execute the Project

### Prerequisites
Ensure your terminal is in the project root directory and your Python environment is activated:
```powershell
cd "c:\Users\dhana\OneDrive\Desktop\Face Mask Detection"
```

---

### Method 1: Launch the Interactive Web User Interface (Recommended)

1. Start the Flask server:
   ```powershell
   python app.py
   ```
2. Open your web browser at:
   👉 **[http://127.0.0.1:5000](http://127.0.0.1:5000)**
3. Choose either:
   - **Single Image:** Upload or drag-and-drop an individual photo.
   - **Folder / Batch:** Select an entire directory of photos to process all of them at once.
4. Adjust the **Confidence Threshold** (default: `0.40`) or **IoU Overlap** (default: `0.45`) if needed.
5. Click **Detect Face Masks**. The annotated photos, class breakdown counts, and live execution logs appear on screen.

---

### Method 2: Command-Line Single Image Inference

Run detection on any individual image file:
```powershell
python scripts/infer.py "path\to\your_image.jpg"
```

*Example with custom output path and confidence:*
```powershell
python scripts/infer.py "C:\Users\dhana\Downloads\unseen data\maksssksksss140.png" --conf 0.40 --output runs/face_mask/predictions
```

---

### Method 3: Command-Line Batch Folder Inference

Run detection across an entire folder of photos:
```powershell
python scripts/infer.py --source "C:\Users\dhana\Downloads\unseen data" --conf 0.40 --limit 20
```

---

### Method 4: Evaluate Model on Validation or Test Splits

Run a complete benchmark evaluation on the test or validation datasets:
```powershell
# Evaluate on the validation split (805 images):
python scripts/evaluate.py --split val

# Evaluate on the held-out test split (75 images):
python scripts/evaluate.py --split test
```

---

### Method 5: Retrain or Fine-Tune the Model

If you ever wish to train further on new data:
```powershell
python scripts/train.py --mode baseline --epochs 10 --imgsz 416 --batch 4 --name custom_run
```

---

## 6. Summary of Key Strengths & Operational Advice

1. **Optimized False-Positive Suppression:** By applying `conf = 0.40` and `iou = 0.45`, phantom boxes on clothing and duplicate overlapping boxes on the same face are eliminated.
2. **True CPU Portability:** Operates at **30+ FPS** on standard consumer laptops with zero external GPU hardware requirements.
3. **Dual Execution Interfaces:** Offers both an intuitive web dashboard for non-technical users and an automated CLI pipeline for batch processing and scripts.
