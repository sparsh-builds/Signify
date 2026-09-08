# Signify | Biometric Forensic Signature Verification Engine

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.1%20CPU-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.9.0-5C3EE8.svg?logo=opencv&logoColor=white)](https://opencv.org)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![Render API](https://img.shields.io/badge/Render-Live%20API-46E3B7.svg?logo=render&logoColor=white)](https://signify-api-rg73.onrender.com/healthz)

Signify is an enterprise-grade, offline biometric signature verification and forensic intelligence platform. Unlike standard black-box neural networks, Signify pairs a 128-dimensional deep metric Siamese CNN with deterministic, explainable classical computer vision pipelines (Harris Corner micro-tremor analysis, vertical Crest-Trough waveforms, and 2D FFT presentation attack detection).

---

## Key Architectural Pillars

### 1. Multi-Metric Hybrid Fusion
The engine scores questioned specimens against genuine baselines using a calibrated three-tier metric pipeline:
* **Deep Metric Latent Projection (30%):** A custom 4-block Siamese CNN with Batch Normalization and Dropout transforms $1\times 64\times 64$ normalized binary stroke tensors into an $L_2$-normalized 128-dimensional latent vector to calculate cosine similarity.
* **Crest-Trough Vertical Waveforms (35%):** Morphological profile projection measures stroke distribution curvature and rhythm to catch mechanical tracing anomalies.
* **Harris Corner & ORB Micro-Tremor Detection (35%):** Keypoint variance algorithms identify unnatural pen stops, hesitation tremors, and structural corner deviations common in simulated forgeries.

### 2. Presentation Attack Detection (Anti-Spoofing PAD)
* **2D Fast Fourier Transform (FFT) Spectral Filtering:** Analyzes high-frequency periodic lattice energy across camera captures (`mode=photo`) to instantly reject digital screen replay attacks exhibiting Moiré interference patterns.

### 3. Forensic Document Examination UI
* **Astronomical Blink Comparator:** Alternates reference and test signatures at a perceptual 4 Hz refresh frequency, allowing human auditors to identify sub-pixel contour variations instantly.
* **Pen Pressure Reconstruction:** Synthetic gradient heatmaps approximate writing velocity and localized pressure hesitations.
* **Interactive Ghost Overlay:** Dynamic opacity sliders allow direct visual alignment of stroke topology.

### 4. Enterprise Compliance & Live Telemetry
* **Cryptographic SHA-256 Audit Trail:** Every verification produces a tamper-evident audit hash encoding the input images, metric parameters, verdict, and timestamp.
* **Executive Power BI Telemetry Stream:** A structured, tabular telemetry endpoint (`/powerbi/telemetry`) enables live data ingestion and KPI modeling in Power BI Desktop.
* **PDF Audit Certificate Export:** Client-side vector certificate generation using `html2pdf.js` for legal archiving and dispute resolution.

---

## System Architecture

```text
[ Document Scan / Camera Photo / Canvas Pad ]
                     │
                     ▼
       [ Image Preprocessing Pipeline ]
       ├── CLAHE Contrast Equalization
       ├── Otsu & Adaptive Gaussian Binarization
       ├── Morphological Despeckling
       └── Aspect-Ratio Bounding Box Normalization
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
  [ Classical Vision ]     [ Deep Metric Siamese CNN ]
  ├── Harris Corners       └── 128-d Latent Embedding Vector
  ├── Crest-Trough Wave            │
  └── 2D FFT Moiré Check           ▼
         │                 [ Cosine Similarity ]
         └───────────┬───────────┘
                     ▼
          [ Dynamic Risk Engine ]
   ├── Low Risk    (Threshold: 55% | KYC / Attendance)
   ├── Medium Risk (Threshold: 65% | Cheques < 50k)
   └── High Risk   (Threshold: 78% | Property / RTGS)
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
  [ Cryptographic Hash ]   [ Verification Telemetry ]
  └── SHA-256 Audit Seal   ├── SQLite / SQLAlchemy Audit Log
                           └── Power BI Direct Ingestion
