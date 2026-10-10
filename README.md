
# Burnsight
### AI-Driven Anomaly Detection in Component Burn-In & Screening

Burnsight is a web-based prototype developed for **Smart India Hackathon 2026**, addressing ISRO Problem Statement 26170.

The project explores how data analysis and machine learning can help identify unusual component behaviour during burn-in testing and estimate potential degradation before the end of the 168-hour test period.

Traditional screening often relies on fixed datasheet limits. Burnsight adds a lot-aware perspective by comparing components with the behaviour of their manufacturing lot and combining anomaly detection with early drift prediction.

**SIH 2026 | Team Burnsight | Team ID: 168888 | PS: 26170**

**[🌐Burnsight_Live Demo](https://burnsight.netlify.app/)**




## 📊 Dashboard Preview
<img width="1800" height="900" alt="Screenshot (1055)" src="https://github.com/user-attachments/assets/9ad500f0-6575-4085-a4dd-d895edaebd98" />



## 🎯 The Problem

High-reliability electronic components may undergo burn-in testing at elevated temperatures to identify potential defects before being used in demanding applications.

A component can remain within its datasheet limits while behaving differently from the rest of its manufacturing lot. Looking only at fixed limits may not reveal this unusual behaviour.

Burnsight explores a data-driven approach to identifying these deviations and predicting possible drift using early test measurements.



## ✨ Key Features

- **Lot-Aware Anomaly Detection:** Uses the median and Median Absolute Deviation (MAD) to identify components that behave unusually relative to their lot.
- **Early Drift Prediction:** Uses measurements from 0 and 24 hours to estimate component parameter values at 168 hours.
- **Risk-Based Screening:** Combines anomaly evidence and predicted drift to support PASS, REVIEW and REJECT decisions.
- **Explainable Results:** Presents parameter deviations, lot behaviour and prediction-related evidence to help users understand screening outcomes.
- **Dataset Upload:** Supports CSV and Excel files for component telemetry ingestion.
- **Data Visualization:** Displays component measurements and trends through interactive charts.



## ⚙️How It Works

### 1. Telemetry Input

The system works with component measurements recorded at 0h, 24h, 96h and 168h. It tracks six electrical parameters:

- Quiescent Drain Current 
- Leakage Current
- Threshold Voltage
- Propagation Delay
- Supply Current
- Operating Frequency


### 2. Lot-Aware Anomaly Detection

The system establishes a lot-level baseline using the median and Median Absolute Deviation (MAD). Components that deviate unusually from this baseline can be flagged for further inspection, even when individual readings remain within fixed limits.

<img width="500" height="250" alt="Screenshot (1056)" src="https://github.com/user-attachments/assets/d5a7c4ec-e245-4d7f-9ce2-b45aa14d35ae" />     <img width="500" height="250" alt="Screenshot (1057)" src="https://github.com/user-attachments/assets/fc980c86-873d-4991-bba0-9f27ac649831" />
<img width="500" height="250" alt="Screenshot (1058)" src="https://github.com/user-attachments/assets/9d75b6da-3ae3-4e23-beda-8cd52648972a" />     <img width="500" height="250" alt="Screenshot (1059)" src="https://github.com/user-attachments/assets/3c9c88d1-0ead-4eaf-b140-b7a268c5ee3b" />
<img width="500" height="250" alt="Screenshot (1060)" src="https://github.com/user-attachments/assets/b8e85bb3-48df-4e1c-aed4-1c166035cb9a" />     <img width="500" height="250" alt="Screenshot (1061)" src="https://github.com/user-attachments/assets/ddaf617c-715e-491f-866b-a8a71991890a" />


### 3. Early Drift Prediction

The prediction module uses early measurements at 0h and 24h to estimate the 168h parameter value. This helps identify components that may exhibit concerning changes during the remainder of the burn-in period.

<img width="1000" height="500" alt="Screenshot (1063)" src="https://github.com/user-attachments/assets/eb223c00-e6f7-42b2-a4ba-54057ba9e773" />


### 4. Screening and Review

The system combines the available anomaly evidence and predicted drift to support three screening outcomes:

- **PASS:** No configured screening condition has triggered a flag.
- **REVIEW:** The component requires further assessment.
- **REJECT:** The component meets the configured rejection criteria.

These outcomes depend on the implemented screening rules and thresholds. The prototype is intended to support quality assessment, not replace formal qualification or engineering review.

<img width="1000" height="500" alt="Screenshot (1067)" src="https://github.com/user-attachments/assets/16425153-c697-4ef7-ac40-4038729e4d1f" />  <img width="1000" height="500" alt="Screenshot (1064)" src="https://github.com/user-attachments/assets/ee4cf653-6778-473d-a314-cb2bf5baa672" />



## 🛠️ Technology Stack

**Frontend**
- HTML5
- CSS
- JavaScript
- Chart.js
- FileReader API

**Backend and Analytics**
- Python, NumPy, & Pandas
- SHAP
- FastAPI
- NumPy
- Pandas
- Scikit-learn

**Deployment**
- Netlify



## 🚀 Getting Started

### Explore the Live Demo

Visit: [https://burnsight.netlify.app/](https://burnsight.netlify.app/)

### Run the Frontend Locally

1. Clone the repository:

   ```bash
   git clone https://github.com/Glitched-Jorsh/Burnsight.git
   ```

2. Open Burnsight.html (or your main frontend file) in any modern web browser. (No local server required for client-side evaluation).
3. Upload a supported CSV or Excel dataset and explore the available screening views.

The repository's current frontend entry point is `Burnsight.html`. If you rename it to `index.html`, update these instructions accordingly.

### Backend Setup

The repository also contains Python backend files and dependencies. Follow the backend setup and model-loading requirements in the source files and `requirements.txt` before running the API. The frontend can be explored separately if its features do not require the backend.



## 📌 Project Status

Burnsight is a Smart India Hackathon 2026 prototype trained and validated on synthetic component telemetry datasets (1.5M records). Integration with real historical ISRO/ATE burn-in feeds and shadow-mode pilot testing are part of the future deployment roadmap.



## 🔬 Research Survey & Academic References

### 📊 Research Survey Matrix

| Aspect | Conventional Screening | Proposed System (Burnsight) |
| :--- | :--- | :--- |
| **Reference Baseline** | Fixed datasheet limits | Lot-relative baseline (Median / MAD) |
| **Time Handling** | Pass / Fail at each read-out | 0h + 24h to forecast of the 168h value |
| **Dimensionality** | One parameter at a time | Multivariate: Joint behavior across parameters |
| **Decision** | PASS / FAIL | PASS / REVIEW / REJECTED with fused risk scoring |
| **Justification** | Implicit in the limit | Human-readable evidence for every flag |

### 📈 Dataset Scale & Methodology
* **Dataset Scale:** Trained and validated on **1.5M synthetic component telemetry records**, with a separate 1.5M synthetic dataset used for testing on unseen 168-hour burn-in trajectories.
* **Methodology:** Features 6-parameter multivariate tracking using random forest ensembles and median absolute deviation (MAD).

### 📑 References
1. **Smart India Hackathon 2026** – Problem Statement 26170 and official idea-submission guidelines. Available: [https://sih.gov.in](https://sih.gov.in)
2. **Sahu & Leidecker** – *EEE-INST-002: EEE Parts Selection, Screening, Qualification and Derating* (NASA GSFC, 2008). Available: [NASA NEPP DocUploads](https://nepp.nasa.gov/DocUploads/FFB52B88-36AE-4378-A05B2C084B5EE2CC/EEE-INST-002_add1.pdf)
3. **NIST/SEMATECH** – *e-Handbook of Statistical Methods – outlier detection, Grubbs' test, process monitoring*. Available: [NIST Handbook](https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h1.htm)
4. **Part Average Testing (AEC-Q001)** – Statistical removal of parts abnormal relative to their population. Available: [Semiconductor Engineering](https://semiengineering.com/part-average-test-pat)
5. **Scikit-learn** – Novelty and outlier detection; ensemble methods and regression. Available: [Scikit-learn Documentation](https://scikit-learn.org/stable/modules/outlier_detection.html)
6. **Lundberg, S. & Lee, S.-I.** – *A Unified Approach to Interpreting Model Predictions (SHAP)*, NeurIPS 2017. Available: [arXiv:1705.07874](https://arxiv.org/abs/1705.07874)



## 👥 Team Burnsight  
Smart India Hackathon 2026  
Problem Statement: PS 26170  
Team ID: 168888



*Developed to explore data-driven anomaly detection and early drift prediction for high-reliability electronic component screening.*

