# AI-Powered Journaling App

**A research prototype exploring NLP-based mental health self-monitoring for students**

[![Live App](https://img.shields.io/badge/Streamlit-Live%20App-FF4B4B?logo=streamlit)](https://ai-powered-journaling-app-ltdcnt5ybjwx32vnlqowrp.streamlit.app/)
[![Model](https://img.shields.io/badge/Hugging%20Face-Model-yellow?logo=huggingface)](https://huggingface.co/paras9o9/journal-distilbert-model)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## Motivation

Journaling has long been associated with reflective self-regulation, and there is growing interest in how structured writing practices may support emotional processing, not unlike the principles behind neuroplasticity-informed habit change. This project does not claim to replicate that mechanism; rather, it is a modest, academically-motivated attempt to explore whether natural language processing can help students notice patterns in their own journal entries, self-monitor mood-related language, and optionally screen for anxiety and depression indicators using validated instruments.

The goal is educational and exploratory: to build, evaluate, and document an end-to-end NLP pipeline for a socially meaningful problem, while being explicit about its limitations as a non-clinical, prototype-stage tool.

## Project Overview

The AI-Powered Journaling App allows a user to write a free-text journal entry, which is then classified into one of four categories:

| Label | Description |
|---|---|
| `NEU` | Neutral, everyday journaling content |
| `HUMOR` | Lighthearted or humorous entries |
| `MH` | General mental-health-related distress |
| `SI` | Language indicative of suicidal ideation |

Depending on the classification and a rule-based decision layer, the app optionally prompts the user with PHQ-9 (depression) and GAD-7 (anxiety) self-assessment forms, following the standard PHQ-2 → PHQ-9 and GAD-2 → GAD-7 screening flow used in clinical literature. These forms are entirely optional and are offered as a self-evaluation aid, not a diagnostic tool.

- **Live app:** [Streamlit Community Cloud](https://ai-powered-journaling-app-ltdcnt5ybjwx32vnlqowrp.streamlit.app/)
- **Model:** [paras9o9/journal-distilbert-model](https://huggingface.co/paras9o9/journal-distilbert-model) on Hugging Face Hub
- **Author:** Paras Sharma, MCA Final Year, Amity University

## System Architecture

```
User journal entry
       |
       v
Text preprocessing
       |
       v
Classification model (DistilBERT / TF-IDF + LinearSVC)
       |
       v
Decision logic layer (rule-based)
       |
   +---+----------------------+
   |                          |
No further action     Optional PHQ-9 / GAD-7 prompt
   |                          |
   v                          v
Result shown to user   Severity score + resources shown
```

The frontend and inference layer are deployed on Streamlit Community Cloud, while the fine-tuned DistilBERT weights are hosted on the Hugging Face Hub and loaded at runtime, keeping the GitHub repository lightweight.

## Repository Structure

```
AI-Powered-Journaling-App/
├── app.py                     # Streamlit application entry point
├── journal_model.py           # Model loading and inference (loads from Hugging Face Hub)
├── decision_logic.py          # PHQ-9 / GAD-7 scoring and safety decision flow
├── requirements.txt           # Python dependencies
├── README.md
├── LICENSE
├── .gitignore
├── assets/
│   ├── screenshots/            # App UI screenshots
│   └── demo/                   # Demo GIF/video
├── notebooks/
│   ├── 01_eda.ipynb                        # Exploratory data analysis
│   ├── 02_data_preparation.ipynb           # Cleaning, labeling, class balancing
│   ├── 03_tfidf_linearsvc_baseline.ipynb   # Classical ML baseline
│   ├── 04_distilbert_finetuning.ipynb      # Transformer fine-tuning
│   └── 05_evaluation_and_error_analysis.ipynb
└── docs/
    ├── model_notes.md          # Model architecture and training notes
    └── dataset_notes.md        # Dataset sourcing, labeling, and ethics notes
```

## Modeling Approach

Two modeling approaches were developed and compared for the four-class journal classification task, using a dataset of 6,180 labeled entries (MH: 2,860, HUMOR: 2,164, SI: 1,315, NEU: 841).

### Baseline: TF-IDF + LinearSVC (class-balanced)

A classical machine learning baseline was trained using TF-IDF vectorization with a LinearSVC classifier and balanced class weights to address label imbalance, particularly the smaller NEU and SI classes.

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| NEU | 0.67 | 0.25 | 0.37 | 126 |
| HUMOR | 0.66 | 0.76 | 0.71 | 325 |
| MH | 0.61 | 0.71 | 0.66 | 429 |
| SI | 0.57 | 0.45 | 0.50 | 197 |
| **Accuracy** | | | **0.62** | 1077 |
| Macro avg | 0.63 | 0.54 | 0.56 | 1077 |
| Weighted avg | 0.62 | 0.62 | 0.61 | 1077 |

**SI-specific metrics:** Recall = 0.447, Precision = 0.568 (TP: 88, FN: 109, FP: 67)

The baseline struggles particularly with NEU recall (0.25) and SI recall (0.447), indicating that lexical overlap between distress-adjacent classes is difficult for a bag-of-words style model to resolve.

### Final Model: Fine-tuned DistilBERT

A pre-trained DistilBERT model was fine-tuned on the same four-class dataset to capture contextual semantics beyond surface-level word frequency.

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| NEU | 0.45 | 0.55 | 0.50 | 126 |
| HUMOR | 0.80 | 0.78 | 0.79 | 325 |
| MH | 0.73 | 0.60 | 0.66 | 429 |
| SI | 0.52 | 0.67 | 0.59 | 197 |
| **Accuracy** | | | **0.66** | 1077 |
| Macro avg | 0.62 | 0.65 | 0.63 | 1077 |
| Weighted avg | 0.68 | 0.66 | 0.66 | 1077 |

DistilBERT improves overall accuracy (0.66 vs 0.62), macro F1 (0.63 vs 0.56), and notably SI recall (0.67 vs 0.447), which is the most safety-critical metric in this application since false negatives on suicidal ideation carry the highest real-world cost. This improvement, alongside contextual understanding, motivated selecting DistilBERT as the deployed model, with the TF-IDF + LinearSVC pipeline retained in the notebooks as an interpretable baseline for comparison.

### Model Comparison Summary

| Metric | TF-IDF + LinearSVC | DistilBERT |
|---|---|---|
| Accuracy | 0.62 | 0.66 |
| Macro F1 | 0.56 | 0.63 |
| SI Recall | 0.447 | 0.67 |
| SI Precision | 0.568 | 0.52 |
| Deployed | No (baseline) | Yes |

## Dataset and Ethics Notes

The dataset used for training combines and relabels text from publicly available sources associated with the four target categories, restructured into a consistent four-class labeling scheme (SI, MH, HUMOR, NEU) for this academic project. Raw dataset files are intentionally **not included** in this repository; only the data preparation and labeling pipeline is shared through the notebooks, to avoid redistributing sensitive or third-party text at scale.

This project is an **academic research prototype**, not a clinical or diagnostic tool. It has not been validated against clinical outcomes, is not a substitute for professional mental health support, and should not be used to make real-world decisions about a person's safety. The PHQ-9 and GAD-7 instruments used are standard, validated self-report screening tools, but their inclusion here is for educational demonstration of a screening workflow, not for clinical deployment. Any indication of suicidal ideation in a real setting should be directed to a licensed mental health professional or a crisis helpline.

## Tech Stack

- **Language:** Python
- **Modeling:** Hugging Face Transformers (DistilBERT), scikit-learn (TF-IDF, LinearSVC)
- **App framework:** Streamlit
- **Model hosting:** Hugging Face Hub
- **Deployment:** Streamlit Community Cloud
- **Development environment:** Google Colab (GPU/TPU)

## Installation and Local Setup

```bash
git clone https://github.com/paras9o9/AI-Powered-Journaling-App.git
cd AI-Powered-Journaling-App
pip install -r requirements.txt
streamlit run app.py
```

The app automatically downloads the fine-tuned DistilBERT model from the Hugging Face Hub (`paras9o9/journal-distilbert-model`) at runtime, so no manual model download is required.

## Features

- Four-class journal entry classification (NEU, HUMOR, MH, SI)
- Optional PHQ-9 depression and GAD-7 anxiety self-assessment flow
- Rule-based decision logic to determine when to surface screening forms
- Model comparison between a classical ML baseline and a fine-tuned transformer
- Full research workflow documented in Jupyter notebooks (EDA, preparation, training, evaluation)

## Future Improvements

- Expand and further validate the dataset with clearer provenance and larger NEU/SI sample sizes
- Explore focal loss or additional class-balancing strategies to improve NEU and SI performance further
- Add persistent, privacy-respecting storage for trend visualization across multiple entries
- Conduct a small-scale user study to gather qualitative feedback under institutional guidance
- Improve model explainability (e.g., LIME/SHAP) to surface influential phrases per prediction

## Acknowledgements

This project was developed as part of an academic exploration into applying NLP for student mental health self-monitoring, under the MCA program at Amity University. PHQ-9 and GAD-7 are established, validated screening instruments used here strictly for educational demonstration purposes.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
