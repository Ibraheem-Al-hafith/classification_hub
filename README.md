# 🏗️ Classification Hub — SAIR PyTorch Mastery

> You covered all the theory. You followed the lectures.
> Now here are five real datasets and five real problems.
> Build the solutions yourself.

---

## Exercises

| # | Modality | Project | Dataset | Difficulty | Time Estimate |
|---|----------|---------|---------|------------|---------------|
| [Ex 1](Ex_1_Tabular_Classification.ipynb) | 📊 Tabular | Rice Type Classifier | [Kaggle](https://www.kaggle.com/datasets/mssmartypants/rice-type-classification) | ⭐ Easy | 3–4 hrs |
| [Ex 2](Ex_2_Image_Classification.ipynb) | 🖼️ Image (scratch) | Animal Face Classifier | [Kaggle](https://www.kaggle.com/datasets/andrewmvd/animal-faces) | ⭐⭐ Medium | 5–7 hrs |
| [Ex 3](Ex_3_Image_Classification_Pretrained.ipynb) | 🌿 Image (pretrained) | Bean Leaf Disease Detector | [Kaggle](https://www.kaggle.com/datasets/marquis03/bean-leaf-lesions-classification) | ⭐⭐ Medium | 5–6 hrs |
| [Ex 4](Ex_4_Audio_Classification.ipynb) | 🎵 Audio | Quran Reciter Identifier | [Kaggle](https://www.kaggle.com/datasets/mohammedalrajeh/quran-recitations-for-audio-classification) | ⭐⭐⭐ Hard | 7–10 hrs |
| [Ex 5](Ex_5_Text_Classification_Transformers.ipynb) | 📝 Text | Sarcasm Detector | [Kaggle](https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection) | ⭐⭐ Medium | 5–7 hrs |

> **Where to start:** Ex 1 → get familiar with the submission format. Then pick whichever modality interests you most.

---

## What Every Submission Must Include

Every notebook you submit must have all of the following:

- A **trained PyTorch model** that solves the stated problem
- A **training report** — loss and accuracy curves for train and validation
- A **final test accuracy score** on held-out data
- A **live inference demo** — the model runs on a new real example and returns a prediction

For **Ex 3 and Ex 5** you also need a written explanation (in a markdown cell)
of your architectural decisions: what model you chose, what you froze, what you added, and why.

For **Ex 4** you need a written explanation of how you converted audio into
a format your model could process.

---

## Target Accuracy

These are realistic targets. Exceeding them means your implementation is strong.

| Exercise | Target Accuracy | Notes |
|----------|----------------|-------|
| Ex 1 — Rice Classifier | ≥ 92% | Clean tabular data, should converge fast |
| Ex 2 — Animal Faces | ≥ 75% | Building a CNN from scratch is genuinely hard |
| Ex 3 — Bean Disease | ≥ 88% | Transfer learning makes this very achievable |
| Ex 4 — Quran Reciter | ≥ 70% | Audio is harder; 70% is a strong result |
| Ex 5 — Sarcasm Detector | ≥ 82% | Fine-tuned transformer should clear this easily |

---

## Submission Rubric

Each submission is assessed across four dimensions:

| Criterion | Full Credit | Partial Credit | Minimal Credit |
|-----------|-------------|----------------|----------------|
| **Model Quality** | Meets or beats target accuracy | Within 5% of target | Below target but trained correctly |
| **Training Report** | Loss & accuracy curves, train vs. val | Curves present but incomplete | Only final score reported |
| **Code Quality** | Clean, readable, functions/classes used | Working but unstructured | Barely working, messy |
| **Inference Demo** | Runs on a real new example with output | Demo exists but hardcoded | No live demo |

For exercises requiring written explanations (Ex 3, Ex 4, Ex 5): the explanation must be specific — "I used ResNet-18 because it was pre-trained on ImageNet which has overlapping features with leaf disease images" is good; "I chose this model because it performs well" is not.

---

## Relevant Lectures

| Exercise | Review Before Starting |
|----------|----------------------|
| Ex 1 — Tabular | Module 2 pipelines; `1_Intro.ipynb`; `2_DataLoader.ipynb` |
| Ex 2 — Image (scratch) | `3_CNN.ipynb`; `labs/lab_3.ipynb` |
| Ex 3 — Image (pretrained) | `4_Transfer_and_ResNet.ipynb`; `labs/lab_4.ipynb` |
| Ex 4 — Audio | `3_CNN.ipynb` (2D input logic); independent research on mel spectrograms |
| Ex 5 — Text | `8A_HuggingFace_Ecosystem.ipynb`; `8B_Hugging_Face_Finetuning.ipynb` |

---

## Note on Exercise 4

Audio classification was not demonstrated step-by-step in the lectures.
You are expected to research the pipeline independently.
You have the mental model — a CNN that classifies 2D input.
The missing piece is how to get from a `.wav` file to a 2D array.

**Hint:** look into mel spectrograms (`librosa.feature.melspectrogram`). Once you have a 2D array, the rest is a standard image classification pipeline.

---

## Setup (Google Colab)

All exercises run on **Google Colab with a T4 GPU**.

**Runtime → Change runtime type → T4 GPU**

For Kaggle dataset downloads:
```python
from google.colab import files
files.upload()  # upload your kaggle.json
!mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
```

---

## Getting Unstuck

If you're stuck, work through this checklist before asking for help:

1. **Data** — Did you check class distribution? Is there severe imbalance?
2. **Shapes** — Print `X.shape`, `y.shape`, and the first batch from your DataLoader
3. **Loss** — Is the initial loss close to `−log(1/n_classes)`? If not, check your output layer
4. **Overfitting** — Is train accuracy high but val accuracy low? Add dropout or data augmentation
5. **Underfitting** — Is both train and val accuracy low? Increase model capacity or train longer
6. **Learning rate** — If loss oscillates wildly, divide `lr` by 10. If it doesn't move, multiply by 10

Still stuck? Post in the [SAIR Telegram community](https://t.me/+jPPlO6ZFDbtlYzU0) with your loss curves and a code snippet.

---

*SAIR PyTorch Mastery Course — Classification Hub*
