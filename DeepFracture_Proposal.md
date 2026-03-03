# DeepFracture: A Research Proposal

## 1. TITLE
**DeepFracture: An Explainable and Risk-Aware Deep Learning Framework for Bone Fracture Detection and Structural Assessment from Radiographs**

## 2. PROBLEM STATEMENT
Bone fractures are among the most common orthopedic injuries globally. Rapid and accurate diagnosis is critical for effective treatment and avoiding long-term complications. However, clinical fracture detection from standard plain radiographs faces several persistent challenges:
- **Inter-Observer Variability:** Diagnoses can vary significantly between clinicians depending on experience, fatigue, and differing interpretations of ambiguous imaging features.
- **Missed Occult/Hairline Fractures:** Subtle fractures are notoriously difficult to detect, often overshadowed by complex surrounding bone structures or overlapping soft tissues.
- **The "Black-Box" Problem in AI:** While Convolutional Neural Networks (CNNs) have shown high accuracy in fracture classification, their clinical adoption remains hampered by a lack of interpretability. Physicians are understandably hesitant to rely on algorithmic outputs without understanding the underlying reasoning, especially when ruling out spurious correlations.
- **Lack of Multimodal Integration:** Current AI models typically rely entirely on image tensors (binary "Fracture vs. Normal" outputs), entirely ignoring the patient's baseline clinical information (Age, BMI, Sex, secondary conditions). As highlighted in recent literature (Kong et al., *Endocrinol Metab* 2022), integrating clinical baseline features alongside image data significantly outperforms standard models in predicting long-term osteoporotic fracture risk.

This project addresses these gaps by developing a robust, multimodal framework that not only detects current clinical fractures but also provides visual explanations and a longitudinal structural risk assessment by effectively merging extractable image biomarkers with synthetic clinical patient profiles.

## 3. OBJECTIVES
The primary objective is to engineer a comprehensive, clinically relevant AI framework for fracture analysis. This is broken down into specific, measurable goals:
1.  **High Sensitivity Detection:** Develop a robust classification model capable of distinguishing fractured from normal radiographs with a high degree of sensitivity (minimizing false negatives, which represent missed fractures).
2.  **Visual Explainability:** Implement Gradient-weighted Class Activation Mapping (Grad-CAM) to establish clinical trust by visualizing the specific spatial regions driving the model's predictions.
3.  **Structural Risk Stratification:** Engineer a secondary predictive module that utilizes foundational texture, edge, and intensity features derived directly from the radiograph to categorize bone structural integrity into discrete risk levels.
4.  **Architectural Comparison:** Conduct a rigorous comparative analysis between a baseline CNN trained from scratch and a transfer-learning model utilizing a pre-trained ResNet50 architecture.

## 4. SCOPE
The scope of this mini-project is strictly defined to ensure academic rigor and feasibility within typical university constraints (e.g., limited GPU resources and timeframe).

*   **Dataset:** We will utilize the MURA (musculoskeletal radiographs) dataset structure, formulating the task as a binary classification problem (Fractured vs. Normal).
*   **Data Pipeline:** The pipeline will include critical preprocessing steps tailored for medical imaging:
    *   **Contrast Enhancement:** Implementation of Contrast Limited Adaptive Histogram Equalization (CLAHE) to reveal subtle bone details.
    *   **Imbalance Handling:** Calculation and application of inverse class frequency weights within the loss function to mitigate dataset biases.
    *   **Advanced Augmentation:** Utilizing spatial (rotation, flipping) and intensity (brightness, contrast) augmentations to improve model generalization.
*   **Modeling Strategy:**
    *   **Model A (Baseline):** A simple, custom-built Convolutional Neural Network trained entirely from scratch.
    *   **Model B (Advanced Classification):** A ResNet50 architecture utilizing transfer learning. Early layers will be frozen to retain generalized edge/texture filters, while the final fully connected layers will be fine-tuned.
    *   **Model C (Multimodal Risk Predictor):** A Random Forest ensemble algorithm algorithmically inspired by DeepSurv methodologies, taking concatenated inputs of CNN confidence, GLCM image texture, and simulated clinical baseline attributes (Age, BMI, Sex).
*   **Evaluation Metrics:** Classification algorithms will be rigorously evaluated using Accuracy, Precision, Recall, F1-Score, Sensitivity (True Positive Rate), Specificity (True Negative Rate), and Area Under the ROC Curve (AUC). For the Risk Stratification module, the **Concordance Index (C-index)** will be utilized to measure predictive accuracy, mimicking medical survival analysis standards.
*   **Explainability:** Grad-CAM will be implemented to generate heatmaps overlaid on the original radiographs, highlighting regions of interest.
*   **Error Analysis:** An automated pipeline will systematically log and save misclassified samples to facilitate visual inspection and discussion regarding model failure modes (e.g., artifacts, casts, ambiguous joints).

## 5. RISK STRATIFICATION REFINEMENT (Inspired by Kong et al. 2022)
Moving beyond unrealistic assertions, this module adopts a multimodal approach to compute an immediate "Structural Risk Factor," mimicking the architecture of proven longitudinal prediction algorithms (like DeepSurv). 

*   **Image Feature Extraction:** We will compute key image biomarkers directly from the radiograph:
    *   **Texture:** Gray-Level Co-occurrence Matrix (GLCM) properties (Contrast and Homogeneity) to assess localized bone density variations (often a proxy for osteoporosis).
    *   **Edge Density:** Utilizing Canny edge detection to quantify structural complexity.
*   **Clinical Feature Integration:** The pipeline will accept synthetic baseline clinical parameters:
    *   Age, Sex (binary), and Body Mass Index (BMI).
*   **Modeling:** The concatenated continuous and categorical variables (CNN Confidence Score + Image Textures + Clinical Baselines) will be fed into a **Random Forest Classifier**. An ensemble method is preferred here to map non-linear interactions between a patient's age/BMI and their bone texture density.
*   **Output:** The model categorizes the patient into discrete structural risk profiles (*Low, Medium, High Risk*), evaluated internally via the **C-index** to ensure robust ranking of potential future fracture failure.

## 6. BIG DATA JUSTIFICATION
While the dataset (e.g., the full MURA dataset) may contain tens of thousands of images, the processing pipeline is designed to be highly memory-efficient and suitable for local or limited-resource environments:

*   **Lazy Loading:** Data is not loaded entirely into RAM. We utilize PyTorch's `DataLoader` and `ImageFolder` to dynamically fetch, augment, and tensorize images only as they are needed for the current batch.
*   **Batch Training:** Images are processed in manageable batches (e.g., 32 at a time), keeping GPU VRAM requirements low.
*   **Efficient Preprocessing:** Augmentations and normalization steps are applied "on the fly" within the CPU-bound dataloader processes (`num_workers`), ensuring the GPU is heavily utilized for continuous tensor matrix multiplication rather than waiting for I/O operations.

## 7. SYSTEM ARCHITECTURE
The system is designed with a clear, modular pipeline, facilitating maintainability and scalability:

1.  **Data Ingestion & Preprocessing Module:** Handles loading raw data, applying CLAHE, resizing, normalizing, and yielding augmented batches.
2.  **Detection Module:** Contains the PyTorch definitions for both the `BaselineCNN` and the `DeepFractureModel` (ResNet50), outputting raw logits and softmax probabilities.
3.  **Explainability Module (Grad-CAM):** Attaches hooks to the final convolutional layers of the chosen model to capture gradients and feature maps during the backward pass, generating the localizing heatmap.
4.  **Risk Module:** An independent script handling feature extraction (GLCM, Canny) and executing the scikit-learn Random Forest classifier.
5.  **Deployment UI:** A Streamlit-based web application orchestrating the entire framework, allowing users to upload novel images and interactively view the diagnosis, confidence, Extracted Biomarkers, and XAI outputs.

## 8. EVALUATION STRATEGY
Model performance will be evaluated holistically, prioritizing clinical utility over pure accuracy:

*   **Sensitivity vs. Specificity Tradeoff:** In medical screening, missing a positive case (False Negative) is generally more detrimental than a False Positive. We will prioritize Sensitivity and analyze the ROC curve to select an optimal decision threshold.
*   **Visual Error Analysis:** The automated logging of misclassifications will form the basis of a critical discussion section within the final report. We will visually inspect these failures to identify systemic weaknesses, such as difficulties handling patients with preexisting orthopedic hardware or severe osteoarthritis.
*   **Confusion Matrix:** A normalized confusion matrix will clearly illustrate the proportion of Type I vs. Type II errors.

## 9. NOVELTY POSITIONING
Standard CNN projects end at the "Fracture/Normal" output. **DeepFracture** differentiates itself via a tripartite approach:

1.  **Combating the Black Box:** The integration of Grad-CAM directly addresses the clinical necessity for algorithmic transparency, proving that the model is making decisions based on cortical disruptions rather than learned dataset artifacts (like institutional text markers).
2.  **Multimodal Clinical Context:** Inspired directly by published works proving that image-only CNNs are insufficient for presymptomatic tracking (Kong et al., *EnM* 2022), the integration of an independent Risk Stratification module that merges classical computer vision (GLCM texture, Edge density) with Baseline Clinical statistics (Age, BMI). This algorithmically captures that a radiograph contains valuable density data *beyond* simply confirming a current break.
3.  **Rigorous Evaluation:** Focusing heavily on clinical metrics (Sensitivity/Specificity) and implementing automated Error Analysis pipelines to dissect *where* and *why* the AI fails, a crucial step often omitted in basic ML projects.

## 10. DEPLOYMENT STRATEGY
The final framework will be deployed as an interactive Streamlit application. This serves several purposes:

*   **Interactive Demo:** Provides a clean, accessible interface for university evaluators to easily test the model without requiring command-line interaction.
*   **Rural Healthcare Simulation:** The web-app structure simulates how such a tool could be deployed as an initial screening aid in low-resource or rural healthcare settings lacking immediate access to specialized radiologists.
*   **Future Scope:** While currently a standalone application, the modular design conceptually demonstrates how the inference pipeline could be packaged as an API for eventual integration into standard clinical Picture Archiving and Communication Systems (PACS).
