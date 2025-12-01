from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
import os

def create_presentation():
    prs = Presentation()

    # Helper to add a slide with title and content
    def add_slide(title_text, content_text=None):
        slide_layout = prs.slide_layouts[1] # Title and Content
        slide = prs.slides.add_slide(slide_layout)
        title = slide.shapes.title
        title.text = title_text
        if content_text:
            content = slide.placeholders[1]
            content.text = content_text
        return slide

    # Helper to add a slide with title and image
    def add_image_slide(title_text, image_path):
        slide_layout = prs.slide_layouts[5] # Title Only
        slide = prs.slides.add_slide(slide_layout)
        title = slide.shapes.title
        title.text = title_text
        
        if os.path.exists(image_path):
            # Center the image
            left = Inches(1)
            top = Inches(1.5)
            height = Inches(5.5)
            slide.shapes.add_picture(image_path, left, top, height=height)
        else:
            print(f"Warning: {image_path} not found.")

    # 1. Title Slide
    slide_layout = prs.slide_layouts[0] # Title Slide
    slide = prs.slides.add_slide(slide_layout)
    title = slide.shapes.title
    subtitle = slide.placeholders[1]
    title.text = "Breast Cancer Classification"
    subtitle.text = "Analysis and Prediction using Machine Learning"

    # 2. Introduction
    intro_text = (
        "Objective: To predict whether a breast mass is benign or malignant based on cell nuclei measurements.\n\n"
        "Importance: Early diagnosis significantly improves survival rates.\n\n"
        "Approach: Machine Learning Classification (Random Forest)."
    )
    add_slide("Introduction", intro_text)

    # 3. Dataset Description
    dataset_text = (
        "Source: Breast Cancer Wisconsin (Diagnostic) Data Set.\n\n"
        "Features: 30 numeric features (radius, texture, perimeter, area, smoothness, etc.) computed from digitized images of fine needle aspirate (FNA).\n\n"
        "Target: Diagnosis (M = Malignant, B = Benign)."
    )
    add_slide("Dataset Overview", dataset_text)

    # 4. EDA: Class Distribution
    add_image_slide("Class Distribution", "class_distribution.png")

    # 5. Correlation Heatmap
    add_image_slide("Feature Correlation", "correlation_heatmap.png")

    # 6. Model Methodology
    model_text = (
        "Model: Random Forest Classifier.\n"
        "Split: 80% Training, 20% Testing.\n"
        "Stratification: Used to maintain class balance in splits.\n"
        "Metric: Accuracy and Confusion Matrix."
    )
    add_slide("Methodology", model_text)

    # 7. Results
    # Read metrics
    metrics_text = "Results:\n"
    try:
        with open("metrics.txt", "r") as f:
            metrics_text += f.read()
    except FileNotFoundError:
        metrics_text += "Metrics file not found."

    add_slide("Model Performance", metrics_text)

    # 8. Confusion Matrix
    add_image_slide("Confusion Matrix", "confusion_matrix.png")

    # 9. Conclusion
    conclusion_text = (
        "Summary: The Random Forest model achieved high accuracy in classifying breast cancer tumors.\n\n"
        "Future Work:\n"
        "- Hyperparameter tuning.\n"
        "- Testing other algorithms (SVM, XGBoost).\n"
        "- Feature selection to reduce dimensionality."
    )
    add_slide("Conclusion", conclusion_text)

    # Save
    output_file = "Breast_Cancer_Presentation.pptx"
    prs.save(output_file)
    print(f"Presentation saved to {output_file}")

if __name__ == "__main__":
    create_presentation()
