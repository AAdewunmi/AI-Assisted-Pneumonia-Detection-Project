"""
app/utils.py
-------------
Utility functions for PneumoDetect Flask app.
"""


def apply_threshold(probability: float, threshold: float) -> str:
    """
    Apply decision threshold for pneumonia classification.

    Args:
        probability (float): Predicted probability of pneumonia.
        threshold (float): Decision boundary between classes.

    Returns:
        str: "High Risk" if prob > threshold else "Low Risk".
    """
    return "High Risk" if probability > threshold else "Low Risk"
