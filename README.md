
# X-Ray Analyzer Application

**A Python-based medical imaging tool for bone segmentation, fracture detection, and X-ray enhancement**

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green?logo=opencv)
![Colab](https://img.shields.io/badge/Google_Colab-Compatible-orange?logo=google-colab)

---

## Features
- **Bone Segmentation**: Color-coded watershed segmentation
- **Fracture Detection**: Identifies fracture types (Transverse/Oblique/Comminuted)
- **Image Enhancement**: CLAHE contrast + noise reduction
- **Interactive UI**: Colab-native widget controls

---

## How to Run
1. Open in [Google Colab](#)
2. Upload X-ray (JPG/PNG)
3. Use processing buttons:
   - Enhance Image
   - Segment Bones  
   - Detect Cracks

```python
xray_analyzer = XRayAnalyzer()  # Runs the interface

## Technical Stack
| Component       | Technology |
|----------------|------------|
| Image Processing | OpenCV, scikit-image |
| UI Framework | ipywidgets |
| Visualization | Matplotlib |





## Limitations
- Accuracy: ~85-90% on clear X-rays
- File Support: JPG/PNG only
- Performance: Slower on >5MB images



