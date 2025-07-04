import cv2
import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from PIL import Image, ImageTk
import ipywidgets as widgets
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
from google.colab import files
import io

class XRayAnalyzer:
    def __init__(self):
        self.original_image = None
        self.processed_image = None
        self.colors = [(255,0,0), (255,255,0), (0,255,0), (0,255,255), 
                      (255,0,255), (128,0,0), (0,128,0), (0,0,128), 
                      (128,128,0), (128,0,128)]
        
        # Create UI elements
        self.create_ui()
    
    def create_ui(self):
        # Upload button
        self.upload_btn = widgets.FileUpload(
            accept='.png,.jpg,.jpeg',
            multiple=False,
            description='Upload X-Ray'
        )
        
        # Control buttons
        self.enhance_btn = widgets.Button(description="Enhance Image", disabled=True)
        self.noise_btn = widgets.Button(description="Reduce Noise", disabled=True)
        self.segment_btn = widgets.Button(description="Segment Bones", disabled=True)
        self.outline_btn = widgets.Button(description="Show Outline", disabled=True)
        self.crack_btn = widgets.Button(description="Detect Cracks", disabled=True)
        self.reset_btn = widgets.Button(description="Reset", disabled=True)
        
        # Output area
        self.output = widgets.Output()
        
        # Set button callbacks
        self.upload_btn.observe(self.load_image, names='value')
        self.enhance_btn.on_click(lambda b: self.enhance_image())
        self.noise_btn.on_click(lambda b: self.reduce_noise())
        self.segment_btn.on_click(lambda b: self.segment_bones())
        self.outline_btn.on_click(lambda b: self.show_outline())
        self.crack_btn.on_click(lambda b: self.detect_cracks())
        self.reset_btn.on_click(lambda b: self.reset())
        
        # Display UI
        display(widgets.VBox([
            widgets.HBox([self.upload_btn]),
            widgets.HBox([self.enhance_btn, self.noise_btn, self.segment_btn]),
            widgets.HBox([self.outline_btn, self.crack_btn, self.reset_btn]),
            self.output
        ]))
    
    def load_image(self, change):
        if not change['new']:
            return
            
        with self.output:
            clear_output()
            print("Processing uploaded image...")
            
        # Get uploaded file
        uploaded_file = next(iter(self.upload_btn.value.values()))
        img_bytes = uploaded_file['content']
        
        # Convert to OpenCV format
        img = Image.open(io.BytesIO(img_bytes))
        self.original_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        self.processed_image = self.original_image.copy()
        
        # Enable buttons
        self.enhance_btn.disabled = False
        self.noise_btn.disabled = False
        self.segment_btn.disabled = False
        self.outline_btn.disabled = False
        self.crack_btn.disabled = False
        self.reset_btn.disabled = False
        
        self.update_display()
    
    def enhance_image(self):
        if self.processed_image is not None:
            gray = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            self.processed_image = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            self.update_display()
    
    def reduce_noise(self):
        if self.processed_image is not None:
            gray = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2GRAY)
            denoised = cv2.fastNlMeansDenoising(gray, None, h=15, templateWindowSize=7, searchWindowSize=21)
            filtered = cv2.bilateralFilter(denoised, d=9, sigmaColor=75, sigmaSpace=75)
            self.processed_image = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)
            self.update_display()
    
    def segment_bones(self):
        if self.processed_image is not None:
            gray = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            dist_transform = ndimage.distance_transform_edt(binary)
            coordinates = peak_local_max(dist_transform, min_distance=20, exclude_border=False)
            mask = np.zeros(dist_transform.shape, dtype=bool)
            mask[tuple(coordinates.T)] = True
            markers = ndimage.label(mask)[0]
            labels = watershed(-dist_transform, markers, mask=binary)
            
            segmented = np.zeros_like(self.processed_image)
            for label in range(1, np.max(labels) + 1):
                mask = labels == label
                color = self.colors[label % len(self.colors)]
                for c in range(3):
                    segmented[:,:,c] = segmented[:,:,c] + (mask * color[c]).astype(np.uint8)
            
            self.processed_image = cv2.addWeighted(self.processed_image, 0.7, segmented, 0.3, 0)
            self.update_display()
    
    def show_outline(self):
        if self.processed_image is not None:
            gray = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            self.processed_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            self.update_display()
    
    def detect_cracks(self):
        if self.processed_image is not None:
            gray = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2GRAY)
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 11, 2)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            result = self.processed_image.copy()
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 50:
                    fracture_type = self.classify_fracture(contour)
                    cv2.drawContours(result, [contour], -1, (0, 0, 255), 2)
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        cv2.putText(result, fracture_type, (cx-20, cy),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            self.processed_image = result
            self.update_display()
    
    def classify_fracture(self, contour):
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return "Unknown"
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        rect = cv2.minAreaRect(contour)
        width, height = min(rect[1]), max(rect[1])
        if width == 0:
            return "Unknown"
        aspect_ratio = height / width
        angle = abs(rect[2])
        if area > 1000 and aspect_ratio > 3:
            return "Major Fracture"
        elif aspect_ratio > 2 and angle < 30:
            return "Transverse Fracture"
        elif aspect_ratio > 2 and 30 <= angle <= 60:
            return "Oblique Fracture"
        elif circularity < 0.2:
            return "Comminuted Fracture"
        else:
            return "Simple Fracture"
    
    def update_display(self):
        with self.output:
            clear_output()
            if self.processed_image is not None:
                # Convert BGR to RGB for matplotlib
                img_rgb = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2RGB)
                plt.figure(figsize=(10, 8))
                plt.imshow(img_rgb)
                plt.axis('off')
                plt.show()
    
    def reset(self):
        if self.original_image is not None:
            self.processed_image = self.original_image.copy()
            self.update_display()

# Run the analyzer in Colab
xray_analyzer = XRayAnalyzer()
