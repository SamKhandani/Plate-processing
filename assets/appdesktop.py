import cv2  # OpenCV برای پردازش تصویر / Import OpenCV for image processing
import numpy as np  # NumPy برای عملیات روی آرایه‌ها / Import NumPy for array operations
from PIL import Image, ImageTk  # PIL برای باز کردن و نمایش تصاویر / Import PIL for image handling and Tkinter compatibility
from ultralytics import YOLO  # YOLO از ultralytics / Import the YOLO model from ultralytics
from huggingface_hub import hf_hub_download  # دانلود وزن از Hugging Face / Import function to download weights from Hugging Face
import logging  # ماژول logging برای ثبت پیام‌های خطا و اطلاعات / Import logging module for error and info logging
import tkinter as tk  # Tkinter برای ایجاد رابط کاربری دسکتاپ / Import Tkinter for desktop UI
from tkinter import filedialog  # فایل دیالوگ برای انتخاب تصویر / Import filedialog for image selection
from tkinter import ttk  # ttk برای بهبود UI/UX / Import ttk for enhanced UI/UX

# پیکربندی logging برای نمایش پیام‌های خطا و اطلاعات / Configure logging to display error and info messages
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ----------------------------- دانلود وزن‌ها و راه‌اندازی مدل‌ها -----------------------------
# Download weights and initialize models
try:
    weights_path = hf_hub_download(
        repo_id="krishnamishra8848/Nepal-Vehicle-License-Plate-Detection", 
        filename="last.pt"
    )
except Exception as e:
    logging.error("Error downloading license plate detection weights: %s", e)
    raise

try:
    model_detection = YOLO(weights_path)
except Exception as e:
    logging.error("Error initializing license plate detection model: %s", e)
    raise

try:
    model_processing = YOLO('F:/my/code folder/plak prosesing/runs/detect/train5/weights/best.pt')
except Exception as e:
    logging.error("Error loading license plate character recognition model: %s", e)
    raise

# ----------------------------- تابع پردازش تصویر -----------------------------
def process_image_gui(image_path):
    """
    پردازش تصویر جهت تشخیص پلاک و حروف آن و بازگرداندن نتایج / 
    Process image for license plate detection and OCR, returning a list of results.
    
    خروجی: لیستی از دیکشنری‌ها شامل کلیدهای "plate_image" (تصویر پلاک پردازش‌شده)
           و "recognized_text" (متن شناسایی‌شده) / 
           Output: A list of dictionaries with keys "plate_image" (processed plate image)
           and "recognized_text" (recognized text).
    """
    results_list = []
    try:
        image = Image.open(image_path)
        img = np.array(image)
        # تبدیل BGR به RGB / Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results_detection = model_detection(img)
    except Exception as e:
        logging.error("Error processing image: %s", e)
        return results_list

    # پردازش هر نتیجه تشخیص پلاک / Process each detected plate
    for result in results_detection:
        if hasattr(result, 'boxes') and result.boxes is not None:
            for box in result.boxes.xyxy:
                try:
                    x1, y1, x2, y2 = map(int, box)
                    # برش منطقه پلاک / Crop the license plate region
                    plate_crop = img[y1:y2, x1:x2]
                    scale_factor = 2  # فاکتور بزرگنمایی / Scaling factor
                    # بزرگنمایی تصویر / Upscale image
                    plate_upscaled = cv2.resize(
                        plate_crop,
                        (plate_crop.shape[1] * scale_factor, plate_crop.shape[0] * scale_factor),
                        interpolation=cv2.INTER_CUBIC
                    )
                    # حذف نویز از تصویر / Denoise image
                    plate_denoised = cv2.fastNlMeansDenoisingColored(
                        plate_upscaled, None,
                        h=10, hColor=10,
                        templateWindowSize=7, searchWindowSize=21
                    )
                    # اعمال فیلتر شارپینگ برای بهبود وضوح / Apply sharpening filter for clarity
                    kernel_sharpening = np.array([
                        [-1, -1, -1],
                        [-1,  9, -1],
                        [-1, -1, -1]
                    ])
                    plate_sharpened = cv2.filter2D(plate_denoised, -1, kernel_sharpening)
                except Exception as e:
                    logging.error("Error processing license plate region: %s", e)
                    continue

                recognized_text = ""
                try:
                    results_chars = model_processing.predict(source=plate_sharpened, conf=0.5)
                    for r in results_chars:
                        if r.boxes is not None and len(r.boxes) > 0:
                            # مرتب‌سازی جعبه‌ها بر اساس مختصات x (چپ به راست) / Sort boxes by x-coordinate (left-to-right)
                            sorted_boxes = sorted(r.boxes, key=lambda b: b.xyxy[0][0])
                            recognized_labels = []
                            for b in sorted_boxes:
                                class_id = int(b.cls[0])
                                label = r.names[class_id]
                                recognized_labels.append(label)
                            recognized_text = "-".join(recognized_labels)
                except Exception as e:
                    logging.error("Error during character recognition: %s", e)
                # اضافه کردن نتیجه به لیست نتایج / Append result to list
                results_list.append({"plate_image": plate_sharpened, "recognized_text": recognized_text})
    return results_list

# ----------------------------- توابع رابط کاربری -----------------------------
def display_results(results):
    """
    نمایش نتایج در رابط کاربری / Display the results in the UI.
    
    هر پلاک به همراه متن شناسایی‌شده در یک فریم جداگانه نمایش داده می‌شود.
    / Each detected plate along with its recognized text is shown in its own frame.
    """
    # پاکسازی نتایج قبلی (در صورت وجود) / Clear previous results (if any)
    for widget in results_frame.winfo_children():
        widget.destroy()
    
    if not results:
        label_status.config(text="No license plate detected. / پلاک تشخیص داده نشد.")
        return
    else:
        label_status.config(text=f"{len(results)} license plate(s) detected. / {len(results)} پلاک شناسایی شد.")
    
    # نمایش هر نتیجه / Display each result
    for idx, result in enumerate(results):
        try:
            plate_img = result["plate_image"]
            # تبدیل تصویر NumPy به PIL / Convert NumPy image to PIL image
            plate_pil = Image.fromarray(plate_img)
            plate_pil = plate_pil.resize((400, 200))  # تغییر اندازه جهت نمایش مناسب / Resize for proper display
            plate_photo = ImageTk.PhotoImage(plate_pil)
            
            # ایجاد فریم برای هر نتیجه با حاشیه و فاصله مناسب / Create a frame for each result with proper border and padding
            result_frame = ttk.Frame(results_frame, relief="ridge", padding=10)
            result_frame.pack(pady=10, padx=10, fill="x", expand=True)
            
            # نمایش تصویر پلاک / Display the license plate image
            img_label = ttk.Label(result_frame, image=plate_photo)
            img_label.image = plate_photo  # نگه داشتن مرجع تصویر / Keep a reference to the image
            img_label.pack(side="left", padx=10)
            
            # نمایش متن شناسایی‌شده / Display the recognized text
            text = result["recognized_text"]
            text_label = ttk.Label(result_frame, text="Recognized plate text: " + text, font=("Arial", 14))
            text_label.pack(side="left", padx=10, anchor="center")
        except Exception as e:
            logging.error("Error displaying result %d: %s", idx, e)

def select_image():
    """
    انتخاب تصویر از کاربر و پردازش آن / Allow user to select an image and process it.
    """
    try:
        file_path = filedialog.askopenfilename(title="Select an Image / انتخاب تصویر")
        if file_path:
            results = process_image_gui(file_path)
            display_results(results)
    except Exception as e:
        logging.error("Error in select_image: %s", e)

# ----------------------------- راه‌اندازی رابط کاربری -----------------------------
def main():
    """
    تابع اصلی جهت راه‌اندازی رابط کاربری دسکتاپ / Main function to initialize the desktop UI.
    """
    root = tk.Tk()
    root.title("License Plate Recognition - Desktop / تشخیص پلاک خودرو")
    root.geometry("900x700")
    
    # استفاده از ttk و انتخاب تم مناسب برای ظاهر بهتر / Use ttk and select a suitable theme for a better appearance
    style = ttk.Style(root)
    style.theme_use('clam')
    
    # فریم بالایی برای کنترل‌ها / Top frame for controls
    top_frame = ttk.Frame(root, padding=10)
    top_frame.pack(side="top", fill="x")
    
    # دکمه انتخاب تصویر / Button to select image
    button_select = ttk.Button(top_frame, text="Select Image / انتخاب تصویر", command=select_image)
    button_select.pack(side="left", padx=10)
    
    # برچسب نمایش وضعیت / Status label
    global label_status
    label_status = ttk.Label(top_frame, text="", font=("Arial", 12))
    label_status.pack(side="left", padx=10)
    
    # فریم نمایش نتایج (تصاویر و متون پلاک) / Frame to display results (license plate images and texts)
    global results_frame
    results_frame = ttk.Frame(root, padding=10)
    results_frame.pack(side="top", fill="both", expand=True)
    
    root.mainloop()

if __name__ == "__main__":
    main()