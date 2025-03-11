import cv2  # وارد کردن OpenCV برای پردازش تصویر / Import OpenCV for image processing
import matplotlib.pyplot as plt  # وارد کردن matplotlib برای نمایش تصاویر / Import matplotlib for displaying images
import numpy as np  # وارد کردن NumPy برای عملیات روی آرایه‌ها / Import NumPy for array operations
from PIL import Image  # وارد کردن PIL برای باز کردن و پردازش تصاویر / Import PIL for image handling
from ultralytics import YOLO  # وارد کردن مدل YOLO از ultralytics / Import the YOLO model from ultralytics
from huggingface_hub import hf_hub_download  # وارد کردن تابع دانلود وزن از Hugging Face / Import the function to download weights from Hugging Face
import logging  # وارد کردن ماژول logging برای ثبت خطاها / Import logging module for error logging

# پیکربندی logging جهت نمایش پیام‌های خطا و اطلاعات / Configure logging to display error and info messages
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ----------------------------- دانلود وزن‌ها و راه‌اندازی مدل‌ها -----------------------------
# Download the vehicle license plate detection model weights from Hugging Face repository
try:
    weights_path = hf_hub_download(
        repo_id="krishnamishra8848/Nepal-Vehicle-License-Plate-Detection", 
        filename="last.pt"
    )
except Exception as e:
    logging.error("Error downloading license plate detection weights: %s", e)
    raise

# راه‌اندازی مدل تشخیص پلاک با وزن‌های دانلود شده / Initialize the license plate detection model with the downloaded weights
try:
    model_detection = YOLO(weights_path)
except Exception as e:
    logging.error("Error initializing license plate detection model: %s", e)
    raise

# بارگذاری مدل تشخیص حروف و اعداد پلاک از فایل محلی / Load the license plate character and numbers recognition model from a local file
try:
    model_processing = YOLO('F:/my/code folder/plak prosesing/runs/detect/train5/weights/best.pt')
except Exception as e:
    logging.error("Error loading license plate character recognition model: %s", e)
    raise

# ----------------------------- تابع پردازش تصویر -----------------------------
def process_image(image_path):
    """
    پردازش تصویر برای تشخیص پلاک و حروف آن / Process image for license plate and character recognition
    """
    # باز کردن تصویر با استفاده از PIL / Open the image using PIL
    try:
        image = Image.open(image_path)
    except Exception as e:
        logging.error("Error opening image '%s': %s", image_path, e)
        return

    # تبدیل تصویر به آرایه NumPy / Convert the image to a NumPy array
    try:
        img = np.array(image)
    except Exception as e:
        logging.error("Error converting image to numpy array: %s", e)
        return

    # تغییر رنگ از BGR به RGB جهت نمایش صحیح / Convert color from BGR to RGB for correct display
    try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        logging.error("Error converting image color space: %s", e)
        return

    # اجرای مدل تشخیص پلاک روی تصویر / Run the license plate detection model on the image
    try:
        results_detection = model_detection(img)
    except Exception as e:
        logging.error("Error during license plate detection: %s", e)
        return

    # پردازش نتایج تشخیص پلاک / Process the detection results
    for result in results_detection:
        # بررسی وجود جعبه‌های تشخیص (bounding boxes) / Check if the result contains detection boxes
        if hasattr(result, 'boxes') and result.boxes is not None:
            # پیمایش بر روی هر جعبه تشخیص یافته / Iterate over each detected bounding box
            for box in result.boxes.xyxy:
                try:
                    # تبدیل مختصات جعبه به اعداد صحیح / Convert the bounding box coordinates to integers
                    x1, y1, x2, y2 = map(int, box)
                    # بریدن ناحیه پلاک از تصویر اصلی / Crop the license plate region from the image
                    plate_crop = img[y1:y2, x1:x2]

                    # بزرگ‌نمایی (Upscale) تصویر پلاک به منظور بهبود کیفیت / Upscale the license plate image to improve quality
                    scale_factor = 2  # فاکتور بزرگنمایی / Scaling factor
                    plate_upscaled = cv2.resize(
                        plate_crop,
                        (plate_crop.shape[1] * scale_factor, plate_crop.shape[0] * scale_factor),
                        interpolation=cv2.INTER_CUBIC  # استفاده از اینترپولاسیون cubic برای کیفیت بهتر / Using cubic interpolation for better quality
                    )

                    # حذف نویز تصویر پلاک با استفاده از الگوریتم fastNlMeansDenoisingColored / Denoise the upscaled license plate image using fastNlMeansDenoisingColored
                    plate_denoised = cv2.fastNlMeansDenoisingColored(
                        plate_upscaled, None,
                        h=10, hColor=10,
                        templateWindowSize=7, searchWindowSize=21
                    )

                    # افزایش وضوح تصویر (Sharpening) با استفاده از فیلتر کانولوشنی / Sharpen the denoised image by applying a convolution filter
                    kernel_sharpening = np.array([
                        [-1, -1, -1],
                        [-1,  9, -1],
                        [-1, -1, -1]
                    ])
                    plate_sharpened = cv2.filter2D(plate_denoised, -1, kernel_sharpening)
                except Exception as e:
                    logging.error("Error processing license plate region: %s", e)
                    continue

                # اجرای مدل تشخیص حروف روی تصویر پلاک پردازش‌شده / Run the character recognition model on the processed license plate image
                try:
                    results_chars_number = model_processing.predict(source=plate_sharpened, conf=0.5)
                except Exception as e:
                    logging.error("Error during character recognition: %s", e)
                    continue

                # پردازش نتایج تشخیص حروف / Process the character detection results
                for r in results_chars_number:
                    # بررسی وجود جعبه‌های تشخیص کاراکترها / Check if there are detected boxes for characters
                    if r.boxes is not None and len(r.boxes) > 0:
                        try:
                            # مرتب‌سازی جعبه‌ها بر اساس مختصات x (چپ به راست) / Sort the bounding boxes by their x-coordinate (from left to right)
                            sorted_boxes = sorted(r.boxes, key=lambda b: b.xyxy[0][0])
                            
                            recognized_labels = []
                            # استخراج برچسب (حرف) هر جعبه / Extract the label (character) from each bounding box
                            for b in sorted_boxes:
                                class_id = int(b.cls[0])
                                label = r.names[class_id]
                                recognized_labels.append(label)
                            # ترکیب حروف تشخیص داده شده با استفاده از علامت "-" به عنوان جداکننده / Combine the recognized characters using "-" as a separator
                            recognized_text = "-".join(recognized_labels)
                            # چاپ متن پلاک شناسایی‌شده / Print the recognized license plate text
                            print("Recognized plate text:", recognized_text[:15])
                        except Exception as e:
                            logging.error("Error processing character recognition results: %s", e)
                            continue

                # نمایش تصویر نهایی پلاک پردازش‌شده / Display the final processed license plate image
                try:
                    plt.imshow(cv2.cvtColor(plate_sharpened, cv2.COLOR_BGR2RGB))
                    plt.axis('off')  # حذف محورهای تصویر / Hide the image axes
                    plt.show()
                except Exception as e:
                    logging.error("Error displaying the processed license plate image: %s", e)

# ----------------------------- نقطه شروع برنامه -----------------------------
if __name__ == "__main__":
    # تعیین مسیر تصویر ورودی و فراخوانی تابع پردازش تصویر / Define the path to the input image and call the process_image function
    image_path = "test-images/7t.jpg"  # مسیر تصویر خودتان / Your image path
    process_image(image_path)
