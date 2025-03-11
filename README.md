# License Plate Recognition and Processing Application  
**برنامه تشخیص و پردازش پلاک خودرو**

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=Python&logoColor=white)](https://www.python.org/)  
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=OpenCV&logoColor=white)](https://opencv.org/)  
[![NumPy](https://img.shields.io/badge/NumPy-1.19%2B-013243?logo=numpy&logoColor=white)](https://numpy.org/)  
[![Pillow](https://img.shields.io/badge/Pillow-8.x-4484CE?logo=Pillow&logoColor=white)](https://python-pillow.org/)  
[![YOLO](https://img.shields.io/badge/YOLO-Ultralytics-green?logo=YOLO&logoColor=white)](https://github.com/ultralytics/ultralytics)  
[![Hugging Face Hub](https://img.shields.io/badge/HuggingFace-Hub-blue?logo=HuggingFace&logoColor=white)](https://huggingface.co/)  
[![Tkinter](https://img.shields.io/badge/Tkinter-GUI-blue?logo=Tkinter&logoColor=white)](https://docs.python.org/3/library/tkinter.html)

---

## Table of Contents (English)
1. [Introduction](#introduction)
2. [Key Features](#key-features)
3. [Technologies Used](#technologies-used)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Troubleshooting](#troubleshooting)
7. [License](#license)

---

## فهرست مطالب (فارسی)
1. [مقدمه](#مقدمه)
2. [ویژگی‌های اصلی](#ویژگی‌های-اصلی)
3. [تکنولوژی‌های استفاده‌شده](#تکنولوژی‌های-استفاده‌شده)
4. [نصب و راه‌اندازی](#نصب-و-راه‌اندازی)
5. [نحوه استفاده](#نحوه-استفاده)
6. [رفع مشکلات](#رفع-مشکلات)
7. [لایسنس](#لایسنس)

---

## Introduction
This project is a desktop application that processes input images to detect vehicle license plates using a YOLO-based detection model. Once detected, the license plate region is cropped, enhanced (upscaling, denoising, and sharpening), and then passed to a second YOLO model that performs character recognition on the plate. The user-friendly GUI built with Tkinter allows users to easily select images and view the processed results.

---

## مقدمه
این پروژه یک برنامه دسکتاپ است که تصاویر ورودی را به منظور تشخیص پلاک خودرو با استفاده از یک مدل مبتنی بر YOLO پردازش می‌کند. پس از شناسایی پلاک، منطقه پلاک برش خورده و با اعمال عملیات افزایش وضوح، حذف نویز و شارپ کردن بهبود می‌یابد. سپس تصویر به مدل دوم YOLO ارسال می‌شود تا حروف و اعداد موجود در پلاک شناسایی شوند. رابط کاربری ساده و کاربرپسند با استفاده از Tkinter طراحی شده تا کاربران بتوانند به راحتی تصاویر را انتخاب و نتایج پردازش‌شده را مشاهده کنند.

---

## Key Features / ویژگی‌های اصلی
- **License Plate Detection / تشخیص پلاک خودرو:**  
  Automatically detects license plates within input images using a YOLO model.
  
- **Image Pre-processing / پیش‌پردازش تصویر:**  
  Crops, upscales, denoises, and sharpens the detected license plate region for better recognition.
  
- **Character Recognition / شناسایی حروف:**  
  Utilizes a second YOLO model to recognize characters on the cropped license plate image.
  
- **Desktop GUI / رابط کاربری دسکتاپ:**  
  Provides an intuitive Tkinter-based interface for selecting images and displaying results.
  
- **Error Logging / ثبت خطا:**  
  Incorporates logging for error tracking and debugging.

---

## Technologies Used / تکنولوژی‌های استفاده‌شده
- **Python**  
- **OpenCV**  
- **NumPy**  
- **Pillow (PIL)**  
- **YOLO (Ultralytics)**  
- **Hugging Face Hub**  
- **Tkinter**

---

## Installation / نصب و راه‌اندازی

### English
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name

----
----
![output4cnn](https://github.com/SamKhandani/Plate-processing/blob/main/runs/detect/train5/val_batch2_pred.jpg)
