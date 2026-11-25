---
title: Code Blue ECG AI Backend
emoji: 🫀
colorFrom: red
colorTo: indigo
sdk: docker
app_port: 7860
---

# Code Blue ECG AI Backend (100% Free CPU HuggingFace Space)

## Licenses (All Permissive — Zero GPL/AGPL)
- **Open-ECG-Digitizer** (`Ahus-AIM/Open-ECG-Digitizer`): CC BY 4.0
- **ECGFounder** (`PKUDigitalHealth/ECGFounder`): MIT License
- **FastAPI / Uvicorn / PyTorch / NumPy / SciPy / Pillow / OpenCV**: MIT / BSD / Apache-2.0

## طريقة النشر المجاني في 3 خطوات (دقيقتين فقط)
1. افتح [huggingface.co/new-space](https://huggingface.co/new-space) واختر:
   - **Space name:** `codeblue-ecg-ai`
   - **SDK:** `Docker` → `Blank`
   - **Hardware:** `CPU basic (Free)`
2. ارفع الملفات الأربعة الموجودة في هذا المجلد (`Dockerfile`, `requirements.txt`, `app.py`, `README.md`) داخل الـ Space.
3. بعد اكتمال البناء (يظهر `Running`)، انسخ رابط الـ Space مثل:
   `https://<your-username>-codeblue-ecg-ai.hf.space`
   وضعه في ملف `.env.local` وفي إعدادات Vercel تحت المتغير:
   ```
   NEXT_PUBLIC_ECG_BACKEND_URL=https://<your-username>-codeblue-ecg-ai.hf.space
   ```
