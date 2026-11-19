# Code Blue Sounds & ECG Lab — دليل التشغيل السريع

هذا الموقع مستقل بالكامل لأصوات القلب والـ ECG وأصوات الصدر والتنفس (يعمل على المنفذ `3002` محلياً).
جميع أصوات القلب، الـ Murmurs، تخطيط الـ ECG المباشر، وأصوات الصدر والتنفس تعمل فورياً حتى قبل نسخ مجسمات الـ 3D.

## خطوات التشغيل على جهازك (بعد نقله إلى `F:\codeblue-sounds`)
1. انسخ `.env.local` من `F:\mrcs-bank\.env.local` إلى `F:\codeblue-sounds\.env.local`.
2. لنسخ مجسمات القلب والرئة فقط (بسرعة وبحجم خفيف جداً) افتح PowerShell داخل `F:\codeblue-sounds` ونفذ:
```powershell
.\COPY_GLB_ASSETS.ps1
```
3. ثبّت المكتبات وشغّل الموقع:
```powershell
npm install
npm run dev
```
4. افتح المتصفح على: `http://localhost:3002`
