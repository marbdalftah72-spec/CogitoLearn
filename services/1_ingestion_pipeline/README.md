# 📥 Service 1: Multimodal Ingestion Pipeline

مرحباً بك يا مهندس الـ Ingestion! 👋  
أنت البوابة الأولى لكل الداتا اللي بتدخل النظام؛ لو شغلك دقيق، كل الفريق اللي بعدك هيشتغل بأعلى دقة!

---

## 🎯 دورك في المشروع باختصار (Mission):
المحاضرات بتجيلنا بصيغ مختلفة: ملفات PDF، ملفات Word، تسجيلات صوتية (MP3/WAV)، أو فيديوهات شرح (MP4).  
دورك إنك تستلم الملفات دي وتحولها لـ **نصوص منظمة شجرية (UnifiedDocumentAST)** بدون ما تفقد:
1. **المعادلات الرياضية**: لازم تخرج بصيغة $\LaTeX$ مش كلام مشوه.
2. **الجداول**: تخرج بصيغة Markdown نظيفة.
3. **الصوت والفيديو**: يتفرغ لنص مع توقيت زمني دقيق لكل كلمة (Word-level timestamps).
4. **الرسوم والمخططات**: تلتقط صورها وتطلب من Gemini Vision يشرح دلالتها.

---

## 📂 خريطة المجلد ومسؤولية كل ملف:

```
1_ingestion_pipeline/
├── parsers/
│   ├── docling_parser.py    # استخدام مكتبة Docling لقراءة الـ PDF/Word واستخراج الجداول والمعادلات
│   └── text_normalizer.py   # تنظيف النصوص وإزالة الرموز الغريبة وتوحيد الحروف العربية
├── audio/
│   ├── whisper_transcriber.py # تفريغ الصوت بـ faster-whisper و WhisperX مع التوقيت بالمللي ثانية
│   └── code_switch_handler.py # معالجة الكلمات الإنجليزية وسط الشرح العربي (Code-switching)
├── vision/
│   └── diagram_descriptor.py  # إرسال الرسوم التخطيطية لـ Gemini 2.5 Flash Vision لتحويلها لنص
└── pipeline.py                # الملف الرئيسي اللي بينسق كل الخطوات ويخرج الـ UnifiedDocumentAST
```

---

## 🔄 ترتيبك في الفريق (مين هيديك ومين هتسلم له):
* **📥 مين بيديك داتا؟**  
  واجهة المستخدم (Service 5) هتبعتلك الملف الخام اللي الطالب رفعه (`RawMediaStream`).
* **📤 مين هتسلمه داتا؟**  
  هتسلم كائن بيانات موحد اسمه `UnifiedDocumentAST` لمهندس الذاكرة والرسم المعرفي (**Service 2**).

---

## 📦 المكتبات المطلوبة لتشغيل مجلدك:
```bash
pip install docling faster-whisper whisperx google-generativeai pydantic
```


## Dataset 
Download the dataset here:
[Google Drive Dataset](https://drive.google.com/drive/folders/17JuWLTBgzolWxFjpcurNfq7pymYRe14s?usp=drive_link)

