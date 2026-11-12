# 🌟 CogitoLearn: Multimodal Cognitive Architecture & Evidence-Grounded Learning Engine

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Pydantic v2](https://img.shields.io/badge/Pydantic-v2-green.svg)](https://docs.pydantic.dev/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Framework-teal.svg)](https://fastapi.tiangolo.com/)
[![Architecture](https://img.shields.io/badge/Architecture-Decoupled%20Micro--Domain-purple.svg)]()

> مشروع مقرر معالجة اللغات الطبيعية (NLP) المتقدم: مساعد تعليمي معرفي متعدد الوسائط يعتمد على الفهرسة الهرمية، والرسوم البيانية المعرفية (PKG)، والتحقق المنطقي والجبري الصارم، ونمذجة إتقان الطالب البايزية.

---

## 🏗️ معمارية المنظومة وتقسيم فريق العمل (Team Topology)

تم تقسيم المشروع إلى 5 نطاقات برمجية مستقلة (Decoupled Domain Services) تتخاطب عبر عقود بيانات موحدة (`common.contracts`):

```
CogitoLearn/
├── common/                               # عقود البيانات المركزية الموحدة (Pydantic v2)
│   └── contracts.py
│
├── services/                             # الخدمات المنفصلة (موزعة على أعضاء الفريق)
│   ├── 1_ingestion_pipeline/             # 1. استخالص المستندات (Docling) وتفريغ الوسائط (WhisperX)
│   ├── 2_memory_and_pkg/                 # 2. الفهرسة الهرمية (BGE-M3 + Qdrant) وبناء الرسم المعرفي (PKG)
│   ├── 3_router_and_verifier/            # 3. توجيه الاستعلامات، وتنسيق RAG، وبوابة التحقق (NLI + SymPy)
│   ├── 4_adaptive_pedagogy/              # 4. بنوك أسئلة بلوم، وتشخيص الأخطاء، ونمذجة الطالب (BKT + FSRS)
│   └── 5_ui_and_eval_harness/            # 5. واجهة الاستخدام، واجهات FastAPI، ومحرك التجارب المعيارية
│
├── data/                                 # مجلد بيانات المحاضرات التجريبية
├── configs/                              # إعدادات النظام ومفاتيح التشغيل
└── tests/                                # الاختبارات الآلية المستقلة
```

---

## 👥 مسؤوليات الفريق وتبادل البيانات (Service Contract-First)

| الخدمة | المسؤول | المدخلات (In) | المخرجات (Out) | التقنيات الأساسية |
|---|---|---|---|---|
| **1. Ingestion Pipeline** | Multimodal Engineer | `RawMediaStream` (PDF, Video) | `UnifiedDocumentAST` | Docling, WhisperX, Gemini Vision |
| **2. Memory & PKG** | Vector DB & Graph Engineer | `UnifiedDocumentAST` | `HybridIndex` & `PKG` | BGE-M3, Qdrant, PostgreSQL |
| **3. Router & Verifier** | RAG & Verification Engineer | `Query` + `StudentState` | `VerifiedResponse` | Adaptive Router, mDeBERTa NLI, SymPy |
| **4. Adaptive Pedagogy** | Pedagogy Engineer | `ConceptID` + `EvidenceSpans` | `BloomBank` + `MasteryState` | Bloom Taxonomy, BKT, FSRS-4.5 |
| **5. System & UI** | System Engineer | مخرجات كافة الخدمات | `FastAPI` + Interactive UI | FastAPI, Streamlit, OpenTelemetry |

---

## 🚀 دليل الإعداد السريع (Quickstart)

### 1. استنساخ المستودع وتثبيت المتطلبات:
```bash
git clone https://github.com/marbdalftah72-spec/CogitoLearn.git
cd CogitoLearn
pip install -r requirements.txt
```

### 2. إعداد المتغيرات البيئية:
انسخ ملف الإعدادات النموذجي:
```bash
cp configs/.env.example configs/.env
```
وقم بإدخال مفتاح `GOOGLE_API_KEY` الخاص بـ Gemini.

### 3. تشغيل واجهة النظام:
```bash
streamlit run services/5_ui_and_eval_harness/app.py
```
