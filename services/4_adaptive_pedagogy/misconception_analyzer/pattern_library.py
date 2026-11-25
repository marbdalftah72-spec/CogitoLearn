"""
Expanded misconception pattern library.
========================================
Catalogue of named cognitive-error patterns used across Service 4.  The
v1 diagnoser only stored raw misconception ids; this library gives every
pattern a definition, a category, a severity weight, the Bloom level at
which remediation should be pitched, and a concrete remediation action —
turning bare ids into actionable diagnoses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class MisconceptionPattern:
    """One named misconception pattern."""

    pattern_id: str
    name_ar: str
    description_ar: str
    category: str               # LOGIC | CONCEPTUAL | PROCEDURAL | LINGUISTIC
    severity: float             # 0..1 — how damaging the pattern is if persistent
    remediation_bloom_level: str  # BloomLevel value to pitch remediation at
    remediation_action_ar: str
    related_patterns: List[str] = field(default_factory=list)


PATTERN_LIBRARY: Dict[str, MisconceptionPattern] = {
    p.pattern_id: p
    for p in [
        MisconceptionPattern(
            pattern_id="MISC_INVERSE_LOGIC",
            name_ar="عكس العلاقة المنطقية",
            description_ar="قلب اتجاه العلاقة بين السبب والنتيجة أو بين الشرط والمشروط.",
            category="LOGIC",
            severity=0.85,
            remediation_bloom_level="ANALYZE",
            remediation_action_ar="تدريب على جداول الحقيقة وأمثلة مضادة تُظهر فشل الاتجاه المعكوس.",
            related_patterns=["MISC_CAUSE_EFFECT_SWAP"],
        ),
        MisconceptionPattern(
            pattern_id="MISC_CAUSE_EFFECT_SWAP",
            name_ar="تبديل السبب بالنتيجة",
            description_ar="اعتبار النتيجة سببًا للظاهرة الأصلية والعكس.",
            category="LOGIC",
            severity=0.80,
            remediation_bloom_level="ANALYZE",
            remediation_action_ar="رسم مخططات سببية للظاهرة ومقارنتها بتفسير الطالب خطوة بخطوة.",
            related_patterns=["MISC_INVERSE_LOGIC"],
        ),
        MisconceptionPattern(
            pattern_id="MISC_CONFLATION",
            name_ar="الخلط بين مفهومين متقاربين",
            description_ar="دمج مفهومين مختلفين في مفهوم واحد بسبب تشابه سطحي.",
            category="CONCEPTUAL",
            severity=0.75,
            remediation_bloom_level="UNDERSTAND",
            remediation_action_ar="جدول مقارنة تفصيلي بين المفهومين مع أسئلة تمييز حدّية.",
            related_patterns=["MISC_TERMINOLOGY_CONFUSION"],
        ),
        MisconceptionPattern(
            pattern_id="MISC_TERMINOLOGY_CONFUSION",
            name_ar="الخلط المصطلحي",
            description_ar="استخدام مصطلح مكان آخر قريب منه لفظيًا أو ترجميًا.",
            category="LINGUISTIC",
            severity=0.45,
            remediation_bloom_level="REMEMBER",
            remediation_action_ar="بطاقات مصطلحات ثنائية اللغة مع أمثلة استعمال صحيحة وخاطئة.",
            related_patterns=["MISC_CONFLATION"],
        ),
        MisconceptionPattern(
            pattern_id="MISC_OVERSIMPLIFICATION",
            name_ar="التبسيط المُخلّ",
            description_ar="اختزال آلية معقدة في قاعدة واحدة تُهمل الشروط والاستثناءات.",
            category="CONCEPTUAL",
            severity=0.70,
            remediation_bloom_level="ANALYZE",
            remediation_action_ar="تقديم حالات حدّية تنكسر فيها القاعدة المبسطة وطلب تفسيرها.",
            related_patterns=["MISC_OVERGENERALIZATION", "MISC_EXAMPLE_AS_RULE"],
        ),
        MisconceptionPattern(
            pattern_id="MISC_OVERGENERALIZATION",
            name_ar="التعميم المتسرّع",
            description_ar="تعميم قاعدة من مثال واحد أو حالات قليلة على كل الحالات.",
            category="LOGIC",
            severity=0.65,
            remediation_bloom_level="EVALUATE",
            remediation_action_ar="تدريب على شروط صحة التعميم وطلب أمثلة مضادة قبل قبول أي قاعدة.",
            related_patterns=["MISC_OVERSIMPLIFICATION"],
        ),
        MisconceptionPattern(
            pattern_id="MISC_EXAMPLE_AS_RULE",
            name_ar="اعتبار المثال قاعدة",
            description_ar="معاملة المثال التوضيحي الوارد في المحاضرة كتعريف شامل للمفهوم.",
            category="CONCEPTUAL",
            severity=0.55,
            remediation_bloom_level="UNDERSTAND",
            remediation_action_ar="عرض أمثلة متنوعة وأمثلة لا-أمثلة (non-examples) للمفهوم نفسه.",
            related_patterns=["MISC_OVERGENERALIZATION"],
        ),
        MisconceptionPattern(
            pattern_id="MISC_INVARIANCE_FALLACY",
            name_ar="مغالطة الثبات",
            description_ar="افتراض أن مخرجات الخوارزمية ثابتة مهما تغيّرت المدخلات والسياق.",
            category="LOGIC",
            severity=0.60,
            remediation_bloom_level="APPLY",
            remediation_action_ar="تجارب تطبيقية بتغيير مدخل واحد في كل مرة وملاحظة تغيّر المخرجات.",
            related_patterns=["MISC_CONTEXT_NEGLECT"],
        ),
        MisconceptionPattern(
            pattern_id="MISC_CONTEXT_NEGLECT",
            name_ar="إهمال السياق",
            description_ar="تطبيق قاعدة صحيحة في سياق لا تنطبق عليه شروطها.",
            category="PROCEDURAL",
            severity=0.70,
            remediation_bloom_level="APPLY",
            remediation_action_ar="قائمة تحقق من شروط التطبيق قبل استخدام القاعدة في مسائل جديدة.",
            related_patterns=["MISC_INVARIANCE_FALLACY"],
        ),
        MisconceptionPattern(
            pattern_id="MISC_STOCHASTIC_CONFUSION",
            name_ar="الخلط العشوائي",
            description_ar="افتراض أن النموذج عشوائي بالكامل وإغفال قيوده الحتمية والمنطقية.",
            category="CONCEPTUAL",
            severity=0.65,
            remediation_bloom_level="UNDERSTAND",
            remediation_action_ar="تفكيك النموذج إلى أجزاء حتمية وأجزاء احتمالية مع أمثلة عددية.",
            related_patterns=["MISC_OVERSIMPLIFICATION"],
        ),
        MisconceptionPattern(
            pattern_id="MISC_SUPERFICIAL_REDUCTION",
            name_ar="الاختزال الشكلي",
            description_ar="اعتبار المفهوم إجراءً شكليًا بلا أثر فعلي على النتائج.",
            category="CONCEPTUAL",
            severity=0.50,
            remediation_bloom_level="UNDERSTAND",
            remediation_action_ar="مقارنة تجريبية للأداء مع المفهوم وبدونه لإظهار أثره القابل للقياس.",
            related_patterns=["MISC_OVERSIMPLIFICATION"],
        ),
        MisconceptionPattern(
            pattern_id="MISC_PARTIAL_APPLICATION",
            name_ar="التطبيق الجزئي",
            description_ar="تنفيذ بعض خطوات الإجراء وإغفال خطوات جوهرية منه.",
            category="PROCEDURAL",
            severity=0.60,
            remediation_bloom_level="APPLY",
            remediation_action_ar="تفكيك الإجراء إلى قائمة خطوات مرقّمة والتدرب عليها بمسائل موجّهة.",
            related_patterns=["MISC_PROCEDURE_WITHOUT_CONCEPT"],
        ),
        MisconceptionPattern(
            pattern_id="MISC_PROCEDURE_WITHOUT_CONCEPT",
            name_ar="إجراء بلا فهم",
            description_ar="حفظ خطوات الحل آليًا دون فهم مبرراتها، فينهار الأداء عند تغيير شكل المسألة.",
            category="PROCEDURAL",
            severity=0.75,
            remediation_bloom_level="UNDERSTAND",
            remediation_action_ar="أسئلة «لماذا هذه الخطوة؟» بعد كل خطوة، ومسائل بشكل مختلف والخطوات نفسها.",
            related_patterns=["MISC_PARTIAL_APPLICATION"],
        ),
        MisconceptionPattern(
            pattern_id="MISC_NEGATION_ERROR",
            name_ar="خطأ النفي",
            description_ar="سوء التعامل مع النفي في العبارات المنطقية (نفي الجملة المركبة خاصة).",
            category="LOGIC",
            severity=0.70,
            remediation_bloom_level="APPLY",
            remediation_action_ar="تدريب مركّز على قوانين دي مورغان ونفي الكمّيات بأمثلة لغوية ورياضية.",
            related_patterns=["MISC_INVERSE_LOGIC"],
        ),
    ]
}


def get_pattern(pattern_id: str) -> Optional[MisconceptionPattern]:
    """Look up a pattern by id; returns ``None`` for unknown ids."""
    return PATTERN_LIBRARY.get(pattern_id)


def patterns_by_category(category: str) -> List[MisconceptionPattern]:
    """All patterns in a category, most severe first."""
    return sorted(
        (p for p in PATTERN_LIBRARY.values() if p.category == category),
        key=lambda p: p.severity,
        reverse=True,
    )
