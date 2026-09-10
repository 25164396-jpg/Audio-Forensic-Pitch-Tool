# Vocalis Forensic Restorer

أداة CLI لفحص ملفات WAV الصوتية وإنشاء مرشحات طبقة صوت مع تقرير جنائي قابل للأرشفة.

## التشغيل

```powershell
python voice_restore.py --help
python voice_restore.py .\evidence.wav --output .\results
```

سيظهر مجلد باسم التسجيل داخل مجلد النتائج، ويحتوي على:

- `00_BEST_NATURAL_VOICE_RECOMMENDED.wav`: أعلى مرشح وفق مؤشر FIQI.
- ملفات المرشحات الأخرى للمقارنة اليدوية.
- `forensic_report.json`: بيانات الملف ونتائج `F0` و`HNR` ونسبة الكلام ودرجات كل مرشح.

يمكن تخصيص المرشحات دون تعديل الكود:

```json
{"factors": [0.82, 0.90, 1.00, 1.10, 1.20]}
```

ثم:

```powershell
python voice_restore.py evidence.wav --config factors.json
```

## حدود علمية مهمة

لا توجد طريقة مضمونة لاستنتاج الصوت الأصلي أو هوية المتحدث من تسجيل عُدّل بنسبة مجهولة. الأداة تنتج مرشحات تحليلية، وتحافظ على عدد العينات ومدة الملف، لكنها لا تُعد بديلاً عن فحص خبير أو دليل بيومتري مستقل.

المكتبة الخارجية الوحيدة هي `numpy`. أما `wave` و`argparse` و`json` وعمليات الملفات فتستخدم Python القياسي.
