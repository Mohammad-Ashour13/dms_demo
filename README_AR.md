# نظام مراقبة السائق النهائي

هذه الحزمة تفصل تدريب LightGBM على Colab عن تشغيل Raspberry Pi 5، وتربط الموديل بمحرك أحداث وFusion FSM وتسجيل فيديو للحوادث. التجارب القديمة خارج هذا الفولدر لا تُستخدم تلقائيًا.

## مسار التسليم المختصر

1. ارفع فولدر `dms_final_system` كاملًا إلى `/content/drive/MyDrive/Drive_monitoring/`.
2. افتح `training_colab/DMS_LIGHTGBM_COLAB.ipynb`، عدّل الخلية الأولى فقط، وشغّل Run All بوضع `smoke`.
3. بعد نجاح smoke، غيّر `RUN_MODE` إلى `full` وشغّل Run All مرة ثانية.
4. نزّل `model_bundle.zip`، وافكّه على Raspberry داخل `models/drowsiness/<run_id>/`.
5. أنشئ `models/drowsiness/active_model.json` من المثال وشغّل فحص الـbundle والـgolden samples.
6. شغّل replay على تسجيلات الشركة، اضبط Fusion على مجموعة tuning فقط، ثم جمّد الإعدادات واختبر acceptance.
7. شغّل الكاميرا الحية، اختبر الحوادث وpre/post video، ثم نفّذ soak test لمدة 30 دقيقة.

التفاصيل في:

- `docs/MODEL_FUSION_SYSTEM_COMPLETE_GUIDE_AR.md` — المرجع الشامل للموديل والميزات والـFusion والإنذار وهيكلية الملفات.
- `docs/YOLO_BEHAVIOR_INTEGRATION_AR.md` — تنزيل وتشغيل وتقييم YOLO11n للهاتف والأكل والتدخين.
- `docs/ARCHITECTURE_AR.md`
- `docs/COLAB_RUNBOOK_AR.md`
- `docs/RASPBERRY_RUNBOOK_AR.md`
- `docs/INTEGRATION_CONTRACT_AR.md`
- `docs/ACCEPTANCE_CHECKLIST_AR.md`
- `docs/PERSONAL_EVALUATION_RUNBOOK_AR.md`
- `docs/RUNTIME_V2_REPAIR_RUNBOOK_AR.md`
- `docs/MODEL_V2_RETRAINING_PLAN_AR.md`
- `docs/ALARM_DEMO_RUNBOOK_AR.md`
- `docs/RUNTIME_V22_ALARM_REGRESSION.md`

لعرض الدكتور مع صوت وشاشة استخدم `configs/runtime.laptop_alarm_demo.json`. للـReplay
والتقييم الصامت استخدم `configs/runtime.replay_silent.json`.

## تجربة الموديل المركب حاليًا

الموديل الفعّال هو النسخة المشتقة `20260731T140605Z_full_runtimefix1` بوضع `SHADOW`. أوزان LightGBM والمعايرة لم تتغير؛ النسخة المشتقة تضيف `feature_reference.json` لمراقبة Domain Shift وتحافظ على الأصل `20260731T140605Z_full` للـrollback. اتبع `docs/RUNTIME_V2_REPAIR_RUNBOOK_AR.md` قبل تجربة القبول.

## أوامر التحقق المحلية

من الفولدر الذي يحتوي `dms_final_system/`:

```bash
PYTHONPATH=. python3 -m compileall -q dms_final_system
PYTHONPATH=. pytest dms_final_system/tests
```

اختبارات `training_colab/tests` تشغلها Notebook بعد تثبيت متطلبات التدريب؛ لا تحتاج مكتبات sklearn/Optuna على Raspberry.

لا يُعتبر الـbundle Production Ready بمجرد نجاح تدريب النوافذ. الحالة القصوى التي يصدرها Colab هي `CANDIDATE_PENDING_RUNTIME_ACCEPTANCE`، ولا تصبح النسخة مقبولة إلا بعد replay واختبارات Raspberry على مستوى الإنذار والحادثة.
