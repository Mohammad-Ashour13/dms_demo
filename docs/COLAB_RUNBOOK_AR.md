# دليل Colab خطوة بخطوة

## 1. ترتيب الملفات على Drive

```text
/content/drive/MyDrive/Drive_monitoring/
├── training_colab/
│   ├── DMS_LIGHTGBM_COLAB.ipynb
│   ├── dms_training/
│   ├── tests/
│   └── requirements.txt
└── feature_engineeringV3/outputs/20260717T160540Z/
    ├── window_features_train.csv
    ├── window_features_test.csv
    └── feature_decisions.csv
```

يمكن وضع CSV في مكان آخر، لكن يجب تعديل المسارات في أول خلية. لا تنقل CSV داخل git package.

## 2. Smoke

1. افتح `DMS_LIGHTGBM_COLAB.ipynb`.
2. اترك `RUN_MODE="smoke"` و`SEEDS=[42,43,44]` كما هي؛ smoke يستخدم Seed 42 فقط داخليًا.
3. اختر High-RAM إن توفر. GPU ليس مطلوبًا لـLightGBM؛ `USE_GPU=False` هو الخيار المستقر.
4. Run All.
5. تحقق أن pytest نجح وأن `split_audit.csv` كل قيم `overlap_groups` فيه صفر.
6. لا تستخدم أرقام smoke في التقرير العلمي.

## 3. Full

غيّر `RUN_MODE="full"` ثم Run All. المراحل هي:

- Stage A: Top‑7/20/40/65، خمسة grouped folds بإعداد محافظ واحد.
- Stage B: أفضل قائمتين، 25 Optuna trial لكل قائمة.
- Stage C: seeds 42/43/44، OOF، LODO، Platt calibration وthreshold من OOF، ثم final fit وtest مرة واحدة.

SAFEE لا يدخل fitting. إذا وجد في test يظهر فقط كـpositive stress test ولا يدخل ROC-AUC الأساسي.

## 4. قراءة النتائج

ابدأ بـ`decision_report.md` ثم:

- `metrics_overall.json`: OOF/test والبوابة.
- `metrics_per_dataset.csv`: انهيار domain داخل test.
- `lodo_metrics.csv`: النقل إلى dataset غير مرئية.
- `seed_metrics.csv`: حساسية seed.
- `errors_fp_fn.csv`: FP/FN مع الفيديو ووقت النافذة.
- `threshold_sweep.csv`: يثبت أن threshold جاء من OOF.

`EXPERIMENTAL` يعني telemetry-only. `CANDIDATE_PENDING_RUNTIME_ACCEPTANCE` يعني اجتاز بوابة النوافذ فقط.

كل Bundle جديد يتضمن `feature_reference.json` من fitting train فقط، ويحتوي q01/q05/median/q95/q99 وgain rank للميزات. لا يدخل test في هذا المرجع.

## 5. تسليم الـbundle

نزّل `model_bundle.zip` ولا تعدّل محتوياته. على Raspberry نفّذ فحص checksum وgolden parity قبل تعديل `active_model.json`. احتفظ بالنسخة السابقة للrollback.
