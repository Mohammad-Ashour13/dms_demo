# تحليل Full LightGBM — 20260731T130149Z

## الحكم المختصر

التشغيل اكتمل ولم يتوقف في Stage B. نفّذ 50 Optuna trial: 25 لقائمة Top‑40 و25 لقائمة Top‑65. كل trial نفّذ 5 grouped folds، ولذلك Stage B نفّذت 250 fit. عدم ظهور 250 صفًا سببه أن `optuna_trials.csv` القديم يسجل صفًا واحدًا لكل trial ولا يسجل folds الداخلية.

الموديل الناتج غير صالح ككاشف إنذار مستقل، وتصنيفه الصحيح هو `EXPERIMENTAL`. لا يوجد threshold يحقق Precision≥75% وRecall≥90% معًا.

## عدد عمليات التدريب

| المرحلة | الحساب | Fits |
|---|---:|---:|
| Stage A | 4 feature lists × 5 folds | 20 |
| Stage B | 2 lists × 25 trials × 5 folds | 250 |
| Stage C | 3 seeds × 5 folds | 15 |
| LODO | 3 held-out datasets | 3 |
| Final fit | كامل train | 1 |
| المجموع |  | **289** |

الإصدار المصحح يصدر `fit_audit.csv` من 289 صفًا و`execution_summary.json` بدل استنتاج العدد من ملف Optuna.

## مشكلة Grouping المكتشفة

في NTHU ظهرت معرفات لنفس السائق بصيغ مختلفة:

- `001` و`1`
- `005` و`5`

في Seed 42 انقسم سائقان بهذه الطريقة بين foldين، وفي Seed 44 انقسم سائق واحد. لم يوجد overlap مع test بعد توحيد المعرفات؛ test يستخدم السائق 6 بينما train يستخدم 1 و2 و5. لذلك نتيجة test السيئة ليست ناتجة عن train/test leakage، لكن CV السابق ليس نظيفًا 100% ويجب عدم اعتماد calibration/threshold الناتجين كنسخة نهائية.

تم تعديل `canonical_group` بحيث تصبح `001 == 1 == 1.0` قبل أي split، مع اختبار آلي لهذا العقد.

## النتائج الإجمالية

| المصدر | Precision | Recall | F1 | FPR | PR-AUC | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| OOF عند threshold 0.372 | 58.70% | 85.13% | 69.49% | 59.42% | 75.03% | 73.49% |
| Fixed test | 58.70% | 69.82% | 63.78% | 45.66% | 69.13% | 68.02% |

Test confusion matrix:

- TN: 4,447
- FP: 3,737
- FN: 2,296
- TP: 5,311

الـFPR البالغ 45.7% يمنع استخدام threshold 0.372 كإنذار مباشر.

## هل الأهداف ممكنة بتغيير threshold فقط؟

لا.

- على OOF، أفضل Precision مع Recall≥90% هي تقريبًا **55.6%** فقط، مع FPR≈71.4%.
- أفضل Recall عند Precision≥75% هي تقريبًا **51.3%** فقط.

المشكلة ليست اختيار threshold سيئًا؛ توزيع الاحتمالات بين الفئتين متداخل بقوة.

على test، threshold مرتفع مثل 0.75 يعطي تقريبًا Precision=80.4% وRecall=24.2% وFPR=5.5%. هذا يجعله دليلًا عالي الثقة داخل Fusion، لكنه لا يغطي النعاس وحده. هذه قيمة تحليلية وليست threshold جديدة معتمدة، لأن test لا يجوز استخدامه للتوليف.

## النتائج حسب Dataset

| Dataset | Precision | Recall | F1 | FPR | PR-AUC | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| NTHU | 43.66% | 38.51% | 40.92% | 34.78% | 44.52% | 45.75% |
| SUST | 57.95% | 70.04% | 63.43% | 44.68% | 65.06% | 68.49% |
| UTA | 59.44% | 70.72% | 64.59% | 46.69% | 71.43% | 68.20% |

NTHU test قريب من العشوائي، بينما SUST وUTA متوسطان لكن FP ما زال مرتفعًا.

## LODO

- Held-out NTHU: PR-AUC 0.461، FPR 90.2%.
- Held-out SUST: PR-AUC 0.489، ROC-AUC 0.509؛ قريب جدًا من العشوائي.
- Held-out UTA: PR-AUC 0.734، وهو الأفضل.

هذا يثبت domain shift؛ الموديل يتعلم أنماطًا لا تنتقل بنفس القوة بين مصادر البيانات.

## ثبات Seeds

- Macro PR-AUC: من 0.650 إلى 0.716.
- Worst-dataset PR-AUC: من 0.455 إلى 0.658.
- جميع Seeds فشلت شرط Precision/Recall.

التفاوت في worst-domain كبير، ولذلك اختيار seed جيد وحده لا يحل المشكلة.

## Stage B واختيار الميزات

- Stage A: Top‑40 كانت الأفضل قبل tuning بـmacro PR-AUC≈0.643.
- Stage B: Top‑65 تحسنت إلى macro PR-AUC≈0.686 وworst≈0.658 واختيرت.
- جميع Trials الخمسين كانت `constraints_met=False`.

Optuna حسّنت ranking، لكنها لم تجعل هدف 75/90 ممكنًا. زيادة Trials إضافية لن تكون الخطوة الأعلى قيمة.

أعلى الميزات في الموديل النهائي شملت `ear_q25`, `left_ear_min`, `roll_energy`, `mar_variance`, `mar_energy`, `yaw_coefficient_of_variation` وميزات pose/motion أخرى. هذا يؤكد ضرورة اختبار تطابق MediaPipe/runtime مع تعريفات V3، لأن الموديل لا يعتمد على EAR وحدها.

## SAFEE

SAFEE احتوت 53 نافذة positive stress فقط:

- متوسط الاحتمال: 0.766
- positive rate: 96.2%

هذه إشارة جيدة للحساسية، لكنها لا تقيس specificity ولا يمكن دمجها في المقاييس الأساسية لأنها single-label.

## القرار والخطوة التالية

1. لا تستخدم bundle الحالية كإنذار مستقل، ولا تعتمد threshold 0.372 في production.
2. ارفع نسخة `training_colab` المصححة إلى Drive؛ هي توحد subject IDs وتصدر سجل 289 fit واضحًا.
3. لا نعيد 250 fit بهدف زيادة Optuna فقط؛ جميع Trials فشلت والحد الحالي سببه labels/domain overlap أكثر من hyperparameters.
4. شغّل Replay على تسجيلات الشركة، مقسمة حسب السائق إلى tuning وacceptance.
5. استخدم LightGBM كدليل risk داخل Fusion:
   - probability مرتفعة ومستمرة = دليل قوي.
   - probability متوسطة لا تطلق إنذارًا وحدها؛ تحتاج PERCLOS/blink/yawn/closure evidence.
   - prolonged eye closure تبقى Critical مباشرة دون انتظار الموديل.
6. اجمع FP من company tuning فقط واربطها بالفيديو والـtelemetry. هذه هي hard negatives الأعلى قيمة.
7. إذا كان مطلوبًا أن يحقق الموديل وحده Recall 90% وPrecision 75%، نحتاج labels زمنية أدق أو weak-label/MIL protocol جديد؛ إعادة ضبط LightGBM على نفس labels لن تكفي.

الـtest الحالي أصبح test تشخيصيًا بعد قراءة نتائجه. أي tuning لاحق يجب ألا يدعي نجاحًا نهائيًا عليه؛ بوابة القبول النهائية يجب أن تكون تسجيلات شركة untouched ومقسمة حسب driver.
