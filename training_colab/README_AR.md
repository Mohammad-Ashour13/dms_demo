# تدريب LightGBM على Colab

الـNotebook تقرأ V3 window CSV مباشرة ولا تعيد بناء النوافذ. `test_windows` ثابت ولا يدخل اختيار الميزات أو Optuna أو calibration أو threshold.

ارفع فولدر `training_colab` نفسه إلى:

`/content/drive/MyDrive/Drive_monitoring/training_colab`

الفولدر مستقل للتدريب، ويجب أن يحتوي مباشرة على `dms_training/` و`tests/` و`requirements.txt` والـNotebook. لا تحتاج لرفع بقية `dms_final_system` حتى تدرب.

ثم افتح `DMS_LIGHTGBM_COLAB.ipynb` وعدّل الخلية الأولى. الإعداد الحالي يستخدم GPU ويتوقف برسالة واضحة إن لم تختر T4 GPU. إذا تعذر backend الخاص بـLightGBM نفسه، ينتقل التدريب إلى CPU لأن بعض Colab wheels لا تحتوي OpenCL رغم ظهور بطاقة NVIDIA.

`OPTUNA_TRIALS=25` تعني 25 trial **لكل واحدة** من أفضل قائمتي ميزات. مع 5 folds ينتج عن Stage B عدد `2 × 25 × 5 = 250` fits. التشغيل الجديد يصدر `fit_audit.csv` بصف لكل fit و`execution_summary.json` يقارن العدد المخطط بالمنفذ.

عند استبدال ملفات الباكيج على Google Drive، استخدم `Runtime > Restart session` مرة واحدة ثم `Run all`. النسخة الحالية تمسح أيضًا أي `dms_training` قديم من ذاكرة Colab وتتحقق من `OUTPUT_CONTRACT_VERSION=2` قبل بدء التدريب، حتى لا تختلط Notebook جديدة بكود تدريب قديم.

شغّل أولًا `RUN_MODE="smoke"`. نجاح smoke يعني سلامة الكود والعقود فقط، وليس نتيجة علمية. بعد ذلك غيّرها إلى `full` وشغّل Run All.

تقرأ الـNotebook ملفات CSV مباشرة من مسارات Google Drive المحددة ولا تنسخها إلى `/content`. المخرجات تحفظ في `OUTPUT_PATH/<run_id>/` على Drive، ويُنشأ ZIP للنتائج وZIP مستقل للـModel Bundle.
