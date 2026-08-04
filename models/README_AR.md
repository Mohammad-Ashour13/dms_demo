# الموديلات المحلية

ضع ملف MediaPipe في:

`models/mediapipe/face_landmarker.task`

وفك Model Bundle القادم من Colab في:

`models/drowsiness/<run_id>/`

ثم انسخ `active_model.example.json` إلى `active_model.json` وعدّل `bundle_path` إلى اسم `<run_id>`. يجب أن يحتوي ذلك الفولدر مباشرة على `model.txt` وملفات JSON الخاصة بالـbundle.

الموديل الحالي مركب ومفعل بوضع `SHADOW` في:

`models/drowsiness/20260731T140605Z_full_runtimefix1/`

حالته `EXPERIMENTAL`؛ يمنع الـRuntime تشغيله بوضع `ACTIVE`. هذه نسخة مشتقة تضيف `feature_reference.json` فقط، بينما يبقى `20260731T140605Z_full/` دون تعديل للـrollback. ملف MediaPipe المطلوب موجود في `models/mediapipe/face_landmarker.task`.
