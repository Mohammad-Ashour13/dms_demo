# المعمارية النهائية

## حدود المسؤولية

`app.py` هو Composition Root فقط. الالتقاط لا يحسب EAR، وMediaPipe لا يقرر الإنذار، والموديل لا يعد الرمشات، وFusion لا يقرأ frame مباشرة، والـRecorder لا يعرف قواعد النعاس. هذا الفصل يسمح بإضافة Detector للهاتف أو التدخين من خلال `EvidenceBus` دون إعادة تدريب موديل النعاس أو تعديل داخله.

المسار التشغيلي:

```text
USB camera -> latest-frame capture -> MediaPipe -> personal calibration
           -> compressed rolling recorder
MediaPipe -> event engine -> 2s/30-point V3 feature window -> LightGBM + Platt
events + calibrated probability + quality -> Fusion FSM -> state/violations/alarm
alarm -> incident video + incident.json + telemetry.jsonl -> filesystem outbox
SHADOW evaluation -> full raw session video + frame timestamps + manual annotations
                  -> timeline/episode metrics + FP/FN report
```

هناك مساران مستقلان للعين. `model_ear` يحافظ على صيغة V3 القديمة ولا يتغير حتى تبقى Golden parity صحيحة. `event_ear` يستخدم أزواج جفن متناظرة ومعايرة لكل عين، وهو المصدر الوحيد للرمش والإغلاق وPERCLOS. لا يسمح بتمرير Event EAR إلى الموديل الحالي.

## الزمن والـFPS

الكاميرا تضبط على 15 FPS، لكن كل مدة ونافذة وTTL تعتمد `monotonic_sec`. نافذة الموديل مدتها ثانيتان وتُعاد كل 0.5 ثانية، ثم تعاد عيناتها إلى 30 نقطة. لذلك هبوط FPS لا يحول blink مدته 200ms إلى حدث مختلف؛ الذي يحدد الحدث هو الزمن.

## عقد الموديل

ترتيب الميزات لا يكتب يدويًا في Raspberry. يقرأه runtime من `feature_schema.json` ويبني القيم بهذا الترتيب. الـbundle يرفض checksum خاطئ، schema version غير متوافق، metadata، ميزات `mouth_open_*` الخاصة بعتبات datasets، أو ميزة لا يحسبها runtime.

LightGBM يخرج raw probability، ثم تطبق معاملات Platt من `calibration.json`. Fusion تستخدم calibrated probability ولا تستخدم label مباشر. بما أن الموديل الحالي `EXPERIMENTAL`، الاحتمال المنفرد يستطيع توليد Warning فقط؛ Drowsy يحتاج دليلًا فسيولوجيًا، وCritical لا ينتج إلا عن إغلاق عين طويل موثوق.

`feature_reference.json` يصف q01/q99 لأهم ميزات التدريب. إذا تجاوزت أكثر من 30% من أهم 20 ميزة مجال التدريب لثلاث نوافذ، يسجل النظام `MODEL_INPUT_OOD` ويوقف مساهمة الموديل مؤقتًا مع استمرار مسار الأحداث.

## الـQueues والفشل

- Camera AI queue محدودة وتحتفظ بالأحدث؛ التأخر يسقط frames قديمة بدل صنع latency متزايدة.
- Recorder وtelemetry لهما workers وqueues محدودة مع counters ظاهرة في Health.
- فشل المعايرة أو جودة الوجه ينتج `UNKNOWN`، وليس `NORMAL`.
- فشل FFmpeg/libx264 يظهر في startup telemetry ويحوّل كتابة الحادثة إلى OpenCV `mp4v`.
- لا توجد عملية upload ضمن هذه النسخة؛ الحادثة المكتملة تبقى في outbox.
- Evaluation Recorder له queue محدودة وعداد dropped frames، ويسجل الفيديو الكامل الخام دون overlay حتى يمكن وسم False Negatives لاحقًا.
- `EXPERIMENTAL + ACTIVE` مرفوض قبل فتح الكاميرا. وضع SHADOW يسجل القرار لكنه لا يسمح بإرسال إنذار خارجي.

## ملاحظة علمية

وسوم التدريب موروثة من الفيديو، ولذلك هي weak labels على مستوى النافذة. LODO وper-dataset ضروريان لكشف domain shortcut، لكن لا يعوضان ground truth زمنيًا. لهذا القرار النهائي هو Fusion متعدد الأدلة، ومقاييس القبول النهائية تحسب على alert/incident في تسجيلات الشركة، لا على window classification وحده.
