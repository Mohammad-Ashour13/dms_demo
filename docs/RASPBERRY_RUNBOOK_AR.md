# دليل Raspberry Pi 5

## التثبيت

يفضل Raspberry Pi OS 64-bit وبيئة Python افتراضية. من الفولدر الأب لـ`dms_final_system`:

```bash
sudo apt update
sudo apt install -y ffmpeg libgomp1 libopenblas-dev build-essential cmake python3-venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r dms_final_system/requirements-raspberry.txt
PYTHONPATH=. pytest dms_final_system/tests
```

إذا تعذر تثبيت MediaPipe wheel على إصدار Python الموجود، استخدم إصدار Raspberry Pi OS/Python المدعوم بدل تغيير كود العقود أو استبدال landmarks بصمت.

## تركيب الموديل

1. فك `model_bundle.zip` في `dms_final_system/models/drowsiness/<run_id>/` بحيث يوجد `model.txt` مباشرة داخله.
2. ضع `face_landmarker.task` في `models/mediapipe/`.
3. افحص وفعّل ذريًا، مع حفظ `active_model.previous.json` للـrollback:

```bash
PYTHONPATH=. python3 -m dms_final_system.runtime.model.activate \
  dms_final_system/models/drowsiness/<run_id> \
  --active-model dms_final_system/models/drowsiness/active_model.json \
  --deployment-mode SHADOW
```

يجب أن تنجح golden samples ضمن `1e-6`. الفشل يمنع التشغيل ولا يعالج برفع tolerance.
أي bundle حالته `EXPERIMENTAL` ممنوع من `ACTIVE` برمجيًا. يجب أن تتطابق قيمة `deployment_mode` في config مع `active_model.json`.

## Replay قبل الكاميرا

اجعل logging `DEBUG` مؤقتًا في نسخة من config ثم:

```bash
PYTHONPATH=. python3 -m dms_final_system.runtime.app \
  --config dms_final_system/configs/runtime.example.json \
  --replay /path/company_recording.mp4
```

أنشئ ملخصًا:

```bash
PYTHONPATH=. python3 -m dms_final_system.runtime.replay_report \
  dms_final_system/logs/<session>.jsonl
```

قسّم التسجيلات حسب السائق، لا حسب الفيديو: tuning لضبط Fusion وacceptance لا تُفتح نتائجها حتى تجميد config. لا تغيّر model threshold لتجميل acceptance؛ اضبط فقط سياسات Fusion على tuning.

## تشغيل USB Camera

```bash
PYTHONPATH=. python3 -m dms_final_system.runtime.app \
  --config dms_final_system/configs/runtime.example.json
```

راقب Live Status وJSONL. أول 10 ثوانٍ معايرة وقد تمتد إلى 15. إذا فشلت تبقى الحالة `UNKNOWN`.

في Runtime V2 راقب أيضًا `eye/event_ear/closure/perclos/model_on/drift` وفرق `state/target`. يجب أن يظهر `feature_reference_available=True` أثناء فحص Bundle المشتق. `MODEL_INPUT_OOD` يعطل مساهمة LightGBM مؤقتًا لكنه لا يعطل Event Engine.

عند تفعيل `evaluation.enabled` تحفظ الجلسة كاملة في `evaluation_sessions/<session_id>/`، بما فيها الفيديو الخام وtimestamps وtelemetry وملف annotations. انقل الجلسة إلى اللابتوب للوسم والتقييم باتباع `PERSONAL_EVALUATION_RUNBOOK_AR.md`؛ نسخة Raspberry تستخدم OpenCV headless ولا تشغل واجهة الوسم الرسومية.

اختبر طبيعي، رمش، تثاؤب، إغلاق طويل، فقدان الوجه، ثم افتح كل `video.mp4` وتحقق من pre/post. بعد ذلك شغّل 30 دقيقة وراقب `Health`: الحرارة والRAM والـeffective FPS وعدادات dropped frames.

## التشغيل النهائي

انسخ config إلى ملف versioned خاص بالنشر، غيّر logging إلى `NORMAL`، ولا تعدّل المثال. احتفظ بالـbundle السابق وملف config السابق للrollback. لا تمسح `incidents/outbox` قبل أن ينهي uploader المستقبلي الإرسال.
