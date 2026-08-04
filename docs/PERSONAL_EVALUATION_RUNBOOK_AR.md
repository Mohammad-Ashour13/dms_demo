# دليل تركيب الموديل وتجربة الكاميرا الشخصية

## ما تم تركيبه

الموديل الفعّال هو `20260731T140605Z_full_runtimefix1` داخل:

```text
dms_final_system/models/drowsiness/20260731T140605Z_full_runtimefix1/
```

ملف `active_model.json` يشير إليه بوضع `SHADOW`. الحالة `EXPERIMENTAL`، لذلك يمنع النظام تشغيله بوضع `ACTIVE`. ملف MediaPipe موجود في `models/mediapipe/face_landmarker.task` داخل النظام.

## تجهيز اللابتوب

من الفولدر الأب لـ`dms_final_system`:

```bash
python3 -m venv .venv-dms
source .venv-dms/bin/activate
pip install -r dms_final_system/requirements-laptop.txt
PYTHONPATH=. pytest dms_final_system/tests
```

افحص الموديل دون تشغيل الكاميرا:

```bash
PYTHONPATH=. python3 -m dms_final_system.runtime.app \
  --verify-bundle dms_final_system/models/drowsiness/20260731T140605Z_full
```

وللنسخة الفعالة مع drift reference:

```bash
PYTHONPATH=. python3 -m dms_final_system.runtime.app \
  --verify-bundle dms_final_system/models/drowsiness/20260731T140605Z_full_runtimefix1
```

يجب أن تكون Golden parity أقل من `1e-6`. لا ترفع tolerance عند الفشل.

## جلسة كاميرا كاملة

إعداد `runtime.example.json` جاهز للتقييم: `SHADOW + DEBUG + full-session recording`. شغّل:

```bash
PYTHONPATH=. python3 -m dms_final_system.runtime.app \
  --config dms_final_system/configs/runtime.example.json
```

أوقف الجلسة بـ`Ctrl+C`. النتيجة تظهر في:

```text
dms_final_system/evaluation_sessions/<session_id>/
```

راجع أن `session.mp4` و`frame_timestamps.csv` و`telemetry.jsonl` و`session_manifest.json` موجودة. وضع SHADOW يحفظ القرارات والحوادث، لكنه لا يرسل إنذارًا خارجيًا. `FATIGUE_WARNING` تنبيه مبكر فقط؛ التصنيف الموجب النهائي هو `DROWSY` أو `CRITICAL`.

## وسم الفيديو بعد التجربة

شاهد الفيديو الخام دون احتمالات الموديل:

```bash
PYTHONPATH=. python3 -m dms_final_system.runtime.evaluation.annotate \
  dms_final_system/evaluation_sessions/<session_id>
```

المفاتيح:

- `Space`: تشغيل/إيقاف.
- `a`: بداية/نهاية `AWAKE`.
- `d`: بداية/نهاية `DROWSY_SIMULATED`.
- `y`: بداية/نهاية `YAWN`.
- `b`: بداية/نهاية `BLINK`.
- `c`: بداية/نهاية `PROLONGED_EYE_CLOSURE`.
- `x`: بداية/نهاية `DISTRACTION`.
- `u`: بداية/نهاية `UNSCORABLE`.
- `j` و`l`: رجوع/تقديم ثانية.
- `q`: حفظ وخروج.

يمكن أن يتداخل YAWN أو closure مع AWAKE/DROWSY. يجب تغطية كل زمن قابل للتقييم بوسم AWAKE أو DROWSY، ووضع UNSCORABLE عندما يكون الوجه غير واضح. لا يجوز تداخل AWAKE/DISTRACTION مع DROWSY/closure.

## حساب الصح والغلط

بعد حفظ `annotations.csv`:

```bash
PYTHONPATH=. python3 -m dms_final_system.runtime.evaluation.evaluate \
  dms_final_system/evaluation_sessions/<session_id>
```

المخرجات:

```text
evaluation_report.md
evaluation_metrics.json
evaluation_timeline.csv
evaluation_errors.csv
episode_matches.csv
event_metrics.csv
```

`evaluation_errors.csv` يعطي توقيت كل FP/FN وأسباب Fusion. التقرير يحسب Timeline confusion matrix كل نصف ثانية، وEpisode Precision/Recall/F1، وFPR، والإنذارات الكاذبة في الساعة، والتأخير، وYawn/closure بصورة مستقلة.

## بروتوكول التجربة

1. فحص تقني 10–15 دقيقة: معايرة، Awake، رمش، تثاؤب، closure، تمثيل نعاس، كلام ونظر للأسفل وفقد وجه. لا تعتمد أرقامه علميًا.
2. جلسة Tuning مدتها 15–20 دقيقة. إذا لزم، عدّل Fusion فقط ثم جمّد config.
3. ثلاث جلسات Acceptance مستقلة بإجمالي 45–60 دقيقة، مع 30 دقيقة Awake و30 blink و20 حالة نعاس و10 yawns و10 closures على الأقل.

بعد وسم الجلسات الثلاث شغّل التقرير المجمع:

```bash
PYTHONPATH=. python3 -m dms_final_system.runtime.evaluation.evaluate \
  dms_final_system/evaluation_sessions/<session_1> \
  dms_final_system/evaluation_sessions/<session_2> \
  dms_final_system/evaluation_sessions/<session_3> \
  --aggregate-output dms_final_system/evaluation_sessions/personal_acceptance
```

النجاح يحتاج Episode Precision≥75% وRecall≥90% وF1≥80% وFPR<10% وmedian delay≤2s، وصفر False CRITICAL في Awake، وAwake NORMAL≥90%، وBlink recall≥85%، وClosure recall=100%، ونجاح المعايرة و90% valid eye observations/windows. النتيجة القصوى هي `PERSONAL_PILOT_PASS`، وليست Production Ready.

## Raspberry Pi

انقل فولدر `dms_final_system` نفسه دون تغيير الـbundle أو Fusion config، وثبت `requirements-raspberry.txt`. أعد bundle verification ثم جلسة مختصرة، وبعدها soak لمدة 30 دقيقة. راقب FPS وinference/Fusion latency وحرارة/RAM وعدادات dropped frames من `Health` في telemetry.
