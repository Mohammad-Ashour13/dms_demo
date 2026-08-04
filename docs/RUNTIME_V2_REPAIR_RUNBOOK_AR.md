# دليل تشغيل واختبار إصلاح Runtime V2

## ما تغير

- بقي `ear/relative_ear` القديم للموديل فقط، وأضيف Event EAR مستقل للرمش والإغلاق.
- المعايرة تستخدم median للموديل وbaseline منفصلًا لكل عين بعد استبعاد القيم الشاذة.
- الرمش بفريم مغلق واحد يمكن التقاطه عند 15 FPS، وPERCLOS يحسب بالزمن الموثوق.
- الموديل التجريبي منفردًا لا ينتج DROWSY أو CRITICAL.
- `CRITICAL` لا ينتج إلا عن إغلاق عين موثوق لمدة 1.5 ثانية.
- اختلاف العينين لا يرفض الإشارة كاملة؛ Event Engine يزامن إغلاق العينين ضمن 0.15 ثانية ويمنع إغلاق عين واحدة من إنتاج Blink أو Critical.
- جودة العين أصبحت `VALID/GRACE/LOST`: الفجوات حتى 0.30 ثانية لا تغيّر الحالة، ولا يدخل النظام `UNKNOWN` إلا بعد فقد مستمر 0.50 ثانية.
- النظر خارج المجال الموثوق `±0.20` لا يبدأ إغلاقًا جديدًا، لمنع انخفاض EAR الكاذب أثناء النظر الجانبي.
- التثاؤب وExcessive Blinking مخالفات تسجل Incident، لكنها لا تغيّر Driver State دون دليل إضافي.
- بعد فتح العين يخرج `CRITICAL` خلال 3 ثوانٍ؛ وإذا بقي PERCLOS مرتفعًا يظهر `POST_CRITICAL_FATIGUE` مع `FATIGUE_WARNING` بدل Drowsy ناتج عن الإغلاق نفسه.
- Bundle الفعال يراقب Domain Shift ويعطل مساهمة الموديل مؤقتًا عند `MODEL_INPUT_OOD`.

## فحص ما قبل الكاميرا

من الفولدر الأب لـ`dms_final_system`:

```bash
source .venv-dms/bin/activate
PYTHONPATH=. pytest -q dms_final_system/tests
PYTHONPATH=. python3 -m dms_final_system.runtime.app \
  --verify-bundle dms_final_system/models/drowsiness/20260731T140605Z_full_runtimefix1
```

يجب أن يظهر `feature_reference_available=True` وأن يكون خطأ Golden Samples أقل من `1e-6`.

## المعايرة الصحيحة

ضع الكاميرا بمستوى العين على `640×480 @ 15 FPS`. خلال أول 10 ثوانٍ انظر للأمام بعين طبيعية ولا تفتحها بصورة مبالغ فيها. الرمش الطبيعي مسموح لأنه يستبعد من Event baseline. إذا لم تكن الإشارة مستقرة تمتد المعايرة إلى 15 ثانية، ثم تفشل إلى `UNKNOWN` مع `failure_reason` واضح.

راقب في Live Status:

- `eye`: حالة العين الفسيولوجية الحالية.
- `event_ear`: النسبة الشخصية؛ الطبيعي قريب من 1.
- `closure`: مدة الإغلاق الحالي.
- `perclos`: يظهر `N/A` حتى تكتمل 10 ثوانٍ موثوقة.
- `model_on`: هل يسمح للموديل بالمساهمة.
- `drift`: `IN_DISTRIBUTION`, `DRIFT_PENDING` أو `MODEL_INPUT_OOD`.
- `state` مقابل `target`: الحالة الفعلية والهدف الحالي أثناء persistence/recovery.
- `eye_obs`: `VALID`, `GRACE` أو `LOST`.
- `recovery`: `NONE`, `CONFIRMING_EYE_OPEN` أو `POST_CRITICAL_FATIGUE`.
- `model_pending`: احتمال مرتفع من الموديل ما زال ينتظر دليلًا فسيولوجيًا.

في JSONL راجع مراحل `Perception`, `Calibration`, `Events`, `Model/feature_drift` و`Fusion`. قرار Fusion يحتوي أيضًا `eye_observation_status`, `recovery_status`, `model_risk_pending`, `state_entry_reason`, `current_reason_codes`, `state_age_sec` و`candidate_age_sec`.

## Pilot ثم Acceptance

1. Pilot واحد: 10 دقائق Awake، 30 رمشة، 10 تثاؤبات، 10 إغلاقات طويلة و20 فترة نعاس تمثيلي.
2. أوقف الجلسة ووسم الفيديو الخام باستخدام `a/d/b/y/c/x/u` دون مشاهدة telemetry.
3. شغّل evaluator واقرأ FP/FN وتغطية Eye signal.
4. يسمح بتعديل thresholds في `configs/runtime.example.json` على Pilot فقط. لا تعدّل LightGBM أو calibration coefficients.
5. جمّد config وشغّل ثلاث جلسات Acceptance مستقلة، ثم التقرير المجمع.

الفشل في أي بوابة يبقي النظام `SHADOW`. نجاحها يعطي `READY_FOR_CONTROLLED_DEMO` فقط.

## Rollback

`active_model.previous.json` يشير إلى Bundle الأصلي. للعودة إليه استخدم أداة activation مع `20260731T140605Z_full` و`SHADOW`. لا تحذف النسخة الأصلية أو المشتقة.
