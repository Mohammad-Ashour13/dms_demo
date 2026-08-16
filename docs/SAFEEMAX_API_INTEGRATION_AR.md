# تكامل DMS مع Safeemax Device API

## ما يدعمه الـ API الحالي

بحسب `http://76.13.131.115:4000/openapi.json` يدعم السيرفر استقبال التنبيهات عبر:

```text
POST /api/events
```

ويتعامل مع `eventId` كمفتاح idempotency؛ إعادة إرسال الحدث نفسه مقبولة ولا تنشئ
تنبيهاً مكرراً. أما `GET /api/devices` فهو مسار قراءة فقط. لا يوجد في مواصفات
OpenAPI الحالية مسار `POST` أو `PATCH` لتحديث حالة الجهاز أو heartbeat، لذلك لا
يرسل الـ runtime حالة جهاز إلى endpoint مخمّن. حالة الـ runtime التفصيلية تبقى
متاحة في الـ dashboard المحلي وفي telemetry إلى أن يضيف فريق الـ backend عقداً
موثقاً لحالة الجهاز.

## الأحداث المرسلة

يرسل النظام حدثاً مرة واحدة عند دخول كل حالة أو ظهور كل مخالفة جديدة:

- الحالات: `FATIGUE_WARNING`, `DROWSY`, `CRITICAL`.
- المخالفات: `PHONE_USE`, `SMOKING`, `EATING`, `SEATBELT_MISSING`.

إذا اختفت المخالفة ثم ظهرت من جديد، تُرسل كحدث جديد مع `eventId` جديد. انتقال
الحالة من `DROWSY` إلى `CRITICAL` يولّد حدثاً جديداً أيضاً.

خريطة الشدة الحالية:

| alarmType | severity |
|---|---|
| `CRITICAL` | `critical` |
| `DROWSY`, `PHONE_USE`, `SMOKING`, `SEATBELT_MISSING` | `high` |
| `FATIGUE_WARNING`, `EATING` | `medium` |

## الاعتمادية وعدم تعطيل AI

قبل أي اتصال شبكي يُكتب الـ payload ذرياً داخل `incidents/api_outbox`. يعمل
الإرسال في thread مستقل، ويعيد المحاولة بتأخير exponential حتى حد 30 ثانية.
استجابات `200` و`201` تحذف الملف من outbox. أخطاء `4xx` الدائمة تُنقل إلى
`incidents/api_outbox/rejected` للفحص بدلاً من إعادة إرسالها بلا نهاية. انقطاع
الكهرباء أو إعادة تشغيل البرنامج لا يفقد الأحداث المعلقة.

## التفعيل

عدّل قسم `safeemax_api` في ملف إعدادات الجهاز:

```json
"safeemax_api": {
  "enabled": true,
  "base_url": "http://76.13.131.115:4000",
  "device_id": "DEVICE_ID_FROM_BACKEND",
  "vehicle": "VEHICLE_NAME_OR_ID",
  "driver": null,
  "location": "",
  "battery": null,
  "outbox_dir": "incidents/api_outbox",
  "request_timeout_sec": 5.0,
  "retry_initial_sec": 1.0,
  "retry_max_sec": 30.0,
  "queue_size": 256,
  "event_states": ["FATIGUE_WARNING", "DROWSY", "CRITICAL"],
  "event_violations": ["PHONE_USE", "SMOKING", "EATING", "SEATBELT_MISSING"]
}
```

إرسال بيانات الجهاز مستقل عن وضع نشر النموذج. يمكن إبقاء النموذج في `SHADOW`
مع تفعيل `safeemax_api.enabled`، بينما يبقى تشغيل replay ممنوعاً من الإرسال حتى
لا تُنشأ بيانات تجريبية في واجهة الشركة.

لم يتم وضع `device_id` أو `vehicle` افتراضيين لأن مواصفات الـ API لا تعطي قيماً
صحيحة لهما؛ يجب الحصول عليهما من فريق الـ backend. بعد التفعيل شغّل:

```bash
dms-runtime --config configs/runtime.raspberry_pi5.json
```

راقب سجلات telemetry للأحداث `event_queued`, `event_delivered`, `event_retry`,
و`event_rejected`. ويظهر ملخص الاتصال أيضاً تحت `remote_api` في حالة الـ
dashboard المحلي.

## المطلوب من فريق الـ backend لإرسال حالة الجهاز

إضافة endpoint موثّق، مثلاً `POST /api/devices/{deviceId}/heartbeat`، مع تحديد:

- الحقول المطلوبة: وقت القياس، online status، نسخة النظام، الحرارة، CPU وRAM.
- سياسة idempotency أو رقم sequence.
- مهلة اعتبار الجهاز offline.
- التوثيق authentication إن كان مطلوباً.

بعد توفر هذا العقد يمكن تغذيته مباشرة من `SystemMetricsSampler` بنفس عامل
الإرسال غير الحاجب المستخدم للتنبيهات.
