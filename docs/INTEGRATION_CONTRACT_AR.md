# عقد التكامل مع Detectors والرفع

## Deployment mode

في `SHADOW` يسمح بتسجيل decisions وIncident evidence فقط، ولا يستدعي أي actuator أو uploader خارجي. لا يجوز لـ`CompanyUploadAdapter` أو detector تجاوز هذه السياسة. الـRuntime يرفض تشغيل bundle حالته `EXPERIMENTAL` إذا كان الوضع `ACTIVE`.

## إضافة الهاتف أو التدخين أو الطعام أو الالتفات

ينشر الـdetector كائن `EvidenceEvent` إلى `EvidenceBus`:

```python
EvidenceEvent(
    kind="PHONE_USE",
    monotonic_sec=now,
    confidence=0.93,
    ttl_sec=2.0,
    source="phone_detector_v1",
    details={"box": [x1, y1, x2, y2]},
)
```

القيم المعروفة للـviolations هي `PHONE_USE`, `SMOKING`, `EATING`, `DISTRACTION`. انتهاء TTL يزيل الدليل تلقائيًا. لا يحق للـdetector تغيير `driver_state` بنفسه.

## رفع الحوادث

طبّق `CompanyUploadAdapter.publish()` خارج مسار AI. مدخلاته Incident مكتملة تحتوي:

- `video.mp4`
- `incident.json`
- `telemetry.jsonl`

يجب أن يكون الرفع idempotent باستخدام `incident_id`، وألا يحذف الفولدر إلا بعد acknowledgment لكل الملفات. أي retry أو authentication أو API endpoint هو مسؤولية Adapter وليس Recorder أو Fusion.
