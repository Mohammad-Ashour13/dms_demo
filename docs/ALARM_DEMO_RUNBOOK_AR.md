# دليل تشغيل إنذار DMS

لتوصيل واختبار بزر Raspberry على GPIO18 راجع
[`GPIO_BUZZER_WIRING_AR.md`](GPIO_BUZZER_WIRING_AR.md). بقية هذا الدليل مخصصة
لخرج سماعات اللابتوب.

هذه النسخة مخصصة لـControlled Demo. يبقى الموديل `EXPERIMENTAL` ووضع النشر
`SHADOW`؛ المسموح فقط هو صوت محلي وشاشة محلية، ولا يفعّل uploader أو أي actuator
خارجي.

## 1. التحضير

من جذر المشروع:

```bash
source .venv-dms/bin/activate
python -m pytest -q dms_final_system/tests
```

استخدم سماعات اللابتوب الافتراضية، واضبط مستوى النظام منخفضًا أول مرة. يفحص النظام
`aplay` ثم `paplay` ثم `pw-play`. لا توجد مكتبة صوت Python إضافية.

اختبار الأصوات يدويًا، ولا يعمل إلا عند تنفيذ الأمر صراحة:

```bash
PYTHONPATH=. python -m dms_final_system.runtime.alarm.test_output --pattern DROWSY --gain 0.30
PYTHONPATH=. python -m dms_final_system.runtime.alarm.test_output --pattern CRITICAL --gain 0.30
```

إذا لم تسمع الصوت، افحص جهاز Linux الافتراضي بـ`aplay -l`. يمكن اختيار جهاز ALSA
غير افتراضي من `alarm.device` واختيار `alarm.backend="aplay"`.

## 2. تشغيل الكاميرا مع الصوت والشاشة

```bash
PYTHONPATH=. python -m dms_final_system.runtime.app \
  --config dms_final_system/configs/runtime.laptop_alarm_demo.json
```

عند البداية يصدر chirp قصير يؤكد أن خرج الصوت يعمل، ثم تبدأ المعايرة. لا يصدر
إنذار Drowsy أو Critical أثناء المعايرة.

- Drowsy: نبضة واحدة، ثم reminder كل 10 ثوانٍ.
- Critical: نبضتان مختلفتا التردد، ثم reminder كل 4 ثوانٍ.
- فتح العين موثوقًا 0.5 ثانية يوقف الصوت؛ تبقى الشاشة في recovery حتى تنهي Fusion فترة التأكيد.
- Warning والتثاؤب يسجلان دون صوت.

الفيديو المسجل خام ولا يحتوي الرسم أو مسارًا صوتيًا. كل تشغيل صوتي موثق في
`telemetry.jsonl` ومرتبط بـ`alarm_id`, `episode_id`, و`incident_id`.

## 3. Replay وتقييم صامت

لا تستخدم إعداد Demo لقياس دقة مصنف مفتوح الحلقة؛ الصوت يغير تصرف السائق بعد
الإنذار. استخدم الإعداد الصامت:

```bash
PYTHONPATH=. python -m dms_final_system.runtime.app \
  --config dms_final_system/configs/runtime.replay_silent.json \
  --replay /absolute/path/to/session.mp4
```

حتى لو تم تمرير إعداد محلي بالخطأ، Replay لا يشغل الصوت ما دام
`allow_replay_audio=false`.

لرؤية دورة القرار كاملة:

```bash
rg '"stage":"Alarm"' dms_final_system/evaluation_sessions/<session_id>/telemetry.jsonl
```

الأحداث المهمة هي `alarm_command_issued`, `audio_play_started`,
`audio_play_completed`, `alarm_acknowledged`, `alarm_suppressed`، و
`alarm_output_failed`.

## 4. سيناريو الاختبار الحي

1. Awake وحركة عين ونظر للأسفل لمدة دقيقتين: يجب ألا يصدر صوت.
2. عيون شبه مغلقة: Drowsy واحدة، دون تكرار كل فريم.
3. استمرار Drowsy 10 ثوانٍ: reminder واحد.
4. إغلاق قوي 1.5 ثانية: نمط Critical.
5. فتح موثوق: توقف الصوت خلال 0.7 ثانية.
6. إغلاق جديد: Critical جديدة فورًا.
7. تثاؤب: يظهر ويسجل دون صوت.

أوقف التشغيل بـ`Ctrl+C` وتحقق من الفيديو والـtelemetry والـIncident JSON. لا تحول
النظام إلى `ACTIVE` قبل اجتياز جلسات القبول الموصوفة في وثائق التقييم.
