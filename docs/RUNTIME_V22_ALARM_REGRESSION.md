# Runtime V2.2 + Alarm Regression

## مصدر المقارنة

- الفيديو: `evaluation_sessions/raspberry-live-8dce40a9/session.mp4`
- Replay النهائي: `evaluation_sessions/raspberry-replay-67eb34d6/`
- الإصدارات: `fusion-2.2.0`, `eye-events-v2.2`, contract `2.2.0`
- الصوت أثناء Replay: `LOG_ONLY`؛ لم يشغل جهاز الصوت.
- Ground truth تقريبي من وصف صاحب الفيديو، لأن `annotations.csv` الأصلي فارغ.

اعتبرت الفترات `142–145`, `155–170`, و`220–255` ثوانٍ موجبة للنعاس أو
الإغلاق، وبقية الزمن بعد المعايرة سالبًا. أخذت عينة كل 0.5 ثانية، واستبعدت
UNKNOWN السالب من مقام FPR.

## النتيجة

| المقياس | Runtime السابق | Runtime V2.2 |
|---|---:|---:|
| Precision | 55.7% | 81.6% |
| Recall | 78.3% | 79.2% |
| F1 | 65.1% | 80.4% |
| FPR | 17.1% | 4.9% |
| FP time | 33.0s | 9.5s |
| False standalone Critical | موجود | صفر |

Confusion time للنسخة النهائية:

```text
TP=42.0s  FP=9.5s  TN=185.5s  FN=11.0s  UNKNOWN_AWAKE=8.0s
```

- لا يوجد Drowsy/Critical عند حركة العين والنظر للأسفل `120–135s`.
- التثاؤبان اكتشفا عند `184.33s` و`193.47s`، ولم ينتجا Drowsy أو صوتًا.
- لا يوجد Drowsy من PERCLOS القديم بعد فتح العين `256.73–268.53s`.
- اكتشفت الفترات الموجبة الثلاث كلها، لكن تأخير البداية كان `2.07s`, `3.00s`,
  و`3.27s`؛ median `3.00s`.
- FP المتبقي هو في معظمه ذيل حالة Critical أثناء تأكيد فتح العين. الصوت نفسه
  يتوقف بعد فتح موثوق 0.5 ثانية، بينما تبقى الحالة الداخلية في recovery حتى 3 ثوانٍ.

## Alarm commands في Replay

- كل انتقال خطر أنتج onset واحدًا، وليس أمرًا لكل فريم.
- Drowsy الحقيقية بدأت أمرًا عند `223.27s` وreminder عند `233.27s`.
- Critical عند `240.47s` أنتج onset ثم reminders عند `244.47`, `248.47`, و`252.47`.
- كل الأوامر سجلت مع `alarm_suppressed=LOG_ONLY_OR_SOURCE_BLOCKED`، ولم يصدر صوت.

## قرار البوابة

- PASS: Precision، F1، FPR، منع False Critical المستقل، وعدم إعادة استخدام PERCLOS.
- PASS: Episode recall التقريبي `3/3` والتثاؤب `2/2`.
- FAIL: Timeline Recall `79.2% < 90%` وmedian delay `3.0s > 2s`.

يبقى النظام `SHADOW / CONTROLLED DEMO`. لا نخفض شرط الإغلاق القوي 1.5 ثانية
لرفع Recall شكليًا؛ القياس النهائي يحتاج annotations فريمية أدق وجلسات قبول جديدة.
