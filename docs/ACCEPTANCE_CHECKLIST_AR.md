# بوابات القبول

## Offline model

- [ ] لا group overlap وtest لم يدخل tuning.
- [ ] Precision ≥ 0.75 وRecall ≥ 0.90 وF1 ≥ 0.80 وwindow FPR < 0.10.
- [ ] مراجعة per-dataset وLODO وعدم انهيار dataset إلى PR-AUC عشوائي.
- [ ] تفاوت seeds مقبول ومفسّر.
- [ ] bundle checksum وgolden parity ≤ 1e-6.

## Replay على تسجيلات الشركة

- [ ] التقسيم حسب driver إلى tuning وacceptance.
- [ ] blink يحسب مرة واحدة لكل open→closed→open.
- [ ] yawn يحسب مرة واحدة بعد اكتمال الحدث.
- [ ] `UNKNOWN` عند فقد الوجه/فشل المعايرة.
- [ ] alert-level FPR < 10% على acceptance.
- [ ] median alert delay ≤ 2s بالنسبة للـground-truth المتاح.
- [ ] كل انتقال state له reason codes ويمكن تتبعه من frame إلى window إلى incident.

## Personal camera pilot

- [ ] التشغيل `SHADOW` والـbundle التجريبي لا يستطيع التحول إلى `ACTIVE`.
- [ ] ثلاث جلسات مستقلة وإجمالي التسجيل 45 دقيقة على الأقل.
- [ ] Awake موسوم لمدة 30 دقيقة على الأقل.
- [ ] 20 drowsiness episodes و10 yawns و10 prolonged closures على الأقل.
- [ ] 30 blink موسومة على الأقل وBlink recall ≥ 85% دون duplicate events.
- [ ] الفيديو الكامل محفوظ؛ لا يعتمد التقييم على Incident clips فقط.
- [ ] Episode Precision ≥ 0.75 وRecall ≥ 0.90 وF1 ≥ 0.80.
- [ ] Timeline FPR < 0.10 وmedian detection delay ≤ 2s.
- [ ] صفر False CRITICAL خلال Awake وRecall كل جلسة ≥ 0.80.
- [ ] Prolonged closure recall = 100% وAwake NORMAL ratio ≥ 90%.
- [ ] calibration READY وvalid eye observations وfeature windows ≥ 90%.
- [ ] النتيجة تسمى `PERSONAL_PILOT_PASS` ولا تقدم كإثبات Production.

## Raspberry

- [ ] 640×480 وقريب من 15 FPS.
- [ ] inference p95 < 20ms وFusion p95 < 5ms.
- [ ] decision cycle لا تتأخر أكثر من 0.7s.
- [ ] recording dropped frames < 1%.
- [ ] pre/post ضمن ±1s والفيديو قابل للفتح.
- [ ] FFmpeg probe ظاهر أو fallback مسجل بوضوح.
- [ ] RAM مستقرة وحرارة مقبولة في soak 30 دقيقة.
- [ ] queues محدودة ولا تتراكم.

إذا فشلت بوابة offline يصنف الموديل `EXPERIMENTAL`. إذا نجح offline وفشل replay/Raspberry يبقى `CANDIDATE_PENDING_RUNTIME_ACCEPTANCE` ولا يُستخدم كقرار سلامة production.
