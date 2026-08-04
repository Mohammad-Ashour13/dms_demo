# عقد إعادة تدريب الموديل التالي

هذه المرحلة لا تغيّر موديل الـControlled Demo. تبدأ فقط بعد تجميد Runtime V2 وجمع تسجيلات زمنية موسومة.

## مدخلات التدريب

- تسجيلات الشركة وتسجيلات جديدة من 10 سائقين على الأقل.
- فترات `AWAKE`, `DROWSY`, `BLINK`, `YAWN`, `PROLONGED_EYE_CLOSURE` و`UNSCORABLE` بتوقيت الفيديو.
- تقسيم السائقين قبل بناء النوافذ إلى train/tuning/acceptance، ومنع وجود السائق نفسه في أكثر من قسم.
- Awake الصعب، النظارات، الإضاءة، الكلام والنظر للأسفل تدخل hard negatives.

## عقد الميزات

- استخدام relative left/right Event EAR وديناميكية الانخفاض والتعافي.
- استخدام Blink/long-blink/closure وPERCLOS المحسوبة بالزمن.
- استخدام pitch/yaw/roll كـdelta نسبة إلى baseline الشخصي فقط.
- استخدام motion بوحدات زمنية موحدة وquality/coverage.
- استبعاد absolute EAR/MAR/pose/gaze، metadata وdataset-specific flags من الموديل المرشح.
- عدم تغيير صيغة Runtime بعد بناء CSV؛ parity samples تغطي كل feature.

## البروتوكول

- Grouped folds حسب السائق وLODO حسب مصدر الداتا.
- calibration وthreshold من OOF فقط، والـtest مرة واحدة.
- مقارنة نموذج صغير بميزات نسبية مع LightGBM الحالي، واختيار الأصغر عند فرق PR-AUC أقل من 0.01.
- تصدير `feature_reference.json` تلقائيًا مع q01/q05/median/q95/q99 وgain rank؛ كود Colab الحالي أصبح يدعم ذلك.
- لا يفعّل Bundle جديد قبل Golden parity وReplay وقبول Raspberry.
