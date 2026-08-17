# دليل تشغيل Safee على Raspberry Pi 5

الهدف الإنتاجي هو Raspberry Pi OS Bookworm 64-bit Lite مع كاميرا CSI عبر Picamera2، وليس USB/OpenCV. المرجع الذي يحتوي checklist ونتائج القياس ومخاطر الطاقة هو [`PI5_OPTIMIZATION_DASHBOARD_PLAN.md`](PI5_OPTIMIZATION_DASHBOARD_PLAN.md).

إذا كان النظام Trixie وبيئة الذكاء الاصطناعي تستخدم إصدار Python مختلفًا عن
`/usr/bin/python3`، لا تحاول بناء `libcamera` على الجهاز. الإعداد
`picamera2_auto` يفحص التوافق تلقائيًا: يستخدم Picamera2 داخل العملية عندما يكون
الـABI متطابقًا، وإلا يشغّل عامل كاميرا صغيرًا بواسطة Python النظام ويرسل أحدث
إطار BGR فقط إلى بيئة الذكاء الاصطناعي عبر قناة محدودة. لا يتطلب هذا العامل
بيئة جديدة أو تثبيت `rpi-libcamera` عبر pip.

## التثبيت

ضع المشروع افتراضيًا في `/opt/safee-dms` ثم:

```bash
cd /opt/safee-dms
chmod +x scripts/install_pi5_bookworm.sh scripts/apply_cpu_ceiling.sh scripts/pi5_soak_benchmark.py
./scripts/install_pi5_bookworm.sh /opt/safee-dms
```

السكريبت يثبت `python3-picamera2` من Raspberry Pi OS مع `--no-install-recommends`، ويثبت FFmpeg، ثم ينشئ البيئة باستخدام `--system-site-packages`. لا تثبّت Picamera2 من pip. صورة Pi لا تحتوي PyTorch أو Ultralytics؛ يستخدم كشف السلوك NCNN فقط.

تأكد قبل التشغيل من وجود:

- `models/mediapipe/face_landmarker.task`.
- bundle LightGBM الموثق و`models/drowsiness/active_model.json`.
- تصدير NCNN 640 في المسار الموجود في `configs/runtime.raspberry_pi5.json`، ويحتوي `.param` و`.bin` و`ncnn_export_manifest.json` مع checksums.

## فحص LightGBM

```bash
.venv/bin/dms-runtime \
  --verify-bundle models/drowsiness/20260731T140605Z_full_runtimefix1
```

يجب أن تنجح golden samples ضمن `1e-6`. الـbundle الحالي `EXPERIMENTAL` ولذلك يبقى `SHADOW`؛ الكود يمنع تشغيله في `ACTIVE`.

لتفعيل bundle آخر ذريًا:

```bash
.venv/bin/dms-activate-model models/drowsiness/<run_id> \
  --active-model models/drowsiness/active_model.json \
  --deployment-mode SHADOW
```

## Replay قبل كاميرا CSI

```bash
.venv/bin/dms-runtime \
  --config configs/runtime.replay_silent.json \
  --replay /path/to/company_recording.mp4
```

قسّم التسجيلات حسب السائق، واجعل مجموعة tuning منفصلة عن acceptance. لا تغيّر LightGBM threshold أو ترتيب الميزات لتحسين النتيجة. يجب الحفاظ على event/Fusion/alarm parity وعلى صور الأدلة الخام.

## التشغيل الحي

```bash
.venv/bin/dms-runtime --config configs/runtime.raspberry_pi5.json
```

يمنع preflight التشغيل إذا فشلت الكاميرا أو ملفات/نسخ/checksums الموديلات أو copy-mux الفعلي في FFmpeg. بعد التشغيل افتح من الشبكة الخاصة:

```text
http://<raspberry-pi-address>:8080/
```

اللوحة read-only ولا تحتوي login بناءً على الطلب، لذلك امنع وصول الإنترنت واسمح للمنفذ 8080 من subnet المركبة/الشركة فقط.

راقب خصوصًا:

- `get_throttled` الحالي والتاريخي وسبب Safe Mode.
- Capture/Core AI/YOLO/Recorder/Preview FPS والـdrops والـqueues.
- LightGBM/Fusion p95 وdecision staleness.
- الحرارة والتردد الحالي/الأقصى وCPU/RAM/RSS والمساحة الحرة.
- حالة drift وإصدارات الموديلات.

Safe Mode يدخل فورًا عند undervoltage حالي أو 78°C، ويخفض preview إلى 5 FPS وYOLO إلى 0.75 ثانية فقط. لا يغيّر camera/face/events/LightGBM/Fusion/alarm timing. الخروج يحتاج 60 ثانية دون flag حالي وتحت أو عند 72°C.

## الحوادث والتخزين

كل episode ينتج فيديو ثابتًا 10 ثوانٍ (5 قبل البداية + 5 بعدها). الهاتف والتدخين والأكل وحالات التعب تسجل؛ warning سببه التثاؤب فقط لا يسجل. الإطارات تضغط JPEG مرة واحدة ثم تنسخ إلى MJPEG-in-MP4 دون decode أو H.264.

كل حادثة تكتب أولًا داخل `.partial` مع JSON وtelemetry وSHA256 وfsync ثم يعاد تسمية المجلد ذريًا. النظام يحافظ على 2GiB فارغة بحذف أقدم حادثة مكتملة فقط، ويسجل audit telemetry لكل حذف. لا يوجد upload أو مزامنة إنترنت.

## اختبار الطاقة والأداء

لا تختَر CPU ceiling بالتخمين. اختبر 1.8 و2.0 و2.2 و2.4GHz مع NCNN threads 1/2/3، وسجل النتائج في خطة Pi5. مثال جمع اختبار ساعتين:

```bash
scripts/pi5_soak_benchmark.py \
  --label cpu-1800mhz \
  --cpu-max-mhz 1800 \
  --ncnn-threads 1 \
  --duration-sec 7200 \
  --output benchmarks/pi5/cpu-1800mhz.json
```

اختر أقل ceiling يحقق كل شروط القبول. إذا لم ينجح أي ceiling دون undervoltage حالي، فالمشكلة blocker في مسار 18W/الكابل؛ لا تخفّض خوارزمية drowsiness الأساسية.

## systemd بعد القبول فقط

ملفات الخدمة تفترض المسار `/opt/safee-dms` والمستخدم `safee`:

```bash
sudo install -m 0644 deploy/systemd/safee-dms.service /etc/systemd/system/
sudo install -m 0644 deploy/systemd/safee-dms-cpu-cap.service /etc/systemd/system/
sudo install -m 0644 deploy/systemd/safee-dms-power.example /etc/default/safee-dms-power
sudo systemctl daemon-reload
sudo systemctl enable --now safee-dms-cpu-cap.service safee-dms.service
```

ضع `SAFEE_CPU_MAX_MHZ` فقط بعد نجاح القياس وتسجيل النتيجة. الوحدة تعيد تطبيق ceiling عابرًا عند الإقلاع ولا تعدل firmware أو إعداد overclock دائم.
