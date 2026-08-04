from __future__ import annotations

from pathlib import Path
import json

import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
VIDEO = ROOT / "dms_final_system/evaluation_sessions/raspberry-live-8dce40a9/session.mp4"


def canvas(title: str, size=(14, 7)):
    fig, ax = plt.subplots(figsize=size)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(title, fontsize=18, fontweight="bold", pad=15)
    return fig, ax


def box(ax, x, y, w, h, text, color="#e8f1fb", edge="#24527a", fontsize=10):
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=color, edgecolor=edge, linewidth=1.7
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, wrap=True)
    return (x, y, w, h)


def arrow(ax, a, b, color="#34495e", style="-|>"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=13,
                                 linewidth=1.5, color=color))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def architecture():
    fig, ax = canvas("SafeDrive AI — System Architecture")
    labels = [
        (0.03, .70, .13, .12, "USB Camera\n15 FPS"),
        (.20, .70, .14, .12, "Capture\nLatest Frame"),
        (.39, .70, .15, .12, "MediaPipe\nFace Landmarks"),
        (.59, .70, .15, .12, "Calibration\n10–15 s"),
        (.79, .70, .17, .12, "Temporal Features\n2 s / 65 inputs"),
        (.79, .43, .17, .12, "LightGBM + Platt\nRisk Probability"),
        (.55, .43, .17, .12, "Event Engine\nBlink/Yawn/PERCLOS"),
        (.31, .43, .17, .12, "Fusion FSM\nState + Reasons"),
        (.07, .43, .17, .12, "Alarm Controller\nAudio + HMI"),
        (.31, .14, .17, .12, "Incident Recorder\nVideo + JSONL"),
        (.55, .14, .17, .12, "Filesystem Outbox\nOffline Evidence"),
        (.07, .14, .17, .12, "YOLO11n\nPhone/Smoking/Eating"),
    ]
    for item in labels: box(ax, *item)
    for a, b in [((.16,.76),(.20,.76)),((.34,.76),(.39,.76)),((.54,.76),(.59,.76)),
                 ((.74,.76),(.79,.76)),((.875,.70),(.875,.55)),((.79,.49),(.72,.49)),
                 ((.55,.49),(.48,.49)),((.31,.49),(.24,.49)),((.395,.43),(.395,.26)),
                 ((.48,.20),(.55,.20))]: arrow(ax,a,b)
    arrow(ax,(.665,.70),(.64,.55))
    arrow(ax,(.24,.20),(.395,.43))
    save(fig, "fig01_system_architecture.png")


def pipeline():
    fig, ax = canvas("Data Engineering V3 — From Frames to Deployable Features")
    xs = [.03,.20,.37,.54,.71,.86]
    texts = ["Raw video-level\nweak labels", "Frame signals\n18 channels", "Cleaning +\nphysical clipping",
             "2 s windows\n0.5 s stride", "390 columns\nengineered", "Selected 65\nfeatures"]
    for x,t in zip(xs,texts): box(ax,x,.55,.12,.17,t,color="#eef7ea",edge="#367c39")
    for x1,x2 in zip(xs[:-1],xs[1:]): arrow(ax,(x1+.12,.635),(x2,.635),color="#367c39")
    ax.text(.5,.29,"Group-safe split by dataset::subject (video fallback for unknown subject)",
            ha="center",fontsize=12,bbox=dict(boxstyle="round",fc="#fff4d6",ec="#aa7a00"))
    ax.text(.5,.16,"Test set never used for feature selection, calibration, or threshold tuning",
            ha="center",fontsize=11,color="#8b1e1e")
    save(fig,"fig02_data_pipeline.png")


def temporal_window():
    fig, ax = canvas("Temporal Window — Duration Is Fixed, Frame Count May Vary")
    rng=np.random.default_rng(4)
    for fps,y,c in [(15,.72,"#1f77b4"),(30,.48,"#2ca02c"),(60,.24,"#d62728")]:
        t=np.arange(0,2,1/fps)
        v=y+.055*np.sin(2*np.pi*t)+rng.normal(0,.006,len(t))
        ax.scatter(.1+.8*t/2,v,s=9,color=c,label=f"Camera {fps} FPS: {len(t)} source frames")
    for i in range(30):
        x=.1+.8*i/29
        ax.plot([x,x],[.10,.86],color="#999",alpha=.09)
    ax.text(.5,.08,"All streams are resampled by timestamp to 30 points over the same physical 2 seconds",
            ha="center",fontsize=11)
    ax.legend(loc="lower left",fontsize=9)
    save(fig,"fig03_temporal_window.png")


def frame_vs_temporal():
    fig, ax = canvas("Why a Single Frame Is Not Enough")
    box(ax,.05,.62,.24,.20,"Single frame\nEye appears closed",color="#fdecec",edge="#b63c3c",fontsize=13)
    box(ax,.38,.67,.24,.10,"Normal blink?",color="#fff4d6",edge="#aa7a00",fontsize=12)
    box(ax,.38,.47,.24,.10,"Prolonged closure?",color="#fff4d6",edge="#aa7a00",fontsize=12)
    box(ax,.70,.58,.25,.20,"Temporal evidence\nDuration + context + quality",color="#e8f4ea",edge="#367c39",fontsize=12)
    arrow(ax,(.29,.72),(.38,.72)); arrow(ax,(.29,.69),(.38,.52));
    arrow(ax,(.62,.72),(.70,.70)); arrow(ax,(.62,.52),(.70,.66))
    ax.text(.5,.24,"The same image can represent a harmless blink, tracking loss, downward gaze, or dangerous closure.",
            ha="center",fontsize=12)
    ax.text(.5,.16,"Decision therefore uses persistence, hysteresis, PERCLOS, model risk, and physiological events.",
            ha="center",fontsize=12,fontweight="bold")
    save(fig,"fig04_frame_vs_temporal.png")


def training_cycle():
    fig, ax = canvas("Scientific Training Protocol and Leakage Control")
    nodes=[(.05,.66,.18,.14,"Train windows\nGroup-safe"),(.30,.66,.18,.14,"Stage A\nK=7/20/40/65"),
           (.55,.66,.18,.14,"Stage B\nOptuna 250 fits"),(.78,.66,.18,.14,"Stage C\n3 seeds + OOF"),
           (.55,.28,.18,.14,"Platt + threshold\nOOF only"),(.30,.28,.18,.14,"Final fit\nfull train"),(.05,.28,.18,.14,"One-time test\nreport only")]
    for n in nodes: box(ax,*n,color="#eef0fb",edge="#3949ab")
    for a,b in [((.23,.73),(.30,.73)),((.48,.73),(.55,.73)),((.73,.73),(.78,.73)),
                ((.87,.66),(.64,.42)),((.55,.35),(.48,.35)),((.30,.35),(.23,.35))]: arrow(ax,a,b,color="#3949ab")
    ax.text(.5,.10,"289 / 289 fits executed — the fixed test never selected features, parameters, calibration, or threshold",
            ha="center",fontsize=11,color="#8b1e1e")
    save(fig,"fig05_training_protocol.png")


def model_compare():
    fig, ax=plt.subplots(figsize=(12,6))
    names=["Raw GRU V2","Raw LSTM","Final LightGBM"]
    precision=[48.51,49.19,51.61]; recall=[96.25,65.55,87.43]; f1=[64.51,54.16,64.91]; fpr=[93.64,62.06,76.20]
    x=np.arange(len(names)); w=.2
    for i,(vals,label,col) in enumerate([(precision,"Precision","#2f5597"),(recall,"Recall","#70ad47"),(f1,"F1","#ffc000"),(fpr,"FPR","#c00000")]):
        ax.bar(x+(i-1.5)*w,vals,w,label=label,color=col)
    ax.set_xticks(x,names); ax.set_ylim(0,105); ax.set_ylabel("Percent")
    ax.set_title("Offline Model Comparison (Different Experimental Protocols)",fontweight="bold")
    ax.legend(ncol=4,loc="upper center"); ax.grid(axis="y",alpha=.2)
    ax.text(.5,-.17,"Values are reported for historical comparison; protocols are not strictly identical.",ha="center",transform=ax.transAxes)
    save(fig,"fig06_model_comparison.png")


def runtime_flow():
    fig,ax=canvas("Runtime Data Flow and Bounded Queues")
    items=[(.03,.68,.15,.13,"Capture thread"),(.23,.68,.15,.13,"AI queue\nlatest only"),(.43,.68,.15,.13,"MediaPipe"),(.63,.68,.15,.13,"Events + Features"),(.82,.68,.15,.13,"Fusion"),
           (.23,.28,.15,.13,"Recorder queue"),(.43,.28,.15,.13,"Raw video"),(.63,.28,.15,.13,"Telemetry queue"),(.82,.28,.15,.13,"JSONL + Health")]
    for n in items: box(ax,*n)
    for a,b in [((.18,.745),(.23,.745)),((.38,.745),(.43,.745)),((.58,.745),(.63,.745)),((.78,.745),(.82,.745)),
                ((.105,.68),(.30,.41)),((.38,.345),(.43,.345)),((.705,.68),(.705,.41)),((.78,.345),(.82,.345))]: arrow(ax,a,b)
    ax.text(.5,.10,"Old AI frames are dropped under load; recording and logging do not block the decision loop.",ha="center",fontsize=11)
    save(fig,"fig07_runtime_flow.png")


def fusion_graph():
    fig,ax=canvas("Fusion — Multiple Evidence, One Explainable Decision")
    sources=[("Calibrated\nmodel risk",.06,.73),("Blink / closure\nPERCLOS",.06,.53),("Yawn / head nod",.06,.33),("YOLO behavior\nevidence",.06,.13)]
    for t,x,y in sources: box(ax,x,y,.20,.12,t,color="#f2f2f2",edge="#666")
    box(ax,.39,.38,.25,.24,"Fusion FSM\nEWMA + persistence\nHysteresis + TTL\nQuality gates",color="#e8f1fb",edge="#24527a",fontsize=12)
    box(ax,.76,.63,.19,.12,"Driver state",color="#e8f4ea",edge="#367c39")
    box(ax,.76,.43,.19,.12,"Violations",color="#fff4d6",edge="#aa7a00")
    box(ax,.76,.23,.19,.12,"Reason codes",color="#fdecec",edge="#b63c3c")
    for _,x,y in sources: arrow(ax,(x+.20,y+.06),(.39,.50))
    for y in [.69,.49,.29]: arrow(ax,(.64,.50),(.76,y))
    save(fig,"fig08_fusion_evidence.png")


def states():
    fig,ax=canvas("Driver-State Finite State Machine")
    coords={"UNKNOWN":(.05,.43),"NORMAL":(.25,.70),"WARNING":(.48,.70),"DROWSY":(.70,.70),"CRITICAL":(.70,.25)}
    colors={"UNKNOWN":"#dddddd","NORMAL":"#dff0d8","WARNING":"#fff2cc","DROWSY":"#f8cbad","CRITICAL":"#f4cccc"}
    for k,(x,y) in coords.items(): box(ax,x,y,.16,.12,k,color=colors[k],edge="#555",fontsize=11)
    for a,b in [("UNKNOWN","NORMAL"),("NORMAL","WARNING"),("WARNING","DROWSY"),("DROWSY","CRITICAL"),("CRITICAL","WARNING"),("WARNING","NORMAL")]:
        xa,ya=coords[a]; xb,yb=coords[b]; arrow(ax,(xa+.08,ya+.06),(xb+.08,yb+.06))
    ax.text(.5,.10,"CRITICAL source: trusted strong bilateral eye closure ≥ 1.5 s. Quality loss routes to UNKNOWN outside Critical.",ha="center",fontsize=10)
    save(fig,"fig09_driver_states.png")


def alarm_state():
    fig,ax=canvas("Alarm Episode Controller")
    nodes=[(.05,.62,.18,.14,"IDLE"),(.30,.62,.18,.14,"ONSET\none pattern"),(.55,.62,.18,.14,"REMINDER\nby timer"),(.78,.62,.18,.14,"ESCALATION\nrecurrence"),(.55,.25,.18,.14,"ACKNOWLEDGED\neye open 0.5 s"),(.30,.25,.18,.14,"COOLDOWN /\nREARM")]
    for n in nodes: box(ax,*n,color="#f5eef8",edge="#6c3483")
    for a,b in [((.23,.69),(.30,.69)),((.48,.69),(.55,.69)),((.73,.69),(.78,.69)),((.86,.62),(.64,.39)),((.55,.32),(.48,.32)),((.39,.39),(.39,.62))]: arrow(ax,a,b,color="#6c3483")
    ax.text(.5,.10,"Fusion state is continuous; audio commands are event-based, timed, preemptible, and never emitted per frame.",ha="center",fontsize=11)
    save(fig,"fig10_alarm_controller.png")


def recorder():
    fig,ax=canvas("Rolling Video Evidence")
    ax.plot([.08,.92],[.55,.55],lw=8,color="#aab7b8")
    ax.plot([.12,.37],[.55,.55],lw=12,color="#5dade2",label="Pre-alert 10 s")
    ax.plot([.37,.68],[.55,.55],lw=12,color="#e74c3c",label="Alert episode")
    ax.plot([.68,.88],[.55,.55],lw=12,color="#58d68d",label="Post-alert 10 s")
    for x,t in [(.12,"buffer"),(.37,"onset"),(.68,"clear"),(.88,"finalize")]:
        ax.plot([x,x],[.45,.66],color="#333"); ax.text(x,.39,t,ha="center")
    ax.legend(loc="upper center",ncol=3)
    ax.text(.5,.20,"Nearby alerts merge (5 s gap); clips are capped at 60 s and linked by incident_id.",ha="center",fontsize=11)
    save(fig,"fig11_rolling_recorder.png")


def offline():
    fig,ax=canvas("Offline-First Operation and Future Synchronization")
    nodes=[(.05,.60,.18,.15,"Local detection\ncontinues offline"),(.30,.60,.18,.15,"Local alarm\nand HMI"),(.55,.60,.18,.15,"Filesystem Outbox\nvideo + JSONL"),(.78,.60,.18,.15,"Future API\nidempotent upload"),(.55,.24,.18,.15,"Retry after\nconnectivity returns"),(.78,.24,.18,.15,"Server +\nDashboard")]
    for n in nodes: box(ax,*n,color="#eaf2f8",edge="#2874a6")
    for a,b in [((.23,.675),(.30,.675)),((.48,.675),(.55,.675)),((.73,.675),(.78,.675)),((.87,.60),(.64,.39)),((.73,.315),(.78,.315))]: arrow(ax,a,b,color="#2874a6")
    ax.text(.5,.10,"Current implementation ends at Filesystem Outbox; server synchronization is a documented integration contract.",ha="center",fontsize=11,color="#8b1e1e")
    save(fig,"fig12_offline_outbox.png")


def deployment():
    fig,ax=canvas("Target Embedded Deployment")
    box(ax,.05,.58,.18,.16,"USB Camera\n640×480 @ 15 FPS",color="#eef7ea",edge="#367c39")
    box(ax,.31,.50,.25,.30,"Raspberry Pi 5\nCPU-only\nMediaPipe + LightGBM\nYOLO process isolation",color="#e8f1fb",edge="#24527a",fontsize=12)
    box(ax,.66,.64,.26,.13,"Speaker / buzzer + HMI",color="#fff4d6",edge="#aa7a00")
    box(ax,.66,.42,.26,.13,"Local incident storage",color="#fff4d6",edge="#aa7a00")
    box(ax,.66,.20,.26,.13,"Future company backend",color="#f2f2f2",edge="#777")
    arrow(ax,(.23,.66),(.31,.66)); arrow(ax,(.56,.68),(.66,.70)); arrow(ax,(.56,.60),(.66,.48)); arrow(ax,(.56,.54),(.66,.26))
    save(fig,"fig13_embedded_deployment.png")


def confusion():
    cm=np.array([[1948,6236],[956,6651]])
    fig,ax=plt.subplots(figsize=(7,6)); im=ax.imshow(cm,cmap="Blues")
    for (i,j),v in np.ndenumerate(cm): ax.text(j,i,f"{v:,}",ha="center",va="center",fontsize=16,color="white" if v>cm.max()/2 else "black")
    ax.set_xticks([0,1],["Predicted awake","Predicted drowsy"]); ax.set_yticks([0,1],["Actual awake","Actual drowsy"])
    ax.set_title("LightGBM Test Confusion Matrix (threshold = 0.317)",fontweight="bold",fontsize=13)
    ax.set_xlabel("Prediction"); ax.set_ylabel("Inherited video-level label"); fig.colorbar(im,ax=ax)
    save(fig,"fig14_confusion_matrix.png")


def runtime_metrics():
    fig,ax=plt.subplots(figsize=(10,5.5)); names=["Precision","Recall","F1","1 − FPR"]
    before=[55.7,78.3,65.1,82.9]; after=[81.6,79.2,80.4,95.1]
    x=np.arange(4); w=.34
    ax.bar(x-w/2,before,w,label="Previous runtime",color="#a5a5a5")
    ax.bar(x+w/2,after,w,label="Runtime V2.2 regression",color="#4472c4")
    ax.set_xticks(x,names); ax.set_ylim(0,105); ax.set_ylabel("Percent"); ax.grid(axis="y",alpha=.2)
    ax.set_title("Runtime Regression on One Manually Described Session",fontweight="bold")
    ax.legend(); ax.text(.5,-.16,"Approximate ground truth; not a production benchmark.",ha="center",transform=ax.transAxes)
    save(fig,"fig15_runtime_regression.png")


def screenshots():
    if not VIDEO.exists():
        return
    cap=cv2.VideoCapture(str(VIDEO))
    for idx,(sec,label) in enumerate([(60,"awake"),(185,"yawn"),(241,"critical")],start=1):
        cap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
        ok,frame=cap.read()
        if not ok: continue
        h,w=frame.shape[:2]
        # Deterministic privacy redaction: blur the central driver-face region only.
        x1,x2=int(.25*w),int(.75*w); y1,y2=int(.08*h),int(.72*h)
        roi=frame[y1:y2,x1:x2]
        if roi.size:
            frame[y1:y2,x1:x2]=cv2.GaussianBlur(roi,(99,99),40)
        cv2.rectangle(frame,(x1,y1),(x2,y2),(230,230,230),2)
        cv2.putText(frame,f"Privacy-redacted evaluation frame — t={sec}s",(15,h-20),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2,cv2.LINE_AA)
        cv2.imwrite(str(OUT/f"fig{15+idx:02d}_session_{label}.png"),frame)
    cap.release()


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    for fn in [architecture,pipeline,temporal_window,frame_vs_temporal,training_cycle,model_compare,
               runtime_flow,fusion_graph,states,alarm_state,recorder,offline,deployment,confusion,runtime_metrics]:
        fn()
    screenshots()
    print(json.dumps({"assets": len(list(OUT.glob('*.png')))}, indent=2))


if __name__ == "__main__":
    main()
