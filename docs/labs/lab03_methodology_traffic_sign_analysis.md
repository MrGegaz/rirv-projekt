# Lab 03: Methodology for Traffic Sign Analysis

## 1. Project Goal
Build a robust traffic sign analysis pipeline that works on:
- Daytime driving video
- Early dusk driving video

The final system should detect and classify traffic signs reliably enough for a live defense demo, with measurable accuracy and stable runtime.

## 2. Starting Point from Lab 02
Current research direction already identified:
- Deep-learning-based detection as the main approach
- YOLO family (especially YOLO26/YOLO11) as a practical real-time baseline
- Traditional color/shape cues as optional support techniques
- Relevant external dataset with classes such as speed limits, stop sign, and traffic lights

This is a good foundation. The next step is to move from general research to an implementation-ready experimental methodology.

## 3. Proposed Technical Direction
Primary approach:
- Use a one-stage detector (YOLO26) as the core model
- Keep YOLO11 as a stable fallback baseline
- Optionally evaluate RT-DETR as a transformer-based comparison model
- Fine-tune on data extracted from professor-provided videos
- Optionally pretrain/fine-tune with external traffic-sign data to improve class coverage

Why this approach:
- Good speed/accuracy tradeoff for video
- Mature training/inference tooling
- Strong 2026 ecosystem support and active maintenance
- Easier to explain and demonstrate in a project defense

## 4. Data Strategy
### 4.1 Data Sources
- Source A: professor-provided driving videos (day + dusk) -> mandatory domain data
- Source B: external dataset (Kaggle link from Lab 02) -> optional augmentation for rare classes

### 4.2 Frame Extraction
Extract frames from both videos at controlled FPS (for example 5-10 FPS) to avoid near-duplicate samples.

Example command:
```bash
ffmpeg -i input_video.mp4 -vf fps=7 frames/%06d.jpg
```

### 4.3 Annotation Plan
Use `CVAT` or `Label Studio`.

Label format:
- Bounding boxes
- One class per visible sign

Suggested minimum class set (adapt if needed to actual scene):
- `stop`
- `speed_limit_XX` (or grouped `speed_limit` if class count becomes too sparse)
- `warning`
- `mandatory`
- `traffic_light_red`
- `traffic_light_green`

Important rule:
- Keep class definitions consistent across day and dusk annotations.

### 4.4 Split Strategy (Critical)
Avoid data leakage by splitting by time segment / sequence, not random frame-level only.

Recommended split:
- Train: 70%
- Validation: 15%
- Test: 15%

And keep separate reporting for:
- Day-only test subset
- Dusk-only test subset

## 5. Preprocessing and Augmentation
Focus on robustness to lighting changes and small-object detection.

Recommended augmentations:
- Brightness/contrast jitter
- Hue/saturation jitter (small)
- Gaussian blur / motion blur (mild)
- Random scale and crop
- Mosaic/mixup (YOLO default, controlled)

Dusk-specific robustness:
- Gamma transformations
- Slight noise injection

Input size:
- Start with `imgsz=640`
- Try `imgsz=960` if small distant signs are missed and hardware allows it

## 6. Model Training Plan
### 6.1 Baseline
- Model: `yolo26n` (fast baseline)
- Goal: establish first measurable results quickly

### 6.2 Improved Model
- Model: `yolo26s` (or `yolo11s` if you need more conservative/stable behavior)
- Add tuned augmentations and potentially higher resolution

### 6.3 Comparative Baseline (Optional but Recommended)
- Model: `rtdetr-l` (or another RT-DETR variant available in your environment)
- Goal: strengthen methodology by comparing one YOLO-family detector with one transformer detector

### 6.4 Training Metrics to Track
- mAP@0.50
- mAP@0.50:0.95
- Precision
- Recall
- FPS / latency on inference video

Example training command:
```bash
yolo detect train model=yolo26n.pt data=data.yaml imgsz=640 epochs=100 batch=16
```

## 7. Evaluation Methodology
### 7.1 Quantitative Evaluation
Report global and per-condition metrics:
- All test data
- Day subset
- Dusk subset

Also report per-class AP for the most important classes (`stop`, speed limits).

### 7.2 Qualitative Evaluation
Prepare a short panel of examples:
- Correct detections
- Missed detections
- False positives
- Difficult dusk frames

This is important for defense because it shows scientific analysis, not only one final number.

## 8. Defense-Oriented Demo Pipeline
During defense, show:
1. Input video frame
2. Predicted boxes + class + confidence
3. Running FPS
4. Short commentary on failure cases

Suggested demo script flow:
- 20-30 seconds daytime segment
- 20-30 seconds dusk segment
- Brief comparison baseline vs improved model

## 9. Risk Analysis and Mitigation
Main risks and actions:
- Too few labeled samples
  - Mitigation: prioritize high-traffic segments, add targeted external data
- Dusk performance drop
  - Mitigation: condition-aware augmentation + dusk-focused validation
- Small signs missed at distance
  - Mitigation: higher input resolution and additional close/far balance in training set
- Class imbalance (many speed limits, few stop signs)
  - Mitigation: reweighting/oversampling and class merging if necessary

## 10. Minimal Deliverables for This Lab Stage
- Defined class taxonomy
- Labeled subset from both videos
- Baseline training run with metrics
- Error analysis table (day vs dusk)
- Improvement iteration plan

## 11. Practical Tool Stack
- Annotation: `CVAT` or `Label Studio`
- Training/Inference: `Ultralytics YOLO26/YOLO11` (with optional `RT-DETR` comparison)
- Data handling: Python + OpenCV + ffmpeg
- Experiment tracking (lightweight): CSV/Markdown logs per run

## 12. Immediate Next Steps
1. Extract frames from both videos with fixed FPS.
2. Freeze final class list based on visible signs in your videos.
3. Annotate first balanced subset (day + dusk).
4. Train baseline `yolo26n` and record metrics.
5. Run error analysis and select one focused improvement for next iteration.

## Open Questions (for alignment before implementation)
- Should traffic lights be part of the final scope, or only static traffic signs?
- Do we keep separate speed-limit classes (10, 20, 30...) or merge into one `speed_limit` first?
- What hardware will be used during defense (GPU/no GPU), since this affects model size choice?
