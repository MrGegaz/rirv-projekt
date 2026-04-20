# Annotation Guidelines (Traffic Signs Only)

## Scope
Annotate only traffic signs from the project taxonomy:
- `speed_limit_10`
- `speed_limit_20`
- `speed_limit_30`
- `speed_limit_40`
- `speed_limit_50`
- `speed_limit_60`
- `speed_limit_70`
- `speed_limit_80`
- `speed_limit_90`
- `speed_limit_100`
- `speed_limit_110`
- `speed_limit_120`
- `stop`

Do **not** annotate traffic lights.

## Bounding Box Rules
- Draw tight boxes around visible sign boundaries.
- Include the full sign when possible.
- If sign is partly occluded, annotate the visible sign area.
- Ignore signs that are too small to confidently classify.
- If motion blur is extreme and class is unclear, skip annotation.

## Class Assignment Rules
- Use exact speed-limit class (e.g., `speed_limit_50`, not generic `speed_limit`).
- If number is unreadable, do not guess.
- Annotate `stop` only for octagonal STOP signs.

## Consistency Rules
- Apply the same rules for day and dusk videos.
- Keep class naming identical across all tasks.
- Prefer fewer but clean labels over noisy labels.

## Suggested First Annotation Batch
- Start with 300-500 frames total.
- Balance by condition:
  - ~60% day
  - ~40% dusk
- Ensure each speed-limit class has at least a few examples if present.

## Export Format
Export annotations in YOLO format and place them into:
- `data/annotations/yolo/images/{train,val,test}`
- `data/annotations/yolo/labels/{train,val,test}`
