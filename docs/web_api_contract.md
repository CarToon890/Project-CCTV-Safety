# Web API contract — Upload & Analyze (v1, `schema_version` "1.0")

`schema_version` stays `"1.0"`: round 2 only **adds** fields/options (Stage 1 `mode`, `sample_fps`, dense mode) and existing clients keep working with the defaults.

Demo prototype only: no production, no live CCTV.
Example responses: `tests/fixtures/api/*.json` (they conform exactly to this document).

## 0. Conventions

- All bodies are JSON (`application/json; charset=utf-8`). Numbers are JSON numbers; `float` values are rounded to 4 decimals.
- **Coordinates:** `xyxy` = `[x1, y1, x2, y2]`, `float`, **pixels in the coordinate system of the original source frame** (`source.width` × `source.height`, origin top-left, x right, y down), `0 <= x1 < x2 <= width`, `0 <= y1 < y2 <= height`. Never letterboxed/model-input coordinates, never normalised.
- **Time:** every `*_s` field is seconds (`float`), measured from the start of the video.
- **Images in responses:** `image_jpeg_b64` = standard base64 (RFC 4648, with `=` padding) of a **raw, un-annotated** JPEG of the frame at full source resolution. **No** `data:image/jpeg;base64,` prefix — the client adds it. Boxes are drawn by the browser.
- "Nullable: no" means the key is always present and never `null`. Keys are never omitted.
- Unknown extra request fields are ignored.

### 0.1 Common envelope (analyze responses and health)

| Field | Type | Nullable | Notes |
|---|---|---|---|
| `is_model_output` | bool | no | `true` on both analyze endpoints. `false` on `/api/health` (it carries no model output). |
| `model` | string | health: yes (`null`); analyze: no | `"yolov8n"` \| `"yolov8s"` (Stage 1), `"x3d_s"` (Stage 2) |
| `schema_version` | string | no | This contract's version: `"1.0"` |
| `disclaimer` | string | no | Fixed text below, identical in every response |

Disclaimer (exact string):

```text
ต้นแบบเพื่อการศึกษา (pilot) ฝึกด้วยชุดข้อมูลสาธารณะเท่านั้น ไม่ใช่ระบบใช้งานจริง และยังไม่ได้ตรวจสอบความถูกต้องกับกล้อง CCTV เป้าหมาย / Educational pilot trained on public datasets only. Not a production system and not validated on the target CCTV cameras.
```

Error responses do **not** use the envelope (see section 4).

### 0.2 Uploads (both analyze endpoints)

- `multipart/form-data`; file field name **`file`** (exactly one file).
- Media type is decided by the **file extension** (case-insensitive); the part's `Content-Type` is not trusted.
  - Image: `.jpg .jpeg .png .bmp .webp` (`image/jpeg`, `image/png`, `image/bmp`, `image/webp`)
  - Video: `.mp4 .avi .mov .mkv` (`video/mp4`, `video/x-msvideo`, `video/quicktime`, `video/x-matroska`)
  - Any other extension → `415 unsupported_media_type`.
- Size limit **100 MB** (104 857 600 bytes) per file → above it `413 file_too_large`.
- A file with an allowed extension that OpenCV/Pillow cannot decode (or a video with 0 decodable frames or fps <= 0) → `422 decode_failed`. Codec support depends on the local OpenCV build; mp4/avi/mov are expected to work.

### 0.3 `source` object (Stage 1 and Stage 2)

| Field | Type | Units | Nullable | Notes |
|---|---|---|---|---|
| `width` | int | px | no | Original frame width |
| `height` | int | px | no | Original frame height |
| `fps` | float | frames/s | image: `null` | Video frame rate reported by the decoder |
| `frame_count` | int | frames | image: `null` | Total frames in the video |
| `duration_s` | float | s | image: `null` | `frame_count / fps` |

## 1. `GET /api/health`

Never loads models; only reports state. Always `200` while the server is up.

| Field | Type | Nullable | Notes |
|---|---|---|---|
| envelope | — | — | `is_model_output: false`, `model: null` |
| `status` | string | no | Always `"ok"` |
| `device` | string | no | e.g. `"cuda:0"` or `"cpu"` |
| `models` | object | no | Keys exactly `yolov8n`, `yolov8s`, `x3d_s`; each value `"loaded"` (in memory) \| `"available"` (weights file present, not loaded yet) \| `"missing"` (weights file absent) |

Fixture: `health.json`.

## 2. `POST /api/stage1/analyze` — YOLOv8 detector (image or video)

Form fields:

| Name | Type | Required | Default | Notes |
|---|---|---|---|---|
| `file` | file | yes | — | Image or video (0.2) |
| `model` | string | yes | — | `yolov8n` \| `yolov8s`; else `400 invalid_parameter` |
| `max_frames` | int | no | `16` | `1..60` inclusive; else `400 invalid_parameter`. Validated in `frames` mode, including images, though images do not use it. Ignored (not validated) in `dense` mode |
| `mode` | string | no | `"frames"` | `"frames"` (sampled frames with JPEGs, the v1 behaviour) \| `"dense"` (video only, boxes only, for playback overlay); anything else → `400 invalid_parameter` |
| `sample_fps` | float | no | `10` | Dense mode only: finite number in `1..30` inclusive (e.g. `"10"`, `"12.5"`); else `400 invalid_parameter`. Ignored (not validated) in `frames` mode |

Empty-string form values count as "not sent" and get the default. The `mode` value has surrounding whitespace trimmed before it is checked.

Response `200`:

| Field | Type | Nullable | Notes |
|---|---|---|---|
| envelope | — | — | `is_model_output: true`, `model` = requested model |
| `detector_schema_version` | int | no | `2` (`cctv_safety/schema.py` `DETECTOR_SCHEMA_VERSION`) |
| `class_names` | string[] | no | `["person","helmet","vest","fall","fire","smoke"]`; `class_id` is the index in this list |
| `media_type` | string | no | `"image"` \| `"video"` |
| `source` | object | no | Section 0.3 |
| `thresholds` | object | no | Per-class min confidence used (from `configs/thresholds.yaml`); keys exactly `class_names` |
| `ppe_min_confidence` | float | no | `min(thresholds.person, thresholds.helmet, thresholds.vest)`; passed to `assess_ppe` (same as `scripts/infer.py`) |
| `mode` | string | no | `"frames"` \| `"dense"` (the mode actually used; images are always `"frames"`) |
| `sample_fps` | float | `frames` mode: `null` | Dense mode: the `sample_fps` used (rounded to 4 decimals) |
| `frames` | Frame[] | no | Ordered by `index` ascending; never empty |

Frame selection, `mode = "frames"` (default; unchanged from v1):
- **Image:** exactly one frame: `index: 0`, `time_s: null`.
- **Video:** `n = min(max_frames, frame_count)`; indices = unique values of `round(linspace(0, frame_count-1, n))` (so the first and last frames are always included); `time_s = index / fps`.

Frame selection, `mode = "dense"` (video only):
- An image upload → `422 video_required`. A video that Stage 2 would split into more than 30 windows (section 3, roughly `D >= 61 s`) → `422 video_too_long`.
- Sample times `t_k = k / sample_fps` for `k = 0, 1, …` while `t_k < duration_s` (`duration_s = frame_count / fps`).
- Frame index per sample, exactly (Python, round-half-to-even): `min(int(round(k / sample_fps * fps)), frame_count - 1)`; duplicates are removed (first occurrence kept). So `sample_fps >= fps` yields every frame.
- `time_s = index / fps` (as in frames mode); `image_jpeg_b64` is `null` for every frame (the browser plays the original file and draws boxes over it).
- Example: 8.0 s at 30 fps with `sample_fps = 10` → 80 frames, indices `0, 3, 6, …, 237`, `time_s` `0.0, 0.1, …, 7.9`.

Frame:

| Field | Type | Units | Nullable | Notes |
|---|---|---|---|---|
| `index` | int | frame number | no | 0-based frame number in the source video (`0` for images) |
| `time_s` | float | s | image: `null`; video: no | |
| `image_jpeg_b64` | string | — | `frames` mode: no; `dense` mode: always `null` | Raw frame JPEG, base64, no `data:` prefix (section 0) |
| `detections` | Detection[] | — | no | Only boxes with `confidence >= thresholds[class_name]`; sorted by `confidence` descending; may be `[]` |
| `ppe` | PpeRow[] | — | no | Output of `cctv_safety.ppe.assess_ppe(detections, ppe_min_confidence)` unchanged; may be `[]` |

Detection:

| Field | Type | Notes |
|---|---|---|
| `class_id` | int | `0..5`, `CLASS_TO_ID[class_name]` |
| `class_name` | string | One of `class_names` |
| `confidence` | float | `0..1` |
| `xyxy` | float[4] | Source-frame pixels (section 0) |

PpeRow (exactly the keys produced by `assess_ppe`; these are **derived** events, not model classes):

| Field | Type | Notes |
|---|---|---|
| `person_index` | int | 0-based index among the frame's `person` detections with `confidence >= ppe_min_confidence`, in `detections` order (i.e. the n-th person in the list) |
| `person_confidence` | float | That person's confidence |
| `has_helmet` | bool | A helmet centre lies in the top 0–35 % of the person box |
| `has_vest` | bool | A vest centre lies in the 20–75 % band of the person box |
| `alerts` | string[] | Subset of `["no_helmet","no_vest"]` in that order; `[]` if compliant |
| `method` | string | `"ppe-centre-in-person-region"` |

Fixtures: `stage1_image.json`, `stage1_video.json` (requested with `max_frames=3`), `stage1_video_dense.json` (`mode=dense`, `sample_fps=10`, 1.0 s clip at 10 fps → 10 frames).

## 3. `POST /api/stage2/analyze` — X3D-S fight classifier (video only)

Form fields: `file` only (video). An image extension → `422 video_required`.

Windowing (`D = duration_s`, `W = 2.0`):
- Non-overlapping windows `[k·W, min((k+1)·W, D)]`, `k = 0, 1, …`.
- The final partial window is included only if its length is **>= 1.0 s**; otherwise those trailing frames are dropped.
- If `D < 2.0`, exactly one window `[0, D]` covering the whole clip (any length).
- More than **30** windows → `422 video_too_long` (i.e. roughly `D >= 61 s`).
- Per window: frames `round(start_s·fps) .. min(round(end_s·fps), frame_count) - 1`, of which 13 are taken with `round(linspace(first, last, 13))` (repeats allowed), then the plan's X3D-S preprocessing (letterbox 224, normalise).

Response `200`:

| Field | Type | Units | Nullable | Notes |
|---|---|---|---|---|
| envelope | — | — | — | `is_model_output: true`, `model: "x3d_s"` |
| `source` | object | — | no | Section 0.3 (video, so no nulls) |
| `class_names` | string[] | — | no | `["non_fight","fight"]` (logit index 0/1) |
| `frames_per_window` | int | frames | no | `13` |
| `window_s` | float | s | no | `2.0` (nominal window length) |
| `windows` | Window[] | — | no | Ordered by `start_s`; 1..30 entries |
| `summary` | object | — | no | See below |

Window:

| Field | Type | Units | Notes |
|---|---|---|---|
| `index` | int | — | 0-based window number |
| `start_s` | float | s | `index · 2.0` |
| `end_s` | float | s | `min(start_s + 2.0, duration_s)` |
| `probs` | object | — | `{non_fight: float, fight: float}` = `softmax(logits)`, each `0..1`, sum 1 (± rounding) |
| `label` | string | — | `class_names[argmax(logits)]`; tie → `"non_fight"` |

Summary:

| Field | Type | Notes |
|---|---|---|
| `max_fight_prob` | float | `max(windows[*].probs.fight)` |
| `fight_windows` | int | Count of windows with `label == "fight"` |
| `total_windows` | int | `len(windows)` |

Fixture: `stage2_video.json` (5.4 s clip → windows 0–2, 2–4, 4–5.4).

## 4. Errors

Every non-2xx response has exactly this body (no envelope):

```json
{"error": {"code": "<code>", "message": "<human-readable English text>"}}
```

`code` is from the fixed list below; clients branch on `code`, never on `message`. FastAPI's default validation errors are remapped to this format.

| HTTP | `code` | When |
|---|---|---|
| 400 | `missing_file` | No `file` part, or the file is empty (0 bytes) |
| 400 | `invalid_parameter` | `model` missing/not `yolov8n`/`yolov8s`; `max_frames` not an int in 1..60 (frames mode); `mode` not `frames`/`dense`; `sample_fps` not a number in 1..30 (dense mode) |
| 413 | `file_too_large` | Upload > 100 MB |
| 415 | `unsupported_media_type` | Extension not in section 0.2 |
| 422 | `decode_failed` | Allowed extension but cannot be decoded, 0 frames, or fps <= 0 |
| 422 | `video_required` | Image sent to Stage 2, or to Stage 1 with `mode=dense` |
| 422 | `video_too_long` | Stage 2 video, or Stage 1 `mode=dense` video, would need > 30 windows of 2 s |
| 503 | `weights_missing` | Weights file for the requested model is absent |
| 503 | `model_load_failed` | Weights present but loading or the runtime contract (class names, threshold keys, strict state_dict) failed |
| 404 | `not_found` | Any path under `/api/` that is not an endpoint above |
| 405 | `method_not_allowed` | Known `/api/` path with the wrong HTTP method (`Allow` header lists the allowed ones) |
| 500 | `internal_error` | Unexpected server error. The body never contains a stack trace; the traceback is logged server-side |

Validation order: `missing_file` → `invalid_parameter` → `unsupported_media_type` → `file_too_large` → `video_required` → `weights_missing`/`model_load_failed` → `decode_failed` → `video_too_long`.

Fixtures: `error_stage2_image.json` (422), `error_weights_missing.json` (503).

Paths outside `/api/` are the static UI (`mockup/`); their 404s use the server's default body, not this format.

## 5. Run the web prototype

```bash
# from the repo root; weights in ./weights (or set CCTV_WEIGHTS_DIR)
.venv-cuda/Scripts/python -m uvicorn webapp.api:app --port 8000
```

Open `http://localhost:8000/` — the UI (`mockup/`) and the API (`/api/...`) share one origin.
Numerical note: the server runs X3D-S (and YOLO) in **fp32**, while the Stage 2 notebook evaluated under fp16 autocast, so Stage 2 probabilities can differ slightly from the recorded test metrics in `artifacts/model_handoff/stage2/`. Preprocessing is identical to the notebook.

Playback note: the browser overlay assumes the video's `currentTime` equals `index / fps` as read by OpenCV. Variable-frame-rate files, or MP4s whose start time is not zero, can drift by about one frame. Dense mode on CPU (up to 1,800 YOLO runs for a 60 s clip at 30 samples/s) can be slow, and no progress is shown.

Models load lazily on the first analyze request (`GET /api/health` shows `available` → `loaded`); the GPU (`cuda:0`) is used when available, otherwise the CPU.
