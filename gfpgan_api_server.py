# gfpgan_api_server_singleton.py
import os
import time
import uuid
import threading
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import cv2
import numpy as np
import torch
from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from basicsr.utils import imwrite
from gfpgan import GFPGANer

app = FastAPI(
    title="GFPGAN API Server (singleton)",
    description="GFPGAN 모델을 한 번만 로드해 재사용하는 API (CLI 파라미터 1:1 대응, 오프라인 가중치 전용)",
    version="1.3.1",
)

# -------------------------------------------------
# 경로/기본값
# -------------------------------------------------
SHARED_DIR     = Path(os.getenv("SHARED_DIR", "/app/shared_data_workspace"))
W2L_OUT_DIR    = SHARED_DIR / "wav2lip_output_queue"   # 입력 영상/오디오 모두 여기서 기본 선택
GFPGAN_OUT_DIR = SHARED_DIR / "gfpgan_output_queue"    # 결과물
WARMUP_DIR     = SHARED_DIR / "warmup"

for d in [GFPGAN_OUT_DIR, WARMUP_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# inference_gfpgan.py 기본값과 동일
DFLT_VERSION          = "1.3"
DFLT_UPSCALE          = 2
DFLT_BG_UPSAMPLER     = "realesrgan"
DFLT_BG_TILE          = 400
DFLT_ONLY_CENTER_FACE = False
DFLT_ALIGNED          = False
DFLT_WEIGHT           = 0.5
DFLT_EXT              = "auto"
DFLT_FRAMERATE        = 25
DFLT_PIX_FMT          = "yuv420p"
DFLT_CRF              = 18
DFLT_AUDIO_COPY       = True         # 기본: 비디오의 오디오를 그대로 복사
DFLT_AUDIO_BITRATE    = "192k"       # 재인코딩 시 사용

# -------------------------------------------------
# 유틸
# -------------------------------------------------
def _pick_latest(directory: Path, patterns: List[str]) -> Optional[Path]:
    files: List[Path] = []
    for pat in patterns:
        files.extend(directory.glob(pat))
    if not files:
        return None
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0]

def _run_ffmpeg(cmd: list):
    import subprocess
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: rc={proc.returncode}, stderr={proc.stderr[:400]}")

def _safe_rmtree(p: Path):
    import shutil
    try:
        shutil.rmtree(p, ignore_errors=True)
    except Exception:
        pass

def _probe_audio_codec(path: Path) -> Optional[str]:
    """ffprobe로 첫 오디오 스트림 codec_name 조회 (없으면 None)."""
    import subprocess
    try:
        proc = subprocess.run(
            ["ffprobe","-v","error","-select_streams","a:0",
             "-show_entries","stream=codec_name","-of","default=nw=1:nk=1", str(path)],
            capture_output=True, text=True, check=True
        )
        name = proc.stdout.strip()
        return name if name else None
    except Exception:
        return None

def _has_audio_stream(path: Path) -> bool:
    return _probe_audio_codec(path) is not None

# -------------------------------------------------
# GFPGAN 모델 경로 결정 (다운로드 차단)
# -------------------------------------------------
def _resolve_gfpgan_model(version: str):
    # url 정보는 참고용으로만 유지(사용하지 않음)
    if version == "1":
        return ("original", 1, "GFPGANv1", "https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/GFPGANv1.pth")
    elif version == "1.2":
        return ("clean", 2, "GFPGANCleanv1-NoCE-C2", "https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth")
    elif version == "1.3":
        return ("clean", 2, "GFPGANv1.3", "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth")
    elif version == "1.4":
        return ("clean", 2, "GFPGANv1.4", "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth")
    elif version == "RestoreFormer":
        return ("RestoreFormer", 2, "RestoreFormer", "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth")
    else:
        raise ValueError(f"Wrong model version {version}")

def _pick_model_file(model_name: str, url: str) -> str:
    """로컬 경로에서만 찾고, 없으면 에러."""
    cands = [
        Path("experiments/pretrained_models") / f"{model_name}.pth",
        Path("gfpgan/weights") / f"{model_name}.pth",
    ]
    for c in cands:
        if c.exists():
            return str(c)
    raise FileNotFoundError(
        f"Model weight not found for {model_name}. "
        f"Expected one of: {', '.join(str(p) for p in cands)}"
    )

# -------------------------------------------------
# 배경 업샘플러 (RealESRGAN) - 로컬 가중치만 사용
# -------------------------------------------------
def _build_bg_upsampler(name: str, bg_tile: int):
    if name != "realesrgan":
        return None
    if not torch.cuda.is_available():  # CPU면 비권장
        return None
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    # 로컬 후보 경로(형님 환경: gfpgan/weights 가 최우선)
    candidates = [
        Path("/app/gfpgan/weights/RealESRGAN_x2plus.pth"),
        Path("gfpgan/weights/RealESRGAN_x2plus.pth"),
        Path("/usr/local/lib/python3.10/dist-packages/weights/RealESRGAN_x2plus.pth"),
    ]
    model_path = next((str(p) for p in candidates if p.exists()), None)
    if model_path is None:
        raise RuntimeError(
            "RealESRGAN_x2plus.pth not found. "
            "Place it under /app/gfpgan/weights or gfpgan/weights."
        )

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23,
                    num_grow_ch=32, scale=2)
    return RealESRGANer(
        scale=2,
        model_path=model_path,   # 로컬 파일만 사용
        model=model,
        tile=int(bg_tile),
        tile_pad=10,
        pre_pad=0,
        half=True,
    )

# -------------------------------------------------
# 싱글톤 러너
# -------------------------------------------------
class GFPGANRunner:
    def __init__(self, *, version: str, upscale: int, bg_upsampler: str, bg_tile: int):
        arch, ch_mul, model_name, url = _resolve_gfpgan_model(version)
        model_path = _pick_model_file(model_name, url)  # 로컬만
        bg = _build_bg_upsampler(bg_upsampler, bg_tile)
        self._cfg = {
            "version": version,
            "upscale": int(upscale),
            "bg_upsampler": bg_upsampler,
            "bg_tile": int(bg_tile),
            "arch": arch,
            "ch_mul": ch_mul,
            "model_name": model_name
        }
        self.restorer = GFPGANer(
            model_path=model_path,
            upscale=int(upscale),
            arch=arch,
            channel_multiplier=ch_mul,
            bg_upsampler=bg,
        )
        self._lock = threading.Lock()

    @property
    def cfg(self) -> Dict[str, Any]:
        return dict(self._cfg)

    def enhance_image(self, bgr_img: np.ndarray,
                      *, weight: float,
                      only_center_face: bool,
                      aligned: bool) -> np.ndarray:
        cropped_faces, restored_faces, restored_img = self.restorer.enhance(
            bgr_img,
            has_aligned=aligned,
            only_center_face=only_center_face,
            paste_back=True,
            weight=weight,
        )
        return restored_img if restored_img is not None else bgr_img

    def enhance_video_to_silent(self, in_video: Path, *,
                                framerate: int,
                                weight: float,
                                only_center_face: bool,
                                aligned: bool) -> Tuple[Path, Path]:
        if not in_video.exists():
            raise FileNotFoundError(f"input video not found: {in_video}")

        job_id = uuid.uuid4().hex
        work_dir = GFPGAN_OUT_DIR / f"gfpgan_work_{job_id}"
        restored_dir = work_dir / "restored_imgs"
        restored_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(in_video))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {in_video}")

        idx = 0
        with self._lock:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                restored = self.enhance_image(
                    frame,
                    weight=weight,
                    only_center_face=only_center_face,
                    aligned=aligned
                )
                # BGR 그대로 저장 (색 왜곡 방지)
                imwrite(restored, str(restored_dir / f"frame_{idx+1:04d}.png"))
                idx += 1
        cap.release()

        if idx == 0:
            _safe_rmtree(work_dir)
            raise RuntimeError("No frames decoded from input video.")

        silent_out = work_dir / "silent.mp4"
        _run_ffmpeg([
            "ffmpeg", "-y",
            "-framerate", str(framerate),
            "-i", str(restored_dir / "frame_%04d.png"),
            "-c:v", "libx264",
            "-pix_fmt", DFLT_PIX_FMT,
            "-crf", str(DFLT_CRF),
            str(silent_out),
        ])
        return work_dir, silent_out

# -------------------------------------------------
# 전역 싱글톤
# -------------------------------------------------
RUNNER: Optional[GFPGANRunner] = None
CURRENT_CFG: Dict[str, Any] = {}
_runner_lock = threading.Lock()

def _ensure_runner(*, version: str, upscale: int, bg_upsampler: str, bg_tile: int):
    global RUNNER, CURRENT_CFG
    desired = {"version": version, "upscale": int(upscale), "bg_upsampler": bg_upsampler, "bg_tile": int(bg_tile)}
    with _runner_lock:
        if RUNNER is None or any(CURRENT_CFG.get(k) != v for k, v in desired.items()):
            RUNNER = GFPGANRunner(version=version, upscale=upscale, bg_upsampler=bg_upsampler, bg_tile=bg_tile)
            CURRENT_CFG = RUNNER.cfg

# -------------------------------------------------
# 엔드포인트
# -------------------------------------------------
class HealthResp(BaseModel):
    status: str = "ok"
    device: str = "cpu"
    model_loaded: bool = False
    config: Dict[str, Any] = {}

@app.get("/health", response_model=HealthResp)
def health():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return HealthResp(status="ok", device=dev, model_loaded=bool(RUNNER is not None), config=CURRENT_CFG)

@app.post("/warmup")
def warmup(
    version: str = Form(DFLT_VERSION),
    upscale: int = Form(DFLT_UPSCALE),
    bg_upsampler: str = Form(DFLT_BG_UPSAMPLER),
    bg_tile: int = Form(DFLT_BG_TILE),
    weight: float = Form(DFLT_WEIGHT),
    only_center_face: bool = Form(DFLT_ONLY_CENTER_FACE),
    aligned: bool = Form(DFLT_ALIGNED),
):
    _ensure_runner(version=version, upscale=upscale, bg_upsampler=bg_upsampler, bg_tile=bg_tile)

    dummy = np.full((256, 256, 3), 255, dtype=np.uint8)
    cv2.circle(dummy, (128, 128), 90, (0, 0, 0), 3)
    cv2.circle(dummy, (95, 115), 10, (0, 0, 0), -1)
    cv2.circle(dummy, (161, 115), 10, (0, 0, 0), -1)
    cv2.ellipse(dummy, (128, 160), (40, 20), 0, 10, 170, (0, 0, 0), 3)

    t0 = time.time()
    restored = RUNNER.enhance_image(dummy, weight=weight, only_center_face=only_center_face, aligned=aligned)
    out_path = WARMUP_DIR / "warmup_gfpgan.png"
    imwrite(restored, str(out_path))
    elapsed_ms = int((time.time() - t0) * 1000)

    return JSONResponse({
        "status":"ok","message":"warmup done (singleton)","output":str(out_path),
        "elapsed_ms":elapsed_ms,"config":CURRENT_CFG,
        "weight":weight,"only_center_face":only_center_face,"aligned":aligned
    })

@app.post("/enhance_video")
def enhance_video(
    # 입력/출력
    input_video_path: Optional[str] = Form(None, description="미지정 시 wav2lip_output_queue 최신 mp4 선택"),
    original_audio_path: Optional[str] = Form(None, description="명시 시 이 오디오 사용(기본은 비디오의 오디오)"),
    output_basename: Optional[str] = Form(None),

    # CLI 동등 파라미터
    version: str = Form(DFLT_VERSION),
    upscale: int = Form(DFLT_UPSCALE),
    bg_upsampler: str = Form(DFLT_BG_UPSAMPLER),
    bg_tile: int = Form(DFLT_BG_TILE),
    only_center_face: bool = Form(DFLT_ONLY_CENTER_FACE),
    aligned: bool = Form(DFLT_ALIGNED),
    weight: float = Form(DFLT_WEIGHT),
    ext: str = Form(DFLT_EXT),

    # 인코딩/동작 옵션
    framerate: int = Form(DFLT_FRAMERATE),
    pix_fmt: str = Form(DFLT_PIX_FMT),
    crf: int = Form(DFLT_CRF),
    audio_copy: bool = Form(DFLT_AUDIO_COPY),
    audio_bitrate: str = Form(DFLT_AUDIO_BITRATE),
):
    """
    내부 싱글톤으로 추론 수행 (모델 재로딩 없음)
    - 기본: Wav2Lip 결과 비디오의 오디오를 그대로 사용
    - 필요시 original_audio_path로 대체 가능
    """
    _ensure_runner(version=version, upscale=upscale, bg_upsampler=bg_upsampler, bg_tile=bg_tile)

    # 영상
    if input_video_path:
        in_video = Path(input_video_path)
        if not in_video.exists():
            raise HTTPException(status_code=400, detail=f"input_video_path not found: {in_video}")
    else:
        in_video = _pick_latest(W2L_OUT_DIR, ["*.mp4", "*.mov", "*.mkv", "*.avi"])
        if not in_video:
            raise HTTPException(status_code=400, detail="No video found in wav2lip_output_queue/")

    # 오디오: 기본은 비디오 자체의 오디오 트랙
    if original_audio_path:
        in_audio = Path(original_audio_path)
        if not in_audio.exists():
            raise HTTPException(status_code=400, detail=f"original_audio_path not found: {in_audio}")
    else:
        in_audio = in_video

    # 비디오에서 오디오가 실제로 존재하는지 확인 (없으면 재지정 요구)
    if in_audio == in_video and not _has_audio_stream(in_audio):
        raise HTTPException(status_code=400, detail="Selected video has no audio stream. Provide original_audio_path.")

    # 결과 파일명
    if output_basename:
        final_name = f"{output_basename}.mp4"
    else:
        final_name = f"gfpgan_{in_video.stem}.mp4"

    final_out = GFPGAN_OUT_DIR / final_name
    if final_out.exists():
        final_out = GFPGAN_OUT_DIR / f"{final_out.stem}_{int(time.time())}.mp4"

    try:
        # 1) 프레임 복원 → 무음 mp4
        work_dir, silent_mp4 = RUNNER.enhance_video_to_silent(
            in_video, framerate=int(framerate),
            weight=weight, only_center_face=only_center_face, aligned=aligned
        )

        # 2) 무음 영상 + 오디오 mux (기본: copy)
        ff_cmd = [
            "ffmpeg", "-y",
            "-i", str(silent_mp4),
            "-i", str(in_audio),
            "-map", "0:v", "-map", "1:a",
            "-c:v", "libx264", "-pix_fmt", str(pix_fmt), "-crf", str(crf),
        ]
        if audio_copy:
            ff_cmd += ["-c:a", "copy"]
        else:
            ff_cmd += ["-c:a", "aac", "-b:a", str(audio_bitrate)]
        ff_cmd += ["-shortest", str(final_out)]
        _run_ffmpeg(ff_cmd)

        return JSONResponse({
            "status": "ok",
            "message": "GFPGAN video enhancement successful (singleton)",
            "input_video": str(in_video),
            "input_audio": str(in_audio),
            "output": str(final_out),
            "config": CURRENT_CFG,
            "encode": {
                "pix_fmt": pix_fmt, "crf": crf,
                "audio_copy": audio_copy,
                "audio_bitrate": audio_bitrate if not audio_copy else "copy"
            },
            "infer_params": {
                "weight": weight, "only_center_face": only_center_face, "aligned": aligned
            }
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GFPGAN enhancement failed: {e}")
    finally:
        if 'work_dir' in locals():
            _safe_rmtree(work_dir)
