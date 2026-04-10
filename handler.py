"""
RVC v2 Worker for RunPod Serverless.

Two modes:
  mode: "train"  → Train RVC model from voice audio → save .pth to /runpod-volume/
  mode: "infer"  → Load .pth from /runpod-volume/ → voice conversion → upload result

Network Volume mounted at /runpod-volume/ for persistent model storage.
"""

import os
import sys
import tempfile
import time
import subprocess
import traceback
import shutil

import requests
import runpod
import numpy as np

# ── Constants ────────────────────────────────────────────────────
RVC_WEBUI_DIR = "/app/rvc-webui"
VOLUME_DIR = "/runpod-volume"
MODELS_DIR = os.path.join(VOLUME_DIR, "rvc_models")


def download_file(url: str, dest_path: str):
    print(f"[Download] {url}")
    resp = requests.get(url, stream=True, timeout=300)
    resp.raise_for_status()
    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    size_mb = os.path.getsize(dest_path) / (1024 * 1024)
    print(f"[Download] Done: {size_mb:.1f} MB")


def upload_file(file_path: str, filename: str, max_retries: int = 3) -> str:
    size_mb = os.path.getsize(file_path) / 1024 / 1024
    print(f"[Upload] {filename} ({size_mb:.1f} MB)...")
    for attempt in range(1, max_retries + 1):
        try:
            with open(file_path, "rb") as f:
                resp = requests.post(
                    "https://tmpfiles.org/api/v1/upload",
                    files={"file": (filename, f, "audio/wav")},
                    timeout=120,
                )
            resp.raise_for_status()
            data = resp.json()
            if data.get("status") != "success":
                raise RuntimeError(f"Response not success: {data}")
            url = data["data"]["url"].replace("tmpfiles.org/", "tmpfiles.org/dl/")
            print(f"[Upload] Done: {url}")
            return url
        except Exception as e:
            print(f"[Upload] Attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                time.sleep(3)
            else:
                raise


def apply_post_fx(audio_path: str, vocal_volume: float = 1.3, reverb_amount: float = 0.25):
    """Apply pedalboard effects."""
    try:
        from pedalboard import Pedalboard, Reverb, Compressor, HighpassFilter, Gain
        from pedalboard.io import AudioFile

        with AudioFile(audio_path) as f:
            sr = f.samplerate
            audio = f.read(f.frames)

        effects = [
            HighpassFilter(cutoff_frequency_hz=80),
            Compressor(threshold_db=-20, ratio=3.0, attack_ms=10, release_ms=100),
        ]
        if vocal_volume != 1.0:
            effects.append(Gain(gain_db=20 * np.log10(max(vocal_volume, 0.01))))
        if reverb_amount > 0:
            effects.append(Reverb(
                room_size=min(0.2 + reverb_amount * 0.8, 0.95),
                damping=min(0.5 + reverb_amount * 0.3, 0.85),
                wet_level=min(0.15 + reverb_amount * 0.55, 0.55),
                dry_level=max(0.7 - reverb_amount * 0.3, 0.45),
                width=0.8,
            ))

        processed = Pedalboard(effects)(audio, sr)
        peak = np.max(np.abs(processed))
        if peak > 1.0:
            processed = processed / peak * 0.95

        out_path = audio_path.replace(".wav", "_fx.wav")
        with AudioFile(out_path, "w", sr, processed.shape[0]) as f:
            f.write(processed)
        print(f"[FX] {len(effects)} effects applied")
        return out_path
    except Exception as e:
        print(f"[FX] Error: {e}")
        return audio_path


# ── TRAIN MODE ───────────────────────────────────────────────────

def handle_train(job_input, tmpdir):
    """
    Train an RVC v2 model from voice audio.
    Input: voice_url, user_id, sample_rate (default 48000), epochs (default 200)
    Output: model saved to /runpod-volume/rvc_models/{user_id}/
    """
    user_id = job_input["user_id"]
    voice_url = job_input["voice_url"]
    sample_rate = int(job_input.get("sample_rate", 48000))
    epochs = int(job_input.get("epochs", 200))
    batch_size = int(job_input.get("batch_size", 4))

    print(f"[Train] user_id={user_id}, sr={sample_rate}, epochs={epochs}, batch={batch_size}")

    # 1. Download voice audio
    voice_path = os.path.join(tmpdir, "voice.wav")
    download_file(voice_url, voice_path)

    # 2. Prepare dataset directory
    dataset_dir = os.path.join(tmpdir, "dataset")
    os.makedirs(dataset_dir, exist_ok=True)
    shutil.copy(voice_path, os.path.join(dataset_dir, "voice.wav"))

    # 3. Model output directory
    model_dir = os.path.join(MODELS_DIR, user_id)
    os.makedirs(model_dir, exist_ok=True)

    model_name = f"rvc_{user_id}"

    # 4. Training pipeline — all steps via subprocess (RVC scripts use sys.argv)
    webui = RVC_WEBUI_DIR
    exp_dir = os.path.join(webui, "logs", model_name)
    os.makedirs(exp_dir, exist_ok=True)

    def run_step(step_name, cmd, timeout_sec=300):
        print(f"[Train] {step_name}...", flush=True)
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec, cwd=webui,
                           env={**os.environ, "PYTHONUNBUFFERED": "1"})
        stdout = r.stdout or ""
        stderr = r.stderr or ""
        if stdout:
            print(f"[Train] {step_name} STDOUT: {stdout[-500:]}", flush=True)
        if stderr:
            print(f"[Train] {step_name} STDERR: {stderr[-300:]}", flush=True)
        # Check both return code and error keywords in output
        if r.returncode != 0:
            raise RuntimeError(f"{step_name} failed (exit {r.returncode}): {stderr[-300:]}")
        if "is shut down" in stdout or "does not exist" in stdout:
            raise RuntimeError(f"{step_name} failed: {stdout[-300:]}")
        print(f"[Train] {step_name} done", flush=True)

    # Step 1: Preprocess
    # Args: inp_root, sr, n_p, exp_dir, per, noparallel
    run_step("Preprocess", [
        "python", "infer/modules/train/preprocess.py",
        dataset_dir, str(sample_rate), "2", exp_dir, "3.7", "1",
    ])

    # Step 2: Extract F0
    # Args: exp_dir, n_processes, f0method
    run_step("Extract F0", [
        "python", "infer/modules/train/extract/extract_f0_print.py",
        exp_dir, "1", "rmvpe",
    ])

    # Step 3: Extract features
    # Official args: device, leng(total_procs), idx(proc_index), n_g(gpu_id), logs_path, version, is_half
    run_step("Extract Features", [
        "python", "infer/modules/train/extract_feature_print.py",
        "cuda:0", "1", "0", "0", exp_dir, "v2", "True",
    ])

    # Step 3.5: Generate config.json (required by train.py, normally done by click_train in WebUI)
    import json as _json
    sr_key = "48k" if sample_rate >= 48000 else ("40k" if sample_rate >= 40000 else "32k")
    config_path = os.path.join(webui, "configs", "v2", f"{sr_key}.json")
    config_save = os.path.join(exp_dir, "config.json")
    if os.path.exists(config_path) and not os.path.exists(config_save):
        with open(config_path, "r") as cf:
            cfg_data = _json.load(cf)
        with open(config_save, "w") as cf:
            _json.dump(cfg_data, cf, indent=4, sort_keys=True)
        print(f"[Train] config.json created from {config_path}")
    else:
        print(f"[Train] config.json: exists={os.path.exists(config_save)}, template={os.path.exists(config_path)}")

    # Step 4: Train model
    # Official: -e name -sr sr -f0 1 -bs batch -g gpus -te epochs -se save -pg G.pth -pd D.pth -l save_latest -c cache_gpu -sw save_weights -v version
    run_step("Train", [
        "python", "infer/modules/train/train.py",
        "-e", model_name,
        "-sr", sr_key,
        "-f0", "1",
        "-bs", str(batch_size),
        "-g", "0",
        "-te", str(epochs),
        "-se", "50",
        "-pg", f"assets/pretrained_v2/f0G{sr_key}.pth",
        "-pd", f"assets/pretrained_v2/f0D{sr_key}.pth",
        "-l", "1",
        "-c", "0",
        "-sw", "1",
        "-v", "v2",
    ], timeout_sec=600)

    # 5. Find and copy trained model to volume
    logs_dir = os.path.join("/app/rvc-webui/logs", model_name)

    # Debug: list all files in logs dir
    if os.path.exists(logs_dir):
        all_files = os.listdir(logs_dir)
        print(f"[Train] Files in {logs_dir}: {all_files}")
    else:
        print(f"[Train] WARNING: {logs_dir} does not exist!")

    pth_files = [f for f in os.listdir(logs_dir) if f.endswith(".pth")] if os.path.exists(logs_dir) else []

    if not pth_files:
        # Check weights directory
        weights_dir = os.path.join("/app/rvc-webui/assets/weights")
        if os.path.exists(weights_dir):
            all_weights = os.listdir(weights_dir)
            print(f"[Train] Files in weights dir: {all_weights}")
            pth_files = [f for f in all_weights if model_name in f and f.endswith(".pth")]
            if pth_files:
                logs_dir = weights_dir

    if not pth_files:
        # Check if training actually produced any output
        for root, dirs, files in os.walk(os.path.join("/app/rvc-webui/logs")):
            pth_in_tree = [f for f in files if f.endswith(".pth")]
            if pth_in_tree:
                print(f"[Train] Found .pth files at {root}: {pth_in_tree}")
        raise RuntimeError(f"No .pth model found after training. Check logs above for file locations.")

    pth_files.sort()
    best_pth = pth_files[-1]
    src_pth = os.path.join(logs_dir, best_pth)
    dst_pth = os.path.join(model_dir, "model.pth")
    shutil.copy(src_pth, dst_pth)

    # Copy index file if exists
    index_files = []
    for d in [logs_dir, os.path.join("/app/rvc-webui/logs", model_name)]:
        if os.path.exists(d):
            index_files += [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".index")]
    dst_index = None
    if index_files:
        dst_index = os.path.join(model_dir, "model.index")
        shutil.copy(index_files[-1], dst_index)

    model_size_mb = os.path.getsize(dst_pth) / (1024 * 1024)
    print(f"[Train] Model saved: {dst_pth} ({model_size_mb:.1f} MB)")

    return {
        "status": "success",
        "user_id": user_id,
        "model_path": dst_pth,
        "model_size_mb": round(model_size_mb, 2),
        "has_index": dst_index is not None,
    }


# ── INFER MODE ───────────────────────────────────────────────────

def handle_infer(job_input, tmpdir):
    """
    Run RVC v2 inference using a trained model.
    Input: user_id, song_url, voice_url (for reference), pitch_shift, etc.
    Output: converted audio URL
    """
    user_id = job_input["user_id"]
    song_url = job_input["song_url"]
    pitch_shift = int(job_input.get("pitch_shift", 0))
    index_rate = float(job_input.get("index_rate", 0.5))
    filter_radius = int(job_input.get("filter_radius", 3))
    vocal_volume = float(job_input.get("vocal_volume", 1.3))
    reverb = float(job_input.get("reverb", 0.25))
    output_format = job_input.get("output_format", "mp3_192")
    cover_image = job_input.get("cover_image", "")
    task_id = job_input.get("task_id", "unknown")

    print(f"[Infer] user_id={user_id}, pitch={pitch_shift}, index_rate={index_rate}")

    # 1. Check model exists
    model_dir = os.path.join(MODELS_DIR, user_id)
    model_path = os.path.join(model_dir, "model.pth")
    index_path = os.path.join(model_dir, "model.index")

    if not os.path.exists(model_path):
        raise RuntimeError(f"Model not found for user {user_id}. Please train first.")

    # 2. Download song
    song_path = os.path.join(tmpdir, "song.wav")
    download_file(song_url, song_path)

    # 3. Separate vocals (using demucs)
    print("[Infer] Separating vocals...")
    demucs_out = os.path.join(tmpdir, "demucs_out")
    demucs_cmd = [
        "python", "-m", "demucs",
        "-n", "htdemucs",
        "--two-stems", "vocals",
        "-o", demucs_out,
        song_path,
    ]
    result = subprocess.run(demucs_cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"Demucs failed: {result.stderr[-300:]}")

    song_name = os.path.splitext(os.path.basename(song_path))[0]
    vocals_path = os.path.join(demucs_out, "htdemucs", song_name, "vocals.wav")
    instrumental_path = os.path.join(demucs_out, "htdemucs", song_name, "no_vocals.wav")

    if not os.path.exists(vocals_path):
        raise RuntimeError("Vocal separation failed")

    # 4. RVC inference using rvc-python or rvc package
    print("[Infer] Running RVC inference...")
    output_wav = os.path.join(tmpdir, "rvc_output.wav")

    infer_script = f"""
import sys
try:
    from rvc.modules.vc.modules import VC
    from pathlib import Path
    from scipy.io import wavfile

    vc = VC()
    vc.get_vc('{model_path}')
    tgt_sr, audio_opt, times, _ = vc.vc_single(
        sid=0,
        input_audio_path=Path('{vocals_path}'),
        f0_up_key={pitch_shift},
        f0_method='rmvpe',
        index_file='{index_path}' if '{index_path}' and __import__('os').path.exists('{index_path}') else '',
        index_rate={index_rate},
        filter_radius={filter_radius},
        protect=0.33,
    )
    wavfile.write('{output_wav}', tgt_sr, audio_opt)
    print(f'RVC inference done, sr={{tgt_sr}}')
except Exception as e:
    print(f'Error: {{e}}')
    import traceback; traceback.print_exc()
    sys.exit(1)
"""
    script_path = os.path.join(tmpdir, "rvc_infer.py")
    with open(script_path, "w") as f:
        f.write(infer_script)

    start = time.time()
    result = subprocess.run(["python", script_path], capture_output=True, text=True, timeout=300)
    infer_time = time.time() - start

    print(f"[Infer] STDOUT: {result.stdout[-300:]}")
    if result.returncode != 0:
        raise RuntimeError(f"RVC inference failed: {result.stderr[-300:]}")

    if not os.path.exists(output_wav):
        raise RuntimeError("RVC produced no output")

    print(f"[Infer] RVC done in {infer_time:.1f}s")

    # 5. Mix with instrumental
    print("[Infer] Mixing...")
    final_wav = os.path.join(tmpdir, "final_cover.wav")
    mix_cmd = [
        "ffmpeg", "-y",
        "-i", output_wav,
        "-i", instrumental_path,
        "-filter_complex",
        "[0:a][1:a]amix=inputs=2:duration=longest",
        "-ac", "2", "-ar", "44100",
        final_wav,
    ]
    subprocess.run(mix_cmd, capture_output=True, timeout=120)

    # 6. Post FX
    if vocal_volume != 1.0 or reverb > 0:
        final_wav = apply_post_fx(final_wav, vocal_volume, reverb)

    # 7. Format conversion + cover image
    import torchaudio
    output_info = torchaudio.info(final_wav)
    output_duration = output_info.num_frames / output_info.sample_rate

    if output_format in ("mp3_320", "mp3_192"):
        bitrate = "320k" if output_format == "mp3_320" else "192k"
        mp3_output = final_wav.replace(".wav", ".mp3")

        cover_path = None
        if cover_image:
            cover_url = f"https://raw.githubusercontent.com/WhistleB/coverversion-worker/main/assets/covers/{cover_image}.png"
            cover_path = os.path.join(tmpdir, "cover.png")
            try:
                download_file(cover_url, cover_path)
            except Exception:
                cover_path = None

        if cover_path and os.path.exists(cover_path):
            convert_cmd = [
                "ffmpeg", "-y", "-i", final_wav, "-i", cover_path,
                "-map", "0:a", "-map", "1",
                "-c:a", "libmp3lame", "-b:a", bitrate,
                "-c:v", "png", "-disposition:v", "attached_pic",
                "-id3v2_version", "3", mp3_output,
            ]
        else:
            convert_cmd = ["ffmpeg", "-y", "-i", final_wav, "-b:a", bitrate, mp3_output]

        subprocess.run(convert_cmd, capture_output=True, timeout=60)
        if os.path.exists(mp3_output):
            final_wav = mp3_output

    # 8. Upload
    output_size_mb = os.path.getsize(final_wav) / (1024 * 1024)
    file_ext = os.path.splitext(final_wav)[1]
    output_url = upload_file(final_wav, f"cover_{task_id}{file_ext}")

    return {
        "task_id": task_id,
        "status": "success",
        "output_url": output_url,
        "duration": round(output_duration, 2),
        "inference_time": round(infer_time, 2),
        "output_format": output_format,
        "size_mb": round(output_size_mb, 2),
    }


# ── HANDLER ──────────────────────────────────────────────────────

def handler(job):
    job_input = job["input"]
    mode = job_input.get("mode", "infer")

    # Warmup
    if mode == "warmup":
        print("[Warmup] Worker ready.")
        return {"status": "warm"}

    print(f"\n{'='*60}")
    print(f"[Job] mode={mode}")
    print(f"{'='*60}")

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            if mode == "train":
                runpod.serverless.progress_update(job, {
                    "stage": "training", "progress": 0.1
                })
                result = handle_train(job_input, tmpdir)
                return result

            elif mode == "infer":
                runpod.serverless.progress_update(job, {
                    "stage": "downloading", "progress": 0.05
                })
                result = handle_infer(job_input, tmpdir)
                return result

            else:
                return {"status": "error", "error": f"Unknown mode: {mode}"}

        except Exception as e:
            traceback.print_exc()
            return {"status": "error", "error": str(e)}


if __name__ == "__main__":
    os.makedirs(MODELS_DIR, exist_ok=True)
    print("[Init] RVC v2 Worker")
    print(f"[Init] Models dir: {MODELS_DIR}")
    print(f"[Init] Volume exists: {os.path.exists(VOLUME_DIR)}")
    runpod.serverless.start({"handler": handler})
