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

        # Fade-in/out to prevent start/end pops (same as Seed-VC)
        fade_samples = int(sr * 0.5)  # 500ms
        if audio.shape[-1] > fade_samples * 2:
            fade_in = np.linspace(0, 1, fade_samples)
            fade_out = np.linspace(1, 0, fade_samples)
            for ch in range(audio.shape[0]):
                audio[ch, :fade_samples] *= fade_in
                audio[ch, -fade_samples:] *= fade_out

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
    # Accept either voice_urls (array) or legacy voice_url (single)
    voice_urls = job_input.get("voice_urls")
    if not voice_urls:
        single = job_input.get("voice_url")
        if single:
            voice_urls = [single]
    if not voice_urls:
        raise ValueError("voice_url or voice_urls required")
    sample_rate = int(job_input.get("sample_rate", 48000))
    epochs = int(job_input.get("epochs", 200))
    batch_size = int(job_input.get("batch_size", 4))
    separate_for_training = bool(job_input.get("separate_for_training", False))  # 上传素材带伴奏时开启

    print(f"[Train] user_id={user_id}, sr={sample_rate}, epochs={epochs}, batch={batch_size}, files={len(voice_urls)}, separate={separate_for_training}")

    # 1. Prepare dataset directory and download all voice files
    download_dir = os.path.join(tmpdir, "downloaded")
    os.makedirs(download_dir, exist_ok=True)
    for i, url in enumerate(voice_urls):
        dst = os.path.join(download_dir, f"voice_{i:02d}.wav")
        download_file(url, dst)
    print(f"[Train] Downloaded {len(voice_urls)} voice file(s)")

    # 1.5 Optional: Run vocal separation if user uploaded songs with backing music
    dataset_dir = os.path.join(tmpdir, "dataset")
    os.makedirs(dataset_dir, exist_ok=True)
    if separate_for_training:
        print(f"[Train] Separating vocals from training audio (BS Roformer)...")
        for i, fname in enumerate(sorted(os.listdir(download_dir))):
            if not fname.endswith('.wav'):
                continue
            src = os.path.join(download_dir, fname)
            sep_out = os.path.join(tmpdir, f"sep_out_{i:02d}")
            try:
                vocals_path, _ = separate_vocals_bs_roformer(src, sep_out)
                # Copy clean vocals to dataset dir with original name
                dst = os.path.join(dataset_dir, fname)
                shutil.copy(vocals_path, dst)
                print(f"[Train] Separated {fname} → clean vocals")
            except Exception as e:
                print(f"[Train] Separation failed for {fname}: {e}, using original")
                shutil.copy(src, os.path.join(dataset_dir, fname))
        print(f"[Train] Vocal separation done for training data")
    else:
        # No separation — copy as-is
        for fname in sorted(os.listdir(download_dir)):
            if fname.endswith('.wav'):
                shutil.copy(os.path.join(download_dir, fname), os.path.join(dataset_dir, fname))

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

    # Step 3.6: Generate filelist.txt (normally done by click_train in WebUI)
    # train.py reads this file to know which training samples to load.
    import random as _random
    gt_wavs_dir = os.path.join(exp_dir, "0_gt_wavs")
    feature_dir = os.path.join(exp_dir, "3_feature768")  # v2
    f0_dir = os.path.join(exp_dir, "2a_f0")
    f0nsf_dir = os.path.join(exp_dir, "2b-f0nsf")
    spk_id5 = 0
    names = (
        set([n.split(".")[0] for n in os.listdir(gt_wavs_dir)])
        & set([n.split(".")[0] for n in os.listdir(feature_dir)])
        & set([n.split(".")[0] for n in os.listdir(f0_dir)])
        & set([n.split(".")[0] for n in os.listdir(f0nsf_dir)])
    )
    print(f"[Train] filelist intersect: {len(names)} samples")
    opt = []
    for name in names:
        opt.append(
            "%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%s"
            % (gt_wavs_dir, name, feature_dir, name, f0_dir, name, f0nsf_dir, name, spk_id5)
        )
    fea_dim = 768  # v2
    now_dir = webui
    for _ in range(2):
        opt.append(
            "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s/logs/mute/2a_f0/mute.wav.npy|%s/logs/mute/2b-f0nsf/mute.wav.npy|%s"
            % (now_dir, sr_key, now_dir, fea_dim, now_dir, now_dir, spk_id5)
        )
    _random.shuffle(opt)
    filelist_path = os.path.join(exp_dir, "filelist.txt")
    with open(filelist_path, "w") as f:
        f.write("\n".join(opt))
    print(f"[Train] filelist.txt written: {len(opt)} entries → {filelist_path}")

    # Step 3.7: Build FAISS index for retrieval (makes index_rate work during inference)
    index_script = f"""
import sys, os, numpy as np, faiss
sys.path.insert(0, '{webui}')

feature_dir = os.path.join('{exp_dir}', '3_feature768')
npys = []
for f in sorted(os.listdir(feature_dir)):
    if f.endswith('.npy'):
        feat = np.load(os.path.join(feature_dir, f))
        npys.append(feat)
if npys:
    big_npy = np.concatenate(npys, axis=0)
    print(f'Total feature vectors: {{big_npy.shape}}')
    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
    n_ivf = max(n_ivf, 1)
    index = faiss.index_factory(768, f'IVF{{n_ivf}},Flat')
    index.train(big_npy)
    index.add(big_npy)
    index_path = os.path.join('{exp_dir}', 'added_index.index')
    faiss.write_index(index, index_path)
    print(f'Index saved: {{index_path}} ({{big_npy.shape[0]}} vectors, IVF{{n_ivf}})')
else:
    print('No feature files found, skipping index')
"""
    index_script_path = os.path.join(tmpdir, "build_index.py")
    with open(index_script_path, "w") as f:
        f.write(index_script)
    ir = subprocess.run(["python", index_script_path], capture_output=True, text=True, timeout=120, cwd=webui)
    print(f"[Train] Index STDOUT: {ir.stdout.strip()}")
    if ir.stderr:
        print(f"[Train] Index STDERR: {ir.stderr[-200:]}")
    if ir.returncode != 0:
        print(f"[Train] Index build failed (non-critical): {ir.stderr[-200:]}")

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
    ], timeout_sec=3600)

    # 5. Convert raw G_*.pth checkpoint into slim inference model.
    # train.py only writes G_*.pth/D_*.pth (full optimizer state ~430MB) — these
    # don't have the {"config", "weight", "info"} keys that VC.get_vc() expects.
    # WebUI's "training done" callback runs extract_small_model() to produce a
    # ~55MB inference-ready file under assets/weights/{name}.pth. We replicate it.
    logs_dir = os.path.join("/app/rvc-webui/logs", model_name)
    if os.path.exists(logs_dir):
        print(f"[Train] Files in {logs_dir}: {os.listdir(logs_dir)}")

    g_files = sorted([f for f in os.listdir(logs_dir)
                      if f.startswith("G_") and f.endswith(".pth")]) if os.path.exists(logs_dir) else []
    if not g_files:
        raise RuntimeError(f"No G_*.pth checkpoint found in {logs_dir}")
    g_path = os.path.join(logs_dir, g_files[-1])
    print(f"[Train] Using G checkpoint: {g_path}")

    # Run extract_small_model in the WebUI's own context (it uses relative paths)
    extract_script = f"""
import sys, os
sys.path.insert(0, '/app/rvc-webui')
os.chdir('/app/rvc-webui')
from infer.lib.train.process_ckpt import extract_small_model
result = extract_small_model(
    '{g_path}',
    '{model_name}',
    '{sr_key}',
    True,
    'Trained on RunPod Serverless',
    'v2',
)
print(f'extract_small_model: {{result}}')
"""
    extract_path = os.path.join(tmpdir, "extract.py")
    with open(extract_path, "w") as f:
        f.write(extract_script)
    er = subprocess.run(["python", extract_path], capture_output=True, text=True, timeout=120, cwd=webui)
    print(f"[Train] extract STDOUT: {er.stdout}")
    if er.stderr:
        print(f"[Train] extract STDERR: {er.stderr[-300:]}")
    if er.returncode != 0:
        raise RuntimeError(f"extract_small_model failed: {er.stderr[-300:]}")

    # The slim model is now at assets/weights/{model_name}.pth
    slim_pth = os.path.join("/app/rvc-webui/assets/weights", f"{model_name}.pth")
    if not os.path.exists(slim_pth):
        raise RuntimeError(f"Slim inference model missing at {slim_pth}")

    dst_pth = os.path.join(model_dir, "model.pth")
    shutil.copy(slim_pth, dst_pth)

    # Copy index file if exists
    index_files = []
    for d in [logs_dir]:
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


# ── BS Roformer + Karaoke separation ─────────────────────────────

def separate_vocals_bs_roformer(song_path: str, output_dir: str):
    """Separate vocals/instrumental using BS Roformer (SDR 10.87)."""
    print(f"[BS-Roformer] Separating vocals...")
    os.makedirs(output_dir, exist_ok=True)
    bsr_input = os.path.join(output_dir, "_input")
    os.makedirs(bsr_input, exist_ok=True)
    shutil.copy(song_path, os.path.join(bsr_input, os.path.basename(song_path)))
    bsr_output = os.path.join(output_dir, "_output")
    os.makedirs(bsr_output, exist_ok=True)

    cmd = [
        "python", "/app/msst/inference.py",
        "--model_type", "bs_roformer",
        "--config_path", "/app/msst/bs_roformer_vocals.yaml",
        "--start_check_point", "/app/msst/bs_roformer_vocals.ckpt",
        "--input_folder", bsr_input,
        "--store_dir", bsr_output,
        "--extract_instrumental",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd="/app/msst")
    if result.stdout:
        print(f"[BS-Roformer] STDOUT: {result.stdout[-300:]}")
    if result.returncode != 0:
        raise RuntimeError(f"BS-Roformer failed: {result.stderr[-300:]}")

    vocals_path = None
    instrumental_path = None
    for root, dirs, files in os.walk(bsr_output):
        for f in files:
            if not f.endswith('.wav'):
                continue
            lower = f.lower()
            full = os.path.join(root, f)
            if 'vocal' in lower and 'instrumental' not in lower and 'other' not in lower:
                vocals_path = full
            elif 'instrumental' in lower or 'other' in lower:
                instrumental_path = full

    if not vocals_path or not instrumental_path:
        all_files = []
        for root, dirs, files in os.walk(bsr_output):
            for f in files:
                all_files.append(os.path.relpath(os.path.join(root, f), bsr_output))
        raise RuntimeError(f"BS-Roformer output not found in: {all_files}")

    print(f"[BS-Roformer] Done.")
    return vocals_path, instrumental_path


def separate_karaoke(vocals_path: str, output_dir: str):
    """Separate lead vocals from backing vocals using BS Roformer Karaoke."""
    print(f"[Karaoke] Separating lead/backing vocals...")
    os.makedirs(output_dir, exist_ok=True)
    karaoke_input = os.path.join(output_dir, "input")
    os.makedirs(karaoke_input, exist_ok=True)
    shutil.copy(vocals_path, os.path.join(karaoke_input, os.path.basename(vocals_path)))

    cmd = [
        "python", "/app/msst/inference.py",
        "--model_type", "bs_roformer",
        "--config_path", "/app/msst/config_karaoke_frazer_becruily.yaml",
        "--start_check_point", "/app/msst/bs_roformer_karaoke_frazer_becruily.ckpt",
        "--input_folder", karaoke_input,
        "--store_dir", output_dir,
        "--extract_instrumental",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd="/app/msst")
    if result.stdout:
        print(f"[Karaoke] STDOUT: {result.stdout[-300:]}")
    if result.returncode != 0:
        raise RuntimeError(f"Karaoke failed: {result.stderr[-300:]}")

    lead_path = None
    backing_path = None
    for root, dirs, files in os.walk(output_dir):
        for f in files:
            if not f.endswith('.wav'):
                continue
            full = os.path.join(root, f)
            lower = f.lower()
            if 'instrumental' in lower or 'other' in lower:
                backing_path = full
            elif 'vocal' in lower:
                lead_path = full

    if not lead_path:
        all_files = []
        for root, dirs, files in os.walk(output_dir):
            for f in files:
                all_files.append(os.path.relpath(os.path.join(root, f), output_dir))
        raise RuntimeError(f"Karaoke lead vocals not found in: {all_files}")

    print(f"[Karaoke] Done.")
    return lead_path, backing_path


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
    separation_engine = job_input.get("separation_engine", "demucs")
    karaoke_enabled = bool(job_input.get("karaoke_enabled", False))

    print(f"[Infer] user_id={user_id}, pitch={pitch_shift}, index_rate={index_rate}, sep={separation_engine}, karaoke={karaoke_enabled}")

    # 1. Check model exists
    model_dir = os.path.join(MODELS_DIR, user_id)
    model_path = os.path.join(model_dir, "model.pth")
    index_path = os.path.join(model_dir, "model.index")

    if not os.path.exists(model_path):
        raise RuntimeError(f"Model not found for user {user_id}. Please train first.")

    # 2. Download song
    song_path = os.path.join(tmpdir, "song.wav")
    download_file(song_url, song_path)

    # 3. Separate vocals
    t = time.time()
    if separation_engine == "bs_roformer":
        vocals_path, instrumental_path = separate_vocals_bs_roformer(
            song_path, os.path.join(tmpdir, "bsroformer_out"))
    else:
        print("[Infer] Separating vocals (demucs)...")
        demucs_out = os.path.join(tmpdir, "demucs_out")
        demucs_cmd = [
            "python", "-m", "demucs",
            "-n", "htdemucs",
            "--two-stems", "vocals",
            "-o", demucs_out,
            song_path,
        ]
        result = subprocess.run(demucs_cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            raise RuntimeError(f"Demucs failed: {result.stderr[-300:]}")
        song_name = os.path.splitext(os.path.basename(song_path))[0]
        vocals_path = os.path.join(demucs_out, "htdemucs", song_name, "vocals.wav")
        instrumental_path = os.path.join(demucs_out, "htdemucs", song_name, "no_vocals.wav")
        if not os.path.exists(vocals_path):
            raise RuntimeError("Vocal separation failed")
    separation_time = time.time() - t
    print(f"[Infer] Separation ({separation_engine}): {separation_time:.1f}s")

    # 3.5 Karaoke separation (optional)
    backing_vocals_path = None
    if karaoke_enabled:
        t = time.time()
        karaoke_out_dir = os.path.join(tmpdir, "karaoke_out")
        lead_path, backing_path = separate_karaoke(vocals_path, karaoke_out_dir)
        vocals_path = lead_path
        backing_vocals_path = backing_path
        karaoke_time = time.time() - t
        print(f"[Infer] Karaoke: {karaoke_time:.1f}s")

    # 4. RVC inference — use RVC WebUI's own VC class (not the broken pypi 'rvc' pkg)
    # VC.get_vc() looks for models in /app/rvc-webui/assets/weights/, so copy ours in.
    print("[Infer] Running RVC inference...")
    weights_dir = os.path.join(RVC_WEBUI_DIR, "assets", "weights")
    os.makedirs(weights_dir, exist_ok=True)
    model_basename = f"rvc_{user_id}.pth"
    shutil.copy(model_path, os.path.join(weights_dir, model_basename))
    has_index = os.path.exists(index_path)
    if has_index:
        # VC.get_vc auto-finds .index inside logs/{exp}/ — give it an absolute path via file_index
        index_arg = index_path
    else:
        index_arg = ""

    output_wav = os.path.join(tmpdir, "rvc_output.wav")

    protect = float(job_input.get("protect", 0.33))
    rms_mix_rate = float(job_input.get("rms_mix_rate", 0.25))
    f0_method = job_input.get("f0_method", "rmvpe")

    infer_script = f"""
import sys, os
sys.path.insert(0, '/app/rvc-webui')
os.chdir('/app/rvc-webui')
# RVC WebUI reads these from .env at import time. We don't go through .env, so set them here.
os.environ['weight_root'] = 'assets/weights'
os.environ['index_root'] = 'logs'
os.environ['rmvpe_root'] = 'assets/rmvpe'
os.environ['outside_index_root'] = 'logs'
try:
    from configs.config import Config
    from infer.modules.vc.modules import VC
    from scipy.io import wavfile

    config = Config()
    vc = VC(config)
    vc.get_vc('{model_basename}')
    # This RVC fork's vc_single returns (info_str, (tgt_sr, audio_opt))
    info, audio_tuple = vc.vc_single(
        0,                    # sid
        '{vocals_path}',      # input_audio_path
        {pitch_shift},        # f0_up_key
        None,                 # f0_file
        '{f0_method}',        # f0_method
        '{index_arg}',        # file_index
        '',                   # file_index2
        {index_rate},         # index_rate
        {filter_radius},      # filter_radius
        0,                    # resample_sr
        {rms_mix_rate},       # rms_mix_rate
        {protect},            # protect
    )
    print(f'RVC info: {{info}}')
    if audio_tuple is None:
        raise RuntimeError(f'vc_single returned None audio: {{info}}')
    tgt_sr, audio_opt = audio_tuple
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

    print(f"[Infer] STDOUT: {result.stdout[-500:]}")
    if result.stderr:
        print(f"[Infer] STDERR: {result.stderr[-500:]}")
    if result.returncode != 0:
        raise RuntimeError(f"RVC inference failed: {result.stderr[-300:]}")

    if not os.path.exists(output_wav):
        raise RuntimeError("RVC produced no output")

    print(f"[Infer] RVC done in {infer_time:.1f}s")

    # 5. Apply FX to vocals ONLY (before mixing) — KTV reverb on vocals,
    #    instrumental stays clean. Volume gain also only affects vocals.
    if vocal_volume != 1.0 or reverb > 0:
        output_wav = apply_post_fx(output_wav, vocal_volume, reverb)

    # 6. Mix processed vocals with original instrumental.
    # If karaoke enabled, merge backing vocals into instrumental first
    if karaoke_enabled and backing_vocals_path and os.path.exists(backing_vocals_path):
        inst_with_backing = os.path.join(tmpdir, "inst_with_backing.wav")
        merge_cmd = [
            "ffmpeg", "-y",
            "-i", instrumental_path,
            "-i", backing_vocals_path,
            "-filter_complex",
            "[0:a][1:a]amix=inputs=2:duration=longest:weights=1 1:normalize=0",
            "-ac", "2", "-ar", "44100",
            inst_with_backing,
        ]
        subprocess.run(merge_cmd, capture_output=True, timeout=120)
        if os.path.exists(inst_with_backing):
            instrumental_path = inst_with_backing
            print(f"[Infer] Backing vocals merged into instrumental")

    print("[Infer] Mixing...")
    final_wav = os.path.join(tmpdir, "final_cover.wav")
    mix_cmd = [
        "ffmpeg", "-y",
        "-i", output_wav,
        "-i", instrumental_path,
        "-filter_complex",
        "[0:a][1:a]amix=inputs=2:duration=longest:weights=1 1:normalize=0",
        "-ac", "2", "-ar", "44100",
        final_wav,
    ]
    subprocess.run(mix_cmd, capture_output=True, timeout=120)

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

    # Check if a trained model exists for the given user_id
    if mode == "check_model":
        user_id = job_input.get("user_id", "").strip()
        if not user_id:
            return {"status": "error", "error": "user_id required"}
        model_path = os.path.join(MODELS_DIR, user_id, "model.pth")
        index_path = os.path.join(MODELS_DIR, user_id, "model.index")
        exists = os.path.exists(model_path)
        size_mb = round(os.path.getsize(model_path) / 1024 / 1024, 2) if exists else 0
        print(f"[CheckModel] user_id={user_id} exists={exists} size={size_mb}MB")
        return {
            "status": "success",
            "user_id": user_id,
            "exists": exists,
            "size_mb": size_mb,
            "has_index": os.path.exists(index_path),
        }

    # Package trained model as zip and return download URL (for Replicate / 第三方 RVC 平台)
    if mode == "download_model":
        user_id = job_input.get("user_id", "").strip()
        if not user_id:
            return {"status": "error", "error": "user_id required"}
        model_path = os.path.join(MODELS_DIR, user_id, "model.pth")
        index_path = os.path.join(MODELS_DIR, user_id, "model.index")
        if not os.path.exists(model_path):
            return {"status": "error", "error": f"Model not found for user {user_id}"}

        import zipfile
        with tempfile.TemporaryDirectory() as ztmpdir:
            zip_name = f"rvc_{user_id}.zip"
            zip_path = os.path.join(ztmpdir, zip_name)
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                # Replicate / AICoverGen 期望文件命名一致：model.pth + model.index
                zf.write(model_path, arcname=f"{user_id}.pth")
                if os.path.exists(index_path):
                    zf.write(index_path, arcname=f"{user_id}.index")
            zip_size_mb = round(os.path.getsize(zip_path) / 1024 / 1024, 2)
            try:
                zip_url = upload_file(zip_path, zip_name)
            except Exception as e:
                return {"status": "error", "error": f"Zip upload failed: {e}"}

        result = {
            "status": "success",
            "user_id": user_id,
            "model_zip_url": zip_url,
            "zip_size_mb": zip_size_mb,
            "has_index": os.path.exists(index_path),
        }
        print(f"[DownloadModel] user_id={user_id} → {zip_url} ({zip_size_mb}MB)")
        return result

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
