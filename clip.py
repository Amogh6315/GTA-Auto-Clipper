"""Find energetic moments in live gameplay and render centered vertical clips.

This command only records/renders. It never uploads or publishes.
"""
import argparse, glob, json, os, re, shutil, subprocess, sys, time
from datetime import datetime
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(HERE, "output")
TEST_OUT = os.path.join(HERE, "test_out")
WORK = os.path.join(HERE, "work")
DEFAULT_MAX_SECONDS = 0
EXCITED = {"oh", "wow", "no", "what", "bro", "dude", "crazy", "insane", "holy", "wait", "omg",
           "god", "damn", "shit", "fuck", "yes", "yo", "look", "watch", "run", "help", "dead",
           "cops", "police", "crash", "explode", "boom", "chase", "scary", "unbelievable"}


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


# ---------- source info + permission gate ----------
def stream_info(url):
    r = run([sys.executable, "-m", "yt_dlp", "--dump-single-json", "--no-download", "--no-warnings", url])
    if r.returncode:
        sys.exit("yt-dlp could not read that URL:\n" + r.stderr[-400:])
    return json.loads(r.stdout)


def source_identity(info, test=False):
    name = info.get("uploader") or info.get("channel") or "local source"
    return {
        "name": name,
        "credit": info.get("uploader") or info.get("channel") or None,
        "channel_id": info.get("channel_id"),
        "test": bool(test),
    }


# ---------- recording ----------
def start_recorder(url, file_, chunk, max_seconds, outdir):
    os.makedirs(outdir, exist_ok=True)
    pat = os.path.join(outdir, "chunk_%05d.ts")
    if file_:
        inputs = ["-i", file_]
    else:
        r = run([sys.executable, "-m", "yt_dlp", "-g", "--no-warnings", "-f", "bv[height<=720]+ba/b[height<=720]/b", url])
        lines = [l for l in r.stdout.splitlines() if l.startswith("http")]
        if r.returncode or not lines:
            sys.exit("could not get a stream URL:\n" + r.stderr[-400:])
        inputs = [x for u in lines[:2] for x in ("-i", u)]  # video url (+ separate audio url)
    maps = ["-map", "0:v:0", "-map", "1:a:0"] if len(inputs) == 4 else ["-map", "0:v:0", "-map", "0:a:0?"]
    duration = ["-t", str(max_seconds)] if max_seconds > 0 else []
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *inputs, *duration,
           *maps, "-c", "copy", "-f", "segment", "-segment_time", str(chunk),
           "-reset_timestamps", "1", pat]
    return subprocess.Popen(cmd)


# ---------- analysis ----------
_model = None
def whisper_words(audio):
    global _model
    from faster_whisper import WhisperModel
    if _model is None:
        _model = WhisperModel("base.en", device="cpu", compute_type="int8")
    segs, _ = _model.transcribe(audio, word_timestamps=True, vad_filter=True, language="en")
    return [(w.start, w.end, w.word.strip()) for s in segs for w in (s.words or [])]


def load_audio(path):
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", path, "-vn", "-f", "f32le", "-ac", "1", "-ar", "16000", "-"],
                         capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def rms_series(x, hop=0.25):
    n = int(16000 * hop)
    k = len(x) // n
    if k == 0:
        return np.zeros(1)
    return 20 * np.log10(np.sqrt((x[:k * n].reshape(k, n) ** 2).mean(1)) + 1e-6)


def select_clip_start(anchor, lead_in, clip_len, duration):
    latest = max(0.0, duration - clip_len)
    return round(min(max(0.0, anchor - lead_in), latest), 2)


def pick_moments(chunk_path, clip_len, top, min_score, lead_in=6):
    audio = load_audio(chunk_path)
    if len(audio) < 16000:
        return [], []
    db = rms_series(audio)
    words = whisper_words(audio)
    hop = 0.25
    med = np.median(db); mad = np.median(np.abs(db - med)) + 1.0
    spike = (db - med) / mad
    dur = len(db) * hop
    if dur < clip_len:
        return [], words
    cands = []
    for start in np.arange(0, dur - clip_len + 0.1, 5.0):
        a, b = int(start / hop), int((start + clip_len) / hop)
        seg = np.sort(spike[a:b])
        loud = seg[-max(3, len(seg) // 10):].mean()
        ws = [w for w in words if start <= w[0] < start + clip_len]
        density = len(ws) / clip_len
        hits = min(6, sum(1 for w in ws if re.sub(r"\W", "", w[2].lower()) in EXCITED))
        cands.append((0.5 * loud + 2.0 * density + 0.4 * hits, start))
    cands.sort(reverse=True)
    chosen = []
    for score, start in cands:
        if score < min_score or len(chosen) >= top:
            continue
        if any(abs(start - c[1]) < clip_len for c in chosen):
            continue
        peak = (a + int(np.argmax(spike[a:b]))) * hop
        start = select_clip_start(peak, lead_in, clip_len, dur)
        # snap start to the nearest word start within 2 s so we never cut mid-word
        near = [w[0] for w in words if abs(w[0] - start) <= 2]
        chosen.append((round(float(score), 2), float(min(near, key=lambda t: abs(t - start)) if near else start)))
    return chosen, words


# ---------- captions + cut ----------
def ass_time(t):
    h, m, s = int(t // 3600), int(t % 3600 // 60), t % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def write_ass(path, words, start, length, credit):
    head = ("[Script Info]\nScriptType: v4.00+\nPlayResX: 1080\nPlayResY: 1920\n\n[V4+ Styles]\n"
            "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding\n"
            "Style: Word,Arial,120,&H00FFFFFF,&H000000FF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,8,2,2,40,40,330,1\n"
            "Style: Credit,Arial,44,&H00FFFFFF,&H000000FF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,4,1,8,40,40,150,1\n\n"
            "[Events]\nFormat: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n")
    ev = []
    inside = [w for w in words if start <= w[0] < start + length]
    for i, (s, e, w) in enumerate(inside):
        t0 = s - start
        t1 = (inside[i + 1][0] - start) if i + 1 < len(inside) else e - start
        t1 = min(max(t1, t0 + 0.12), t0 + max(0.35, e - s + 0.25))  # don't hold a word through silence
        txt = re.sub(r"[^\w'!?.,-]", "", w).upper()
        if txt:
            ev.append(f"Dialogue: 0,{ass_time(t0)},{ass_time(min(t1, length))},Word,,0,0,0,,{txt}")
    if credit:
        ev.append(f"Dialogue: 1,{ass_time(0)},{ass_time(4)},Credit,,0,0,0,,via {credit}")
    open(path, "w", encoding="utf-8").write(head + "\n".join(ev) + "\n")


def build_video_filter(ass_name):
    return ("[0:v]split[a][b];[a]scale=1080:1920:force_original_aspect_ratio=increase,"
            "crop=1080:1920,boxblur=30:5[bg];[b]scale=1080:-2[fg];"
            "[bg][fg]overlay=(W-w)/2:(H-h)/2,setpts=PTS-STARTPTS,ass=" + ass_name)


def cut(chunk_path, words, start, length, out_mp4, credit):
    d = os.path.dirname(out_mp4)
    ass_name = "cap_" + os.path.basename(out_mp4) + ".ass"
    write_ass(os.path.join(d, ass_name), words, start, length, credit)
    fg = build_video_filter(ass_name)
    r = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{start:.2f}",
                        "-i", os.path.abspath(chunk_path), "-t", f"{length}",
                        "-filter_complex", fg, "-af", "asetpts=PTS-STARTPTS",
                        "-c:v", "libx264", "-preset", "veryfast",
                        "-crf", "21", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart",
                        os.path.basename(out_mp4)], cwd=d, capture_output=True, text=True)
    os.remove(os.path.join(d, ass_name))
    if r.returncode:
        print("ffmpeg cut failed:", r.stderr[-300:])
        return False
    return True


# ---------- main loop ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url", nargs="?")
    ap.add_argument("--file")
    ap.add_argument("--test", action="store_true", help="render to test_out instead of output")
    ap.add_argument("--chunk", type=int, default=90)
    ap.add_argument("--max-seconds", type=int, default=DEFAULT_MAX_SECONDS,
                    help="stop after N seconds; 0 keeps recording until Ctrl+C (default: 0)")
    ap.add_argument("--clip-len", type=int, default=30)
    ap.add_argument("--lead-in", type=float, default=6,
                    help="seconds before the strongest audio peak to start each clip")
    ap.add_argument("--top", type=int, default=1, help="clips per chunk")
    ap.add_argument("--min-score", type=float, default=1.5)
    ap.add_argument("--keep-minutes", type=int, default=30)
    ap.add_argument("--output-dir", default=None, help="override the output folder")
    a = ap.parse_args()
    if not a.url and not a.file:
        sys.exit(__doc__)
    if a.file and not a.test:
        sys.exit("--file is only allowed with --test.")
    if a.max_seconds < 0 or a.chunk < 1 or a.clip_len < 1 or a.top < 1:
        sys.exit("--max-seconds must be 0 or more; --chunk, --clip-len and --top must be positive.")

    if a.url:
        info = stream_info(a.url)
        if not info.get("is_live") and not a.test:
            sys.exit("The source is not live. Use --test for a local/non-live test input.")
        who = source_identity(info, a.test)
        title = info.get("title", "")
    else:
        who, title, info = source_identity({"uploader": "local source"}, True), os.path.basename(a.file), {}
    out_dir = os.path.abspath(a.output_dir) if a.output_dir else (TEST_OUT if who["test"] else OUTPUT_DIR)
    os.makedirs(out_dir, exist_ok=True)
    wdir = os.path.join(WORK, datetime.now().strftime("%Y%m%d-%H%M%S"))
    print(f"source: {title!r} by {who['name']}  ->  {out_dir}")
    rec = start_recorder(a.url, a.file, a.chunk, a.max_seconds, wdir)
    done, n_clips = set(), 0
    try:
        while True:
            chunks = sorted(glob.glob(os.path.join(wdir, "chunk_*.ts")))
            finished = chunks if rec.poll() is not None else chunks[:-1]
            todo = [c for c in finished if c not in done]
            if not todo:
                if rec.poll() is not None:
                    break
                time.sleep(3)
                continue
            c = todo[0]; done.add(c)
            print(f"analysing {os.path.basename(c)} ...")
            moments, words = pick_moments(c, a.clip_len, a.top, a.min_score, a.lead_in)
            for score, start in moments:
                stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                name = f"gta-{stamp}-{n_clips + 1}"
                mp4 = os.path.join(out_dir, name + ".mp4")
                if cut(c, words, start, a.clip_len, mp4, who.get("credit")):
                    n_clips += 1
                    text = " ".join(w[2] for w in words if start <= w[0] < start + a.clip_len)
                    with open(os.path.join(out_dir, name + ".json"), "w", encoding="utf-8") as meta:
                        json.dump({
                            "video": name + ".mp4", "channel": "gta", "source_url": a.url or a.file,
                            "streamer": who["name"], "credit": who.get("credit"), "stream_title": title,
                            "score": score, "transcript": text, "title": None, "youtube_description": None,
                            "instagram_caption": None, "hashtags": [], "post_at": None,
                            "approved": False, "posted": {}, "test": who["test"],
                        }, meta, indent=2, ensure_ascii=False)
                    print(f"  clip {name}.mp4  score {score}")
            for old in glob.glob(os.path.join(wdir, "chunk_*.ts")):
                if old in done and time.time() - os.path.getmtime(old) > a.keep_minutes * 60:
                    os.remove(old)
    except KeyboardInterrupt:
        print("stopping capture; finishing current work ...")
    finally:
        if rec.poll() is None:
            rec.terminate()
            try:
                rec.wait(timeout=10)
            except subprocess.TimeoutExpired:
                rec.kill()
                rec.wait()
        shutil.rmtree(wdir, ignore_errors=True)
    print(f"done. {n_clips} clip(s) in {out_dir}. Nothing posted.")


if __name__ == "__main__":
    main()
