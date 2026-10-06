"""
Story Cut: 30-min Burmese movie recap -> 2-3 min connected 9:16 story video.
Usage: python story_cut.py input.mp4
Env:   GEMINI_API_KEY (required), TARGET_SECONDS=150, GEMINI_MODEL, WATERMARK=KK.Ent
"""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from google import genai
from google.genai import types

VIDEO = sys.argv[1]
TARGET = int(os.environ.get("TARGET_SECONDS", "150"))
MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
WATERMARK = os.environ.get("WATERMARK", "KK.Ent")
CHUNK = 300  # seconds per transcription chunk (keeps timestamps accurate)
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

WORK = Path("work")
OUT = Path("output")
WORK.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
DEVNULL = subprocess.DEVNULL


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, **kw)


def duration(path):
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)])
    return float(out)


def ask(parts, retries=5):
    for i in range(retries):
        try:
            r = client.models.generate_content(
                model=MODEL,
                contents=parts,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json", temperature=0.2),
            )
            return json.loads(r.text)
        except Exception as e:  # rate limit / bad JSON -> wait and retry
            print(f"Gemini retry {i + 1}: {e}")
            time.sleep(15 * (i + 1))
    raise RuntimeError("Gemini failed after retries")


# ---------- 1. Transcribe Burmese narration (chunked) ----------
TRANSCRIBE_PROMPT = (
    "This audio is Burmese narration of a movie recap. Transcribe it in Burmese script. "
    "Return a JSON array of objects {\"start\": number, \"end\": number, \"text\": string}. "
    "start/end are seconds from the beginning of THIS clip. Each item is one complete "
    "sentence or thought of 2-12 seconds. No overlaps. Skip music-only parts."
)


def transcribe():
    cache = WORK / "transcript.json"
    if cache.exists():
        return json.loads(cache.read_text())

    audio = WORK / "audio.mp3"
    run(["ffmpeg", "-y", "-i", VIDEO, "-vn", "-ac", "1", "-ar", "16000",
         "-b:a", "32k", str(audio)], stdout=DEVNULL, stderr=DEVNULL)
    total = duration(audio)

    segs = []
    for k, off in enumerate(range(0, int(total), CHUNK)):
        part = WORK / f"chunk{k}.mp3"
        run(["ffmpeg", "-y", "-ss", str(off), "-t", str(CHUNK), "-i", str(audio),
             "-c", "copy", str(part)], stdout=DEVNULL, stderr=DEVNULL)
        f = client.files.upload(file=str(part))
        while f.state.name == "PROCESSING":
            time.sleep(3)
            f = client.files.get(name=f.name)
        data = ask([f, TRANSCRIBE_PROMPT])
        for d in data:
            s = max(0.0, float(d["start"]))
            e = min(float(d["end"]), CHUNK)
            if e > s and d.get("text", "").strip():
                segs.append({"start": off + s, "end": off + e, "text": d["text"].strip()})
        print(f"chunk {k} done, {len(segs)} segments so far")
        time.sleep(7)  # stay under free-tier rate limit

    segs.sort(key=lambda x: x["start"])
    for i, s in enumerate(segs):
        s["id"] = i
    cache.write_text(json.dumps(segs, ensure_ascii=False, indent=1))
    return segs


# ---------- 2. Pick connected story segments ----------
def select(segs):
    cache = WORK / "selection.json"
    if cache.exists():
        return json.loads(cache.read_text())

    listing = "\n".join(
        f'{s["id"]}|{s["start"]:.1f}-{s["end"]:.1f}|{s["text"]}' for s in segs)
    prompt = f"""You are editing a movie recap into a viral short video.
Below is the full Burmese narration, one line per segment: id|start-end seconds|text

Pick segments so the total duration is about {TARGET} seconds (within +/-15s).
Rules:
- Keep the ORIGINAL ORDER (ids ascending) so the story stays connected and makes sense.
- The result must feel like a complete mini-story: hook, setup, conflict, climax, ending.
- Prefer consecutive segments over scattered ones; avoid jumpy cuts that lose context.
- Start with an attention-grabbing moment, end on a strong or conclusive line.
- Never cut mid-sentence; use whole segments only.
Return JSON: {{"ids": [list of segment ids]}}

{listing}"""
    data = ask([prompt])
    ids = sorted({int(i) for i in data["ids"] if 0 <= int(i) < len(segs)})
    cache.write_text(json.dumps(ids))
    return ids


# ---------- 3. Build cut ranges, snap to silence ----------
def silence_points():
    p = subprocess.run(
        ["ffmpeg", "-i", str(WORK / "audio.mp3"), "-af",
         "silencedetect=noise=-30dB:d=0.25", "-f", "null", "-"],
        stderr=subprocess.PIPE, text=True)
    starts = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", p.stderr)]
    ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", p.stderr)]
    return [(a + b) / 2 for a, b in zip(starts, ends)]


def snap(t, points, tol=0.8):
    near = min(points, key=lambda x: abs(x - t), default=None)
    return near if near is not None and abs(near - t) <= tol else t


def build_ranges(segs, ids):
    ranges = []
    for i in ids:
        s, e = segs[i]["start"], segs[i]["end"]
        if ranges and s - ranges[-1][1] < 0.6:  # merge near-continuous segments
            ranges[-1][1] = e
        else:
            ranges.append([s, e])

    pts = silence_points()
    final, prev_end = [], 0.0
    for s, e in ranges:
        s, e = snap(s, pts), snap(e, pts)
        s = max(s, prev_end)
        if e - s >= 1.0:
            final.append((s, e))
            prev_end = e
    return final


# ---------- 4. Render ----------
def render(ranges):
    vf = ("[0:v]split[a][b];"
          "[a]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=20:5[bg];"
          "[b]scale=1080:-2[fg];"
          "[bg][fg]overlay=(W-w)/2:(H-h)/2")
    if WATERMARK:
        vf += (f",drawtext=text='{WATERMARK}':fontfile={FONT}:fontcolor=white@0.6:"
               "fontsize=38:x=w-tw-30:y=70")
    vf += "[v]"

    listfile = WORK / "list.txt"
    lines = []
    for n, (s, e) in enumerate(ranges):
        d = e - s
        clip = WORK / f"clip{n:03d}.mp4"
        run(["ffmpeg", "-y", "-ss", f"{s:.2f}", "-t", f"{d:.2f}", "-i", VIDEO,
             "-filter_complex", vf, "-map", "[v]", "-map", "0:a:0",
             "-af", f"afade=t=in:d=0.04,afade=t=out:st={max(d - 0.04, 0):.2f}:d=0.04",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-r", "30",
             "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", str(clip)],
            stdout=DEVNULL, stderr=DEVNULL)
        lines.append(f"file '{clip.name}'")
    listfile.write_text("\n".join(lines))

    final = OUT / "story_cut.mp4"
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listfile),
         "-c", "copy", str(final)], stdout=DEVNULL, stderr=DEVNULL)
    return final


if __name__ == "__main__":
    segments = transcribe()
    chosen = select(segments)
    cuts = build_ranges(segments, chosen)
    total = sum(e - s for s, e in cuts)
    print(f"{len(cuts)} cuts, {total:.0f}s total")
    result = render(cuts)
    print("Saved", result)