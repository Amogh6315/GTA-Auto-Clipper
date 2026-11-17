# GTA Auto Clipper

Find high-energy moments in live gameplay and render centered 9:16 clips with word-timed captions. The clipper runs on your computer. A separate Zernio command can schedule the finished clip to your connected YouTube and Instagram accounts.

The software is free and MIT-licensed. The optional AI workflow guide and prompt pack are a separate digital product.

## What it does

- Searches current YouTube live results for a phrase such as `GTA 6` and prints machine-readable JSON for AI agents.
- Records a live source in rolling chunks, then ranks windows by audio peaks, speech density, and reaction words.
- Starts clips a few seconds before the strongest audio peak (6 seconds by default).
- Keeps the horizontal gameplay centered in the vertical frame; a blurred source frame fills the space above and below.
- Resets audio and video timestamps before burning word-timed captions.
- Saves MP4 and JSON metadata in `output/`. It never posts as part of clipping.
- Schedules one reviewed clip to YouTube and Instagram through Zernio; dry-run is the default.

## Install

Requirements: Python 3.10+, FFmpeg in PATH, and a stable connection for first-run model downloads.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For live-stream search, set a YouTube Data API key in the terminal. Keep keys out of prompts, source files, and GitHub:

```powershell
$env:YOUTUBE_API_KEY = "YOUR_YOUTUBE_DATA_API_KEY"
python find_streams.py "GTA 6" --max 25 --json
```

The YouTube API returns live search candidates in `viewCount` order. It is not a dependable ranking of current live viewers; open a candidate and confirm the stream before use.

## Make clips

Start live capture and keep it running until you press Ctrl+C. The clipper writes every selected highlight to `output/`:

```powershell
python clip.py "https://www.youtube.com/watch?v=VIDEO_ID" --max-seconds 0 --clip-len 30 --lead-in 6 --top 2
```

Important options:

| Option | Default | Purpose |
| --- | ---: | --- |
| `--max-seconds` | `0` | Capture duration; `0` runs until stopped |
| `--chunk` | `90` | Analysis chunk size in seconds |
| `--clip-len` | `30` | Rendered highlight length |
| `--lead-in` | `6` | Seconds before the strongest audio peak |
| `--top` | `1` | Highlights kept from each chunk |
| `--min-score` | `1.5` | Minimum moment score |
| `--output-dir` | `output/` | Custom destination folder |

Test a local video first. Test output goes to `test_out/` and cannot be published:

```powershell
python clip.py --file "C:\videos\gameplay.mp4" --test --clip-len 30
```

## Schedule to YouTube + Instagram

Connect your YouTube account and an Instagram Creator or Business account to Zernio. Set the key and account IDs as private environment variables:

```powershell
$env:ZERNIO_API_KEY = "YOUR_ZERNIO_API_KEY"
$env:ZERNIO_YOUTUBE_ACCOUNT_ID = "YOUR_YOUTUBE_ACCOUNT_ID"
$env:ZERNIO_INSTAGRAM_ACCOUNT_ID = "YOUR_INSTAGRAM_ACCOUNT_ID"
```

Review the clip and its JSON metadata, then mark the metadata `"approved": true`. Preview the schedule first:

```powershell
python zernio_publish.py "output\gta-CLIP-ID" --scheduled-for "2026-11-18T12:00:00+05:30"
```

The dry run sends nothing. To schedule the post to both accounts, add `--publish`:

```powershell
python zernio_publish.py "output\gta-CLIP-ID" --scheduled-for "2026-11-18T12:00:00+05:30" --publish
```

For unattended posting, first make sure the AI agent is restricted to sources you have the right to use and that you have tested its output. Then set `GTA_UNATTENDED_POSTS=1` in the machine's private environment and run with `--publish --unattended`. A scheduled post already accepted by Zernio can publish while your computer is asleep. Discovery and clip rendering still require an awake, connected runner.

The publisher stores the Zernio post ID and an idempotency key in the metadata. If an API call fails, check the Zernio dashboard before retrying.

## Website and Razorpay guide checkout

The Pages site is in `site/`; its Functions create a fixed ₹149 Razorpay order, verify the checkout HMAC and captured payment on the server, and only then issue a short-lived download cookie. The buyer guide is stored as a Cloudflare Pages secret and is not part of the public GitHub repository or static asset directory. See [docs/RAZORPAY-SETUP.md](docs/RAZORPAY-SETUP.md) before configuring test or live keys.

Checkout remains disabled until the Razorpay key ID, key secret, and guide-content secret are configured. Test mode must be used first. A test payment does not collect real money or mean the merchant account has been approved for live payments.

## Use with an AI agent

Give the agent this project folder and ask it to follow [AI_WORKFLOW.md](AI_WORKFLOW.md). It can discover candidates, start the clipper, inspect `output/*.json`, draft metadata, and run the publisher in dry-run mode. It must not put secrets in the prompt. For reusable client instructions, see [AGENTS.md](AGENTS.md).

## Rights and platform notes

This release has no creator allowlist; the user chooses the source. You are responsible for obtaining rights and permissions to record, edit, and publish the footage. Creator credit does not replace permission. YouTube search ranking does not report a verified live concurrent-viewer leaderboard. This independent fan project is not affiliated with Rockstar Games, Take-Two Interactive, YouTube, Instagram, Meta, or Zernio.

## License

MIT. See [LICENSE](LICENSE). No OpenShorts application code or AutoCrop-Vertical code is bundled in this focused release.
