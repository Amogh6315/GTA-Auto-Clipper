# AI workflow

This is the short public runbook. The paid guide adds setup screenshots, reusable prompts, channel/account setup, and a recurring unattended workflow.

```text
1. Search: python find_streams.py "GTA 6" --max 25 --json
2. Confirm the stream is a source the user is allowed to use.
3. Capture: python clip.py STREAM_URL --max-seconds 0 --clip-len 30 --lead-in 6 --top 2
4. Inspect output/*.mp4 and output/*.json; fix metadata only from evidence in the clip.
5. Preview: python zernio_publish.py output/CLIP_ID --scheduled-for 2026-11-18T12:00:00+05:30
6. Schedule: add --publish only for authorized clips and connected accounts.
```

For sleep-time scheduling, set the Zernio API key and account IDs outside the prompt. A scheduled post can go live while the computer sleeps; discovering streams and rendering clips cannot.

## Standing unattended mode

After testing, a user can set `GTA_UNATTENDED_POSTS=1` in the private environment and grant the agent a standing source-rights policy. The agent must use only authorized creators, use `--unattended`, set a future `--scheduled-for` time, and review Zernio's returned status. It must not use `--now` for unattended runs.
