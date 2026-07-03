# CricAPI Key Setup

The fixture auto-fill feature loads schedules from **CricAPI** and/or **Sportmonks**.
Configure at least one key (both recommended for the widest match coverage).

**Never commit your real API keys to GitHub.**

## Local development (recommended)

Edit this file on your machine:

```
.streamlit/secrets.toml
```

```toml
CRICAPI_KEY = "your_cricapi_key_here"
SPORTMONKS_API_KEY = "your_sportmonks_key_here"
```

This file is gitignored and will not be pushed to GitHub.

You can reuse the same `SPORTMONKS_API_KEY` from your `Cricket_KP` project.

## Alternative: `.env` file

Create a `.env` file in the project root (also gitignored):

```
CRICAPI_KEY=your_cricapi_key_here
SPORTMONKS_API_KEY=your_sportmonks_key_here
```

## Streamlit Cloud / hosted deployment

On [Streamlit Community Cloud](https://streamlit.io/cloud), open your app settings and paste the same TOML into **Secrets**:

```toml
CRICAPI_KEY = "your_cricapi_key_here"
SPORTMONKS_API_KEY = "your_sportmonks_key_here"
```

Other hosts (Railway, Render, etc.): set `CRICAPI_KEY` and `SPORTMONKS_API_KEY` as environment variables in the platform dashboard.

## Verify

Run the app:

```bash
streamlit run app/main_dashboard.py
```

Pick a date in the sidebar. If at least one key is configured, the **Select Match** dropdown will load fixtures instead of showing the setup message.

## Fixture cache

To stay within API limits, fixtures use a two-layer cache:

| File | Committed to Git? | Purpose |
|------|-------------------|---------|
| `data/fixtures_cache_bundled.json` | Yes | Shipped with every deploy; keeps the match dropdown available after Streamlit Cloud sleep/restart |
| `data/fixtures_cache.json` | No | Runtime overlay refreshed from APIs (lost when the cloud container restarts) |

### How it works

1. On app load, the dashboard reads the **runtime cache** if present; otherwise it uses the **bundled cache** from the repo.
2. The match dropdown is available immediately, even on a cold Streamlit Cloud start.
3. If the cache is older than **24 hours** (or the date window is out of date), the **first user visit** triggers a CricAPI refresh (~8 requests).
4. That refresh updates only the runtime cache file on the server.
5. Normal date changes in the dashboard read from cache and do not consume API quota.

### Update bundled cache before deploy

Run locally, then commit the bundled file so Streamlit Cloud ships with fresh fixtures:

```bash
python scripts/update_bundled_fixtures.py
git add data/fixtures_cache_bundled.json
git commit -m "Update bundled cricket fixtures"
```

You do not need a scheduled cloud job. Refresh still happens when a user opens the app, but the dropdown no longer starts empty after sleep because the bundled file is part of the deployment.
