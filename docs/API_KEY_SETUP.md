# CricAPI Key Setup

The fixture auto-fill feature needs a free CricAPI key from [cricketdata.org/signup](https://cricketdata.org/signup.aspx).

**Never commit your real API key to GitHub.**

## Local development (recommended)

Edit this file on your machine:

```
.streamlit/secrets.toml
```

```toml
CRICAPI_KEY = "your_actual_key_here"
```

This file is gitignored and will not be pushed to GitHub.

## Alternative: `.env` file

Create a `.env` file in the project root (also gitignored):

```
CRICAPI_KEY=your_actual_key_here
```

## Streamlit Cloud / hosted deployment

On [Streamlit Community Cloud](https://streamlit.io/cloud), open your app settings and paste the same TOML into **Secrets**:

```toml
CRICAPI_KEY = "your_actual_key_here"
```

Other hosts (Railway, Render, etc.): set `CRICAPI_KEY` as an environment variable in the platform dashboard.

## Verify

Run the app:

```bash
streamlit run app/main_dashboard.py
```

Pick a date in the sidebar. If the key is configured, the **Select Match** dropdown will load fixtures instead of showing the setup message.

## Fixture cache

To stay within the free 100 requests/day limit, fixtures use a two-layer cache:

| File | Committed to Git? | Purpose |
|------|-------------------|---------|
| `data/fixtures_cache_bundled.json` | Yes | Shipped with every deploy; keeps the match dropdown available after Streamlit Cloud sleep/restart |
| `data/fixtures_cache.json` | No | Runtime overlay refreshed from CricAPI (lost when the cloud container restarts) |

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
