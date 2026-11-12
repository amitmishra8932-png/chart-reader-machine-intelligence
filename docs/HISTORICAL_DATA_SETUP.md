# 5paisa Historical Data Centre setup

## Streamlit Secrets

Add these values in Streamlit Community Cloud → app → Settings → Secrets. Do not put credentials in GitHub files.

```toml
FIVEPAISA_API_KEY = "your 5paisa API key"
FIVEPAISA_ENCRYPTION_KEY = "your 5paisa encryption key"
FIVEPAISA_USER_ID = "your 5paisa user ID"
FIVEPAISA_REDIRECT_URL = "https://your-exact-streamlit-app-url/"
```

The adapter also accepts common aliases such as `API_KEY`, `APP_KEY`, `ENCRYPTION_KEY`, and `USER_ID`. If you already saved those values, do not duplicate them unnecessarily; add the callback URL secret.

## Register the callback URL

In the 5paisa Xstream app configuration, set Redirect URL to the exact same URL used for `FIVEPAISA_REDIRECT_URL`. Use the active Streamlit deployment URL, including its scheme and path. Do not add a comma or other punctuation.

## Daily login

1. Open Chart Reader and select **Authenticate with 5paisa** in the sidebar.
2. Complete the login on the official 5paisa page using the required client code, PIN and OTP/TOTP.
3. 5paisa redirects to Chart Reader with a short-lived `RequestToken`.
4. Chart Reader exchanges it for the daily AccessToken.
5. The AccessToken remains in Streamlit session state only and is not written to GitHub or disk. Authenticate again when it expires or the app session is lost.

## Fetching candles

Open **Historical Data Centre**. Choose a date range of six months or less, select up to 10 NIFTY 50 stocks per batch, and keep both 5m and 15m selected. The app fetches the NSE equity scrip master to resolve each symbol to its 5paisa ScripCode, then fetches historical candles sequentially. Results remain in the current app session and can be downloaded as CSV.

This module only calls authentication, scrip-master and historical market-data endpoints. It does not call order placement, modification or cancellation endpoints.

## Important

The NIFTY 50 symbol list in this build is dated 28-Oct-2026 and should be refreshed after NSE Indices constituent reviews. Intraday historical API requests are limited to six months per request. API permissions, returned history and rate limits are controlled by 5paisa.
