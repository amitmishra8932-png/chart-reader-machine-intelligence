# Staging deployment — NSE Bhavcopy

This branch is a staging candidate only. Do not point the existing production Streamlit app at this branch.

## Create a separate Streamlit Community Cloud app

1. Open https://share.streamlit.io/ and sign in with the GitHub account that can access this repository.
2. Choose **Create app** and select this repository.
3. Set branch to `feature/nse-bhavcopy-integration`.
4. Set main file path to `app.py`.
5. Give the staging app a clearly separate name (for example, `amit-ai-scanner-bhavcopy-staging`).
6. Deploy it without changing the existing production app's repository, branch, or settings.

## Staging-only secrets

For Scanner, set a new staging-only `APP_PASSWORD` in the staging app's **Advanced settings → Secrets**. This is required to unlock the UI. Kite credentials are optional for the Bhavcopy upload test; do not copy production secrets unless separately approved.

For Chart Reader, no secrets are required to test the Bhavcopy upload. 5paisa credentials are optional for this test. If 5paisa OAuth is tested later, register the staging app's callback URL separately with 5paisa and configure staging-only secrets:
- `FIVEPAISA_API_KEY`
- `FIVEPAISA_ENCRYPTION_KEY`
- `FIVEPAISA_USER_ID`
- `FIVEPAISA_REDIRECT_URL`

Never paste secrets into GitHub, source code, this document, or chat.

## Sample import verification

1. Open the staging app's **NSE Bhavcopy** area (Scanner expander or Chart Reader tab).
2. Select trading date **20-Nov-2026**.
3. Upload the official `BhavCopy_NSE_CM_0_0_0_20261120_F_0000.csv.zip`.
4. Import it and confirm **3,742 valid rows**, **0 quarantined rows**, and SHA-256:
   `627df3757f5c6d1aa5ba8f3ad8db623fb5334f9620d64f1cf1a1533835de2ac3`.
5. Repeat the import. The daily table must remain at **3,742** rows; a second ingestion-run audit entry is expected.
6. Confirm the raw ZIP is present under the staging app's own `raw_bhavcopy/2026/11/20/<sha256>.zip` directory.
7. Confirm the other app has a separate database and archive directory. Do not copy database files between apps.

## Persistence and release gate

Streamlit Community Cloud's local container filesystem is not a durable database/archive service. This branch stores the database and raw ZIP locally for controlled staging only. A restart/redeploy may reset those files. Do not rely on this local store as the sole long-term archive; before production use, select and test a persistent storage service and backup/restore procedure.

The upload smoke test validates the parser and the in-app UI path. It does not establish persistent storage, automated NSE download reliability, production readiness, or permission to merge. Keep this PR in draft until the owner reviews staging results and approves the release plan.
