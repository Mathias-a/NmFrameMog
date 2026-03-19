---
name: deploy
description: Deploy a challenge solution to Google Cloud Run
disable-model-invocation: true
argument-hint: [challenge-name]
---

Deploy the $ARGUMENTS challenge solution to Cloud Run.

## Steps

1. **Verify quality gates pass**
   ```bash
   uv run ruff check . && uv run ruff format --check . && uv run mypy
   ```

2. **Create Dockerfile** (if not exists)
   - Python 3.13 slim base
   - Install uv and dependencies
   - Copy source code
   - Expose port 8080

3. **Build and deploy**
   ```bash
   gcloud run deploy $ARGUMENTS --source . --region europe-north1 --allow-unauthenticated
   ```

4. **Verify deployment**
   - Check the service URL is accessible
   - Run a basic health check

5. **Report** the public URL for submission to the competition platform.

Refer to `docs/nm-ai/shared/google-cloud.md` for platform-specific guidance.
