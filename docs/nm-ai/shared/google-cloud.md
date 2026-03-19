# Google Cloud Guidance Exposed by the NM i AI Docs Server

These docs were discovered by MCP search even though they were not listed in the initial catalog.

Sources:

- `challenge://google-cloud/overview`
- `challenge://google-cloud/setup`
- `challenge://google-cloud/services`
- `challenge://google-cloud/deploy`

## MCP excerpts

### Overview

- Google Cloud is an official NM i AI 2026 partner.
- Selected teams receive a free GCP project with no credit limits or billing setup.
- Available services include Gemini models, Cloud Run, Vertex AI, and collaboration tools.

### Setup

- Cloud Shell is a free terminal built into the Google Cloud console.
- It comes with Python, git, gcloud CLI, and Docker pre-installed.
- The setup docs also describe Gemini Code Assist, Gemini CLI, and Gemini Cloud Assist.

### Services

- Cloud Run is recommended for **Tripletex** and **Astar Island**.
- Compute Engine is suggested only when a GPU or persistent background server is required.
- Vertex AI, Model Garden, and AI Studio are exposed as Gemini-adjacent tooling options.

### Deploy

- "Two of the four competition tasks — Tripletex and Astar Island — require you to host a public HTTPS endpoint that our validators call. Cloud Run is the easiest way to deploy one."
- Cloud Run summary: it takes a Docker container and gives you a public HTTPS URL, handling scaling and TLS.
- Submission flow excerpt:
  1. Copy the Cloud Run URL.
  2. Go to the submission page for the task.
  3. Paste the URL and submit.
  4. Validators will start calling the endpoint.

## Why this matters

- Tripletex and Astar Island should share deployment primitives where possible.
- Cloud Run is the default hosted target unless the team discovers a hard need for GPUs or long-lived workers.
