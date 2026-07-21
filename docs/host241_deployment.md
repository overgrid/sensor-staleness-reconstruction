# Deploying the live pipeline to `host241`

Reference: `server_infrastructure_reference.md`. This follows the same
patterns already used for `mlflow` on that server (section 8's checklist)
— own service account, own venv, systemd units with `Restart=always`, and
a plain nginx `location` block since this deliberately skips JupyterHub's
Services/OAuth mechanism.

Not yet run or verified on the real server — this is a from-scratch
runbook to work through step by step, not a script to blindly execute.
Everything here uses port `8091` (next free port after `mlflow_proxy.py`'s
`8090`, per the server doc's port map) and a new `staleness` service
account/group — adjust if either is already taken by something else by
the time you get to this.

---

## 1. Install Docker Engine (not Desktop — this is a headless Ubuntu server)

```bash
# Official Docker apt repo (not Ubuntu's own docker.io package, which lags)
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Headless: no GUI, no "Docker Desktop" — the daemon itself starts on
# boot via systemd, matching how jupyterhub.service and mlflow.service
# already run on this box.
sudo systemctl enable --now docker
sudo systemctl status docker   # confirm it's active
```

## 2. Create the `staleness` service account

Mirrors the `mlflow` user pattern exactly (section 2 of the infra doc):

```bash
sudo useradd -r -m -d /opt/staleness -s /bin/bash staleness

# Let it run docker/docker-compose without sudo:
sudo usermod -aG docker staleness
```

## 3. Clone the repo and set up the venv

```bash
sudo -u staleness -H bash -c '
  cd /opt/staleness
  git clone <your-repo-url> sensor-staleness-reconstruction
  cd sensor-staleness-reconstruction
  python3 -m venv /opt/staleness/venv
  /opt/staleness/venv/bin/pip install -e ".[dev]"
'
```

## 4. Move the Kafka broker here

Same `docker-compose.yml` already in the repo — just run it as the
`staleness` user instead of your laptop:

```bash
sudo -u staleness -H bash -c '
  cd /opt/staleness/sensor-staleness-reconstruction
  docker compose up -d
'
```

`restart: unless-stopped` in that file plus `docker.service` now being
enabled at boot (step 1) means this survives reboots with nothing manual
— the same property your laptop setup never had.

## 5. Systemd unit for the producer (`staleness simulate`)

`/etc/systemd/system/staleness-simulate.service`:

```ini
[Unit]
Description=Staleness pipeline — simulated live sensor feed (Kafka producer)
After=docker.service
Requires=docker.service

[Service]
Type=simple
User=staleness
WorkingDirectory=/opt/staleness/sensor-staleness-reconstruction
ExecStart=/opt/staleness/venv/bin/staleness simulate \
  --columns "285d4816a398__Air_Temperature_Sensor__aht_temperature" \
  --point-id "285d4816a398" \
  --speed 1000 \
  --loop
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

(Adjust `--columns`/`--point-id` to whatever you're simulating; add more
`staleness simulate` instances — same unit file, different name — for
additional sensors/points if you want several streaming at once.)

## 6. Systemd unit for the dashboard

`/etc/systemd/system/staleness-dashboard.service`:

```ini
[Unit]
Description=Staleness pipeline — live dashboard
After=docker.service staleness-simulate.service
Requires=docker.service

[Service]
Type=simple
User=staleness
WorkingDirectory=/opt/staleness/sensor-staleness-reconstruction
Environment=STALENESS_KAFKA_BOOTSTRAP_SERVERS=localhost:9092
Environment=STALENESS_KAFKA_TOPIC=sensor-readings
ExecStart=/opt/staleness/venv/bin/staleness dashboard --host 127.0.0.1 --port 8091
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable and start both:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now staleness-simulate.service
sudo systemctl enable --now staleness-dashboard.service
sudo systemctl status staleness-simulate staleness-dashboard
```

First dashboard startup will download the ~100MB Chronos model into
`staleness`'s HF cache (`/opt/staleness/.cache/huggingface` by default) —
watch `journalctl -u staleness-dashboard -f` for that on first boot.

## 7. nginx — new `location` block (not a JupyterHub Service)

Per section 6 of the infra doc, this is exactly the case for a new
top-level `location` block rather than registering as a JupyterHub
Service: it's unrelated to notebooks and you specifically don't want Hub
OAuth gating it.

Add inside the existing `server { listen 443 ... }` block in
`/etc/nginx/sites-available/jupyterhub`, **before** the catch-all
`location /` block (nginx matches the most specific prefix, but keeping
more specific blocks above the catch-all avoids any ordering surprises):

```nginx
location /staleness/ {
    auth_basic           "Staleness dashboard";
    auth_basic_user_file /etc/nginx/.htpasswd-staleness;

    proxy_pass http://127.0.0.1:8091/;   # trailing slash strips the /staleness/ prefix
    proxy_http_version 1.1;

    # Required for the WebSocket (/staleness/ws) to work through nginx —
    # without these, the browser's WebSocket upgrade request just gets
    # proxied as a normal HTTP request and fails.
    proxy_set_header Upgrade    $http_upgrade;
    proxy_set_header Connection "upgrade";

    proxy_set_header Host              $host;
    proxy_set_header X-Real-IP         $remote_addr;
    proxy_set_header X-Forwarded-For   $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;

    proxy_read_timeout 3600s;   # keep long-lived WebSocket connections open
}
```

Create the Basic Auth password file (you'll be prompted for a password):

```bash
sudo apt-get install -y apache2-utils
sudo htpasswd -c /etc/nginx/.htpasswd-staleness <username-you-want>
# -c creates the file; omit -c to ADD more users to an existing file
```

Reload nginx:

```bash
sudo nginx -t   # validate syntax first
sudo systemctl reload nginx
```

Dashboard is now at `https://jupyter.overgrid.eu/staleness/` — note the
**trailing slash**. Because `proxy_pass` here strips the `/staleness/`
prefix (unlike JupyterHub's CHP+Services routing, which does NOT strip
it — that's specifically why MLflow needed `--static-prefix`), and
because `dashboard_static/index.html`'s JS builds every URL relative to
the page's own path rather than hardcoding a leading `/`, no app-side
prefix flag is needed here. But the trailing slash on the URL itself
still matters — visiting `/staleness` without it will have the browser
resolve `api/series` against the wrong base path.

## 8. Verify end to end

```bash
# Broker + producer + dashboard process all alive:
sudo systemctl status staleness-simulate staleness-dashboard docker

# Kafka actually has data flowing:
sudo -u staleness docker exec -it staleness-kafka \
  /opt/kafka/bin/kafka-console-consumer.sh --bootstrap-server localhost:9092 \
  --topic sensor-readings --from-beginning --max-messages 5
```

Then load `https://jupyter.overgrid.eu/staleness/` in a browser, enter
the Basic Auth credentials from step 7, and confirm the chart appears and
updates live.

## Notes / things intentionally left open

- **No TLS/cert changes needed** — this rides on the existing
  Let's Encrypt cert for `jupyter.overgrid.eu`, since it's just a new
  `location` block on the same `server { listen 443 }` block.
- **Basic Auth is intentionally simple**, per your call — it's one
  shared username/password, not per-user like JupyterHub's PAM login. If
  you later want per-person access or audit logging, that's the point
  where moving to a real JupyterHub Service (with `HubOAuth`, like
  `mlflow_proxy.py`) would start to pay off — not needed now.
- **`generate_synthetic_series()` is still a stub** (see
  `kafka_producer.py`) — the systemd producer unit above only replays
  real CSV data, same as your laptop testing so far.
