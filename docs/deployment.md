# Cloud deployment (AWS EC2)

OvaScope runs on one virtual machine with Docker Compose: the same `docker-compose.yml` as on a laptop
(section 14 of the README). [scripts/deploy_vm.sh](../scripts/deploy_vm.sh) installs and starts everything.
The guide uses AWS EC2; the script also works on Google Cloud, Azure or any Ubuntu 24.04 x86-64 VM.

| Public address | Serves |
|---|---|
| **https://ovascope.mauricenshimyumukiza.world** | web app (React) |
| **https://ovascope.api.mauricenshimyumukiza.world** | REST API, interactive docs at `/docs` |

HTTPS is handled by **Caddy** ([deploy/Caddyfile](../deploy/Caddyfile), added by
[docker-compose.cloud.yml](../docker-compose.cloud.yml)), which gets and renews Let's Encrypt certificates
automatically. `docker-compose.yml` is unchanged, so the laptop setup stays plain HTTP on ports 8080/8000.

## Why a single VM with Docker Compose

| Requirement | What it implies |
|---|---|
| React web app + nginx gateway + 1–3 API containers + Streamlit | several containers → Docker Compose |
| Uploaded images, SQLite database, model versions must survive restarts | persistent disk (bind mounts) |
| Retraining must really run in the cloud | TensorFlow on CPU: 2 vCPU and ~5 GB RAM for the API container |
| Locust comparison of 1/2/3 containers | the API service can be scaled and load-balanced |

Free PaaS tiers (Render, Hugging Face Spaces, Streamlit Cloud) run one small container without a persistent disk,
so retraining and persistence would not work there.

## Architecture

```
                         ┌─ ovascope.<domain>      ──► web (React, nginx) ──/api/*──┐
Internet ──HTTPS :443──► Caddy                                                      ▼
                         └─ ovascope.api.<domain>  ──────────────────────► gateway (nginx, least_conn)
                                                                                    │
                                                         api × N (FastAPI + TensorFlow CPU)
                                                                                    │
          host folders on the EBS disk: models/  db/  data/uploads/  data/bundle/  artifacts/
```

## 1. Launch the EC2 instance (AWS console → EC2 → Launch instance)

| Setting | Value | Why |
|---|---|---|
| Name | `ovascope` | |
| Image (AMI) | **Ubuntu Server 24.04 LTS, 64-bit (x86)** | the script uses `apt`; `tensorflow-cpu` has no ARM build, so not Graviton |
| Instance type | **`m7i-flex.large`** (2 vCPU, 8 GB) or `t3.large` | TensorFlow needs ~5 GB RAM; `t2/t3.micro` (1 GB) are far too small |
| Key pair | create one, download `ovascope.pem` | SSH access |
| Network → security group | see the table below | |
| Storage | **30 GB gp3** (default is 8 GB) | images ≈ 6 GB + 4 GB swap + data |

Security group (inbound rules):

| Type | Port | Source | Purpose |
|---|---|---|---|
| SSH | 22 | My IP | administration |
| HTTP | 80 | My IP → later `0.0.0.0/0` | certificate check by Let's Encrypt, redirect to HTTPS |
| HTTPS | 443 | My IP → later `0.0.0.0/0` | web app and API |
| Custom TCP | 8080 | My IP | direct access for the first sign-in (step 5) |
| Custom TCP | 8501 | optional, My IP | Streamlit |

Open 80 and 443 to everyone only **after** you created your own account (step 5): the first account becomes
the administrator.

`t3` instances are burstable: long load tests or retraining use up CPU credits and then slow down
(unless "unlimited" is on). `m7i-flex.large` does not have that limit for this workload. Check the hourly price and
whether your account's free-tier credits cover the type before launching.

**Static address:** EC2 → Elastic IPs → Allocate → Associate with the instance. Without it the public IP changes
whenever the instance is stopped and started, and the DNS records would point to the wrong server.

## 2. DNS records (at the registrar of mauricenshimyumukiza.world)

Add two **A records** pointing to the Elastic IP:

| Type | Name / host | Value | TTL |
|---|---|---|---|
| A | `ovascope` | `<ELASTIC-IP>` | 300 |
| A | `ovascope.api` | `<ELASTIC-IP>` | 300 |

Check from your computer (may take a few minutes): `dig +short ovascope.mauricenshimyumukiza.world` and
`dig +short ovascope.api.mauricenshimyumukiza.world` must both print the Elastic IP.

## 3. Copy the data bundle (on your computer)

The code and models come from GitHub. Retraining also needs `data/bundle` (validation set + replay sample, not in git):

```bash
python scripts/export_artifacts.py                                 # once, if data/bundle does not exist
tar czf ~/ovascope_bundle.tar.gz -C . data/bundle                  # ≈130 MB
chmod 400 ~/Downloads/ovascope.pem
scp -i ~/Downloads/ovascope.pem ~/ovascope_bundle.tar.gz ubuntu@<ELASTIC-IP>:~
```

## 4. Deploy (on the instance)

```bash
ssh -i ~/Downloads/ovascope.pem ubuntu@<ELASTIC-IP>
curl -fsSL https://raw.githubusercontent.com/N-Maurice/ovascope-MLOP/main/scripts/deploy_vm.sh -o deploy_vm.sh
WEB_DOMAIN=ovascope.mauricenshimyumukiza.world API_DOMAIN=ovascope.api.mauricenshimyumukiza.world \
  bash deploy_vm.sh        # first build ≈10 min; prints the addresses and checks the DNS records
```
The domain names are saved in `~/ovascope-MLOP/.env` (with `COMPOSE_FILE` including docker-compose.cloud.yml),
so later updates are simply `bash deploy_vm.sh` and every `sudo docker compose ...` command includes Caddy.

The script: checks the CPU type and disk space → installs Docker, Compose and git → adds 4 GB swap → clones the
repository → creates `.env` with a random admin key (`grep ADMIN_KEY ~/ovascope-MLOP/.env`) → unpacks the bundle →
saves the domain names → `docker compose up -d --build` (with Caddy) → waits for `/health`.

## 5. First account, then open the site

1. Open `http://<ELASTIC-IP>:8080` → *Create account*. The first account becomes the administrator.
2. Security group: change the source of ports 80 and 443 to `0.0.0.0/0`, and remove the 8080 rule.
3. Caddy now obtains the certificates (about a minute; `sudo docker compose logs caddy` shows
   `certificate obtained successfully`). Open https://ovascope.mauricenshimyumukiza.world and sign in.

## 6. Verify from your computer

```bash
API=https://ovascope.api.mauricenshimyumukiza.world
curl $API/health                                                   # "model_loaded": true
python scripts/evaluate_deployed.py --url $API --per-class 0
locust -f load_testing/locustfile.py --host $API --headless -u 20 -r 5 -t 60s
```

Checklist:
- [ ] both https addresses open with a valid certificate (padlock), and http redirects to https
- [ ] `/health` answers from the public address with the active model loaded
- [ ] a prediction in the web app returns a class
- [ ] uploaded labelled images appear in the admin records
- [ ] retraining finishes and a new model version appears
- [ ] `sudo docker compose restart` keeps uploads, jobs and the active model

## Operations

```bash
cd ~/ovascope-MLOP
sudo docker compose ps                         # status
sudo docker compose logs -f api                # logs
sudo docker compose up -d --scale api=2        # more API containers (each may use up to 5 GB while retraining)
sudo docker compose logs caddy                 # HTTPS certificates
bash ~/deploy_vm.sh                            # update after `git push` (keeps .env, domains, database and models)
sudo docker compose down                       # stop (data stays on the disk)
```

Costs stop only when the instance is **stopped** (the disk and Elastic IP still cost a little) or **terminated**.
Keep it running until the project has been graded.

## Limitations

- SQLite on one host: fine for a demo; several hosts would need PostgreSQL and object storage (S3).
- Retraining runs in the API container that received the request and competes with predictions for CPU.
- Let's Encrypt must reach ports 80/443 to issue and renew the certificates, so those ports stay open to everyone.
