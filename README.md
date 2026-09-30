# Automated Fruit Sorter

Local web dashboard and camera-driven classifier for a fruit-sorting station. It stores classifications and retained evidence frames in SQLite-backed local storage.

## Deploy with Docker

1. Install Docker Engine and Docker Compose on the machine that can reach the camera stream.
2. Copy `.env.example` to `.env` and set `FRUIT_SORTER_CAMERA_URL` to the camera's MJPEG stream URL. When using the installed command, a camera IP can also be supplied at startup: `fruit-sorter 10.11.219.31` (or `fruit-sorter --camera-ip 10.11.219.31`).
3. Start the service:

   ```sh
   docker compose up -d --build
   ```

4. Open `http://<sorter-host>:8000` and select **Start station**.

The `fruit_sorter_data` Docker volume keeps the SQLite database and evidence frames across container replacements. The service restarts automatically unless explicitly stopped.

## Health checks

- `GET /healthz` confirms that the web service is alive.
- `GET /readyz` confirms that the model files and writable runtime directory are available.

These endpoints are suitable for Docker, a reverse proxy, or an external monitor. A healthy API does not imply that the camera is connected; camera and model state are exposed by `GET /api/status`.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `FRUIT_SORTER_CAMERA_URL` | `http://10.11.222.250:8080/stream` | MJPEG camera source |
| `FRUIT_SORTER_PORT` | `8000` | Host port used by Compose |
| `FRUIT_SORTER_DATA_DIR` | `./data` | SQLite database and captured frames directory |
| `FRUIT_SORTER_FRAME_LIMIT` | `500` | Number of camera frames retained; `0` disables frame retention |
| `FRUIT_SORTER_ENABLE_DEMO` | `false` | Enables the development-only demo API endpoint |

## Run without Docker

Use Python 3.10 or later and install the locked dependencies with `uv sync`, then run:

```sh
uv run uvicorn API.SERVE:app --host 0.0.0.0 --port 8000
```

To start with a camera IP, use the included launcher:

```sh
chmod +x run-sorter.sh
./run-sorter.sh 10.11.221.17
```

It also accepts a full stream URL, such as `./run-sorter.sh http://10.11.221.17:8080/stream`.

For a network-exposed installation, place the dashboard behind an authenticated HTTPS reverse proxy or restrict access to the trusted local network. The control endpoints intentionally start and stop physical processing, so they should not be exposed directly to the public internet.

## Raspberry Pi deployment

Use a **64-bit** Raspberry Pi OS installation on a Pi 4 (4 GB RAM minimum) or Pi 5 (8 GB recommended). The TensorFlow model is CPU-only and model loading/inference will be noticeably slower than on a desktop. A 32-bit OS is not supported by the container's Python/TensorFlow dependencies.

On the Pi, install Docker Engine and the Compose plugin, then enable Docker so the sorter is restored after a reboot:

```sh
sudo systemctl enable --now docker
git clone <your-repository-url> AutomatedFruitSorter
cd AutomatedFruitSorter
cp .env.example .env
nano .env
docker compose up -d --build
```

Set the camera URL in `.env`, then browse to `http://<pi-ip-address>:8000`. Verify the container and probes with:

```sh
docker compose ps
curl -fsS http://localhost:8000/healthz
curl -fsS http://localhost:8000/readyz
```

The Compose `restart: unless-stopped` policy starts the sorter automatically whenever Docker starts. Use `docker compose logs -f` to diagnose camera or model errors, and `docker compose down` to stop it deliberately.
