"""Local MQTT broker for the demo; no Mosquitto installation required."""
import asyncio
import logging
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.3):
            return True
    except OSError:
        return False


@contextmanager
def local_broker(config: dict, *, enabled: bool = True):
    """Start the demo broker if needed; never stop a broker we did not start."""
    if not enabled or not config.get("enabled", True):
        yield
        return

    broker_config = config.get("broker", {})
    host = broker_config.get("host", "127.0.0.1")
    port = int(broker_config.get("port", 1883))
    if host not in ("127.0.0.1", "localhost") or port != 1883:
        print(f"MQTT personnalise : {host}:{port} (courtier a demarrer separement).", flush=True)
        yield
        return
    if port_open(host, port):
        print(f"Port MQTT {host}:{port} deja ouvert : utilisation du service existant.", flush=True)
        yield
        return

    config_path = PROJECT_ROOT / "config" / "broker.yaml"
    local_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if local_config["listeners"]["default"]["bind"] != "127.0.0.1:1883":
        raise ValueError("Demarrer scripts/start_broker.py separement pour ce config/broker.yaml personnalise.")

    print("Demarrage automatique du courtier MQTT local...", flush=True)
    process = subprocess.Popen(
        [sys.executable, "-u", str(PROJECT_ROOT / "scripts" / "start_broker.py")],
        cwd=PROJECT_ROOT,
        stdin=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 15
        while True:
            if process.poll() is not None:
                raise RuntimeError("Le courtier MQTT n'a pas demarre. Consulter le message ci-dessus.")
            if port_open(host, port):
                break
            if time.monotonic() >= deadline:
                raise RuntimeError("Delai de demarrage MQTT depasse (15 secondes).")
            time.sleep(0.1)
        yield
    finally:
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


async def serve() -> None:
    from amqtt.broker import Broker

    config_path = PROJECT_ROOT / "config" / "broker.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    broker = Broker(config)
    await broker.start()
    address = config["listeners"]["default"]["bind"]
    print(f"Courtier MQTT actif sur {address}. Ctrl+C pour quitter.", flush=True)
    try:
        await asyncio.Event().wait()
    finally:
        await broker.shutdown()


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        print("Courtier arrete.")
    except Exception as exc:
        raise SystemExit(f"Impossible de demarrer MQTT : {exc}. Si le port 1883 est deja utilise, "
                         "conserver le courtier existant ; ne pas en lancer un deuxieme.") from exc
