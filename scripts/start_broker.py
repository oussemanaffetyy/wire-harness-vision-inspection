"""Local MQTT broker for the demo; no Mosquitto installation required."""
import asyncio
import logging
from pathlib import Path

import yaml
from amqtt.broker import Broker


async def serve() -> None:
    config_path = Path(__file__).resolve().parents[1] / "config" / "broker.yaml"
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
