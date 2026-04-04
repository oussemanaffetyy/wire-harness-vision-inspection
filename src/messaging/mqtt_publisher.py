from __future__ import annotations

import json
import logging
from typing import Any

import paho.mqtt.client as mqtt


class MqttPublisher:
    def __init__(self, config: dict[str, Any], logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger
        self.client: mqtt.Client | None = None
        self.connected = False

        broker_cfg = config.get("broker", {})
        publisher_cfg = config.get("publisher", {})

        self.enabled = bool(config.get("enabled", True))
        self.host = broker_cfg.get("host", "127.0.0.1")
        self.port = int(broker_cfg.get("port", 1883))
        self.keepalive = int(broker_cfg.get("keepalive", 60))
        self.client_id = broker_cfg.get("client_id", "wire-harness-inspector")
        self.topics = config.get("topics", {})
        self.qos = int(publisher_cfg.get("qos", 0))
        self.retain_status = bool(publisher_cfg.get("retain_status", True))
        self.retain_metrics = bool(publisher_cfg.get("retain_metrics", True))

    def connect(self) -> None:
        if not self.enabled:
            self.logger.info("MQTT is disabled in config.")
            return

        try:
            self.client = mqtt.Client(client_id=self.client_id)
            self.client.connect(self.host, self.port, self.keepalive)
            self.client.loop_start()
            self.connected = True
            self.logger.info("Connected to MQTT broker at %s:%s", self.host, self.port)
        except Exception as exc:  # pragma: no cover - network failures depend on environment
            self.connected = False
            self.client = None
            self.logger.warning(
                "MQTT connection failed: %s. Expected a broker at %s:%s. "
                "Node-RED is not an MQTT broker by itself; start Mosquitto or another local broker first.",
                exc,
                self.host,
                self.port,
            )

    def publish(self, topic_key: str, payload: dict[str, Any]) -> bool:
        topic = self.topics.get(topic_key)
        if not self.enabled or not topic:
            return False
        if not self.client or not self.connected:
            return False

        retain = topic_key in {"status", "metrics"} and (
            (topic_key == "status" and self.retain_status)
            or (topic_key == "metrics" and self.retain_metrics)
        )

        message = json.dumps(payload)
        result = self.client.publish(topic, message, qos=self.qos, retain=retain)
        return result.rc == mqtt.MQTT_ERR_SUCCESS

    def close(self) -> None:
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
        self.client = None
        self.connected = False
