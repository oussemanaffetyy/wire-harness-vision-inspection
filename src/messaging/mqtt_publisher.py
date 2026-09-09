from __future__ import annotations

import json
import logging
from threading import Event
from typing import Any

import paho.mqtt.client as mqtt


class MqttPublisher:
    def __init__(self, config: dict[str, Any], logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger
        self.client: mqtt.Client | None = None
        self.connected = False
        self._ready = Event()
        self._last_publish: mqtt.MQTTMessageInfo | None = None

        broker_cfg = config.get("broker", {})
        publisher_cfg = config.get("publisher", {})

        self.enabled = bool(config.get("enabled", True))
        self.host = broker_cfg.get("host", "127.0.0.1")
        self.port = int(broker_cfg.get("port", 1883))
        self.keepalive = int(broker_cfg.get("keepalive", 60))
        self.client_id = broker_cfg.get("client_id", "wire-harness-python")
        self.topics = config.get("topics", {})
        self.qos = int(publisher_cfg.get("qos", 0))
        self.retain_status = bool(publisher_cfg.get("retain_status", True))
        self.retain_metrics = bool(publisher_cfg.get("retain_metrics", True))
        self.retain_video_stream = bool(publisher_cfg.get("retain_video_stream", True))

    def _on_connect(self, client, userdata, flags, reason_code, properties) -> None:
        self.connected = not reason_code.is_failure
        if self.connected:
            self._ready.set()
            self.logger.info("Connected to MQTT broker at %s:%s", self.host, self.port)
        else:
            self._ready.clear()
            self.logger.warning("MQTT broker rejected connection: %s", reason_code)

    def _on_disconnect(self, client, userdata, flags, reason_code, properties) -> None:
        self.connected = False
        self._ready.clear()
        if reason_code.is_failure:
            self.logger.warning("MQTT disconnected: %s; retrying in background.", reason_code)

    def connect(self) -> None:
        if not self.enabled:
            self.logger.info("MQTT is disabled in config.")
            return

        try:
            self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=self.client_id)
            self.client.on_connect = self._on_connect
            self.client.on_disconnect = self._on_disconnect
            self.client.reconnect_delay_set(min_delay=1, max_delay=10)
            self.client.max_queued_messages_set(20)
            self.client.connect_async(self.host, self.port, self.keepalive)
            self.client.loop_start()
            if not self._ready.wait(timeout=3):
                self.logger.warning("MQTT not connected to %s:%s yet; retrying in background. "
                                    "Start a broker (Node-RED alone is not a broker).", self.host, self.port)
        except Exception as exc:  # pragma: no cover - network failures depend on environment
            self.close()
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

        retain = topic_key in {"status", "metrics", "video_stream"} and (
            (topic_key == "status" and self.retain_status)
            or (topic_key == "metrics" and self.retain_metrics)
            or (topic_key == "video_stream" and self.retain_video_stream)
        )

        message = json.dumps(payload)
        result = self.client.publish(topic, message, qos=self.qos, retain=retain)
        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            self._last_publish = result
        return result.rc == mqtt.MQTT_ERR_SUCCESS

    def close(self) -> None:
        if self.client:
            if self.connected and self._last_publish is not None:
                try:
                    self._last_publish.wait_for_publish(timeout=2)
                except RuntimeError:
                    pass
            self.client.disconnect()
            self.client.loop_stop()
        self.client = None
        self.connected = False
        self._ready.clear()
        self._last_publish = None
