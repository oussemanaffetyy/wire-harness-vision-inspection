// Node-RED Function node: keep verdict transitions, not one row per video frame.
const key = 'whiInspectionV2';
const state = context.get(key) || { current: null, history: [] };
if (msg.topic === 'inspection/request') {
    return { _client: msg._client, topic: 'inspection/state', payload: { kind: 'inspection', ...state } };
}
if (msg.topic !== 'factory/inspection/status') return null;
const p = msg.payload;
if (!p || !['OK', 'NOK'].includes(p.status) || !Number.isFinite(Date.parse(p.timestamp))) return null;
const current = {
    status: p.status, timestamp: p.timestamp,
    source_name: typeof p.source_name === 'string' ? p.source_name.slice(0, 512) : '',
    frame_index: Number.isInteger(p.frame_index) ? p.frame_index : null,
    detail: String((Array.isArray(p.details) && p.details[0]) || p.note || '').slice(0, 512)
};
const previous = state.current;
const duplicate = previous && current.timestamp === previous.timestamp &&
    current.frame_index === previous.frame_index && current.source_name === previous.source_name && current.status === previous.status;
if (duplicate) return null;
// A retained MQTT message restores the last verdict but is not a new inspection.
const changed = !previous || previous.status !== current.status || previous.source_name !== current.source_name ||
    (current.frame_index !== null && previous.frame_index !== null && current.frame_index < previous.frame_index);
state.current = current;
const added = (changed || !state.history.length) && msg.retain !== true;
if (added) {
    state.history.unshift({ ...current, id: `${current.timestamp}:${current.source_name}:${current.frame_index}` });
    state.history = state.history.slice(0, 300);
}
context.set(key, state);
return { topic: 'inspection/state', payload: {
    kind: 'inspection', current, ...(added ? { history: state.history } : {})
} };
