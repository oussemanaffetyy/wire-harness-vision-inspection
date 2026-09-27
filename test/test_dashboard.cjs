const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { test } = require('node:test');
const dir = path.join(__dirname, '../nodered');
const flow = JSON.parse(fs.readFileSync(path.join(dir, 'wire_harness_dashboard_flow.json')));
const historySource = fs.readFileSync(path.join(dir, 'inspection_history.js'), 'utf8');
const vue = fs.readFileSync(path.join(dir, 'dashboard.vue'), 'utf8');
const component = new Function(vue.match(/<script>([\s\S]*?)<\/script>/)[1].replace('export default', 'return'))();
function runner() {
    const values = {};
    const context = { get: k => values[k], set: (k, v) => { values[k] = v; } };
    const fn = new Function('msg', 'context', historySource);
    return msg => structuredClone(fn(msg, context));
}
function message(frame, status = 'OK', source = 'test1.MOV', retain = false) {
    return { topic: 'factory/inspection/status', retain, payload: {
        status, frame_index: frame, source_name: source,
        timestamp: new Date(1780000000000 + frame * 100).toISOString(), details: ['Detail']
    } };
}
test('one FlowFuse base, valid references, standalone flow in sync with sources', () => {
    assert.equal(flow.filter(n => n.type === 'ui-base').length, 1);
    const ids = new Set(flow.map(n => n.id));
    assert.equal(ids.size, flow.length);
    for (const n of flow) {
        for (const id of (n.wires || []).flat()) assert(ids.has(id));
        for (const key of ['group', 'page', 'ui', 'theme', 'broker']) {
            if (n[key] && n.type !== 'mqtt-broker') assert(ids.has(n[key]), `${n.id}: ${key}`);
        }
        assert(!n.type.startsWith('ui_'));
    }
    assert.equal(flow.find(n => n.id === 'whi_history').func, historySource);
    assert.equal(flow.find(n => n.id === 'whi_dashboard').format, vue);
    assert.equal(flow.find(n => n.id === 'whi_dashboard').height, 0);
});
test('consecutive frames make one history entry, verdict/source/restart make new entries', () => {
    const run = runner();
    assert.equal(run(message(0)).payload.history.length, 1);
    assert.equal(run(message(1)).payload.history, undefined);
    assert.equal(run(message(2, 'NOK')).payload.history.length, 2);
    assert.equal(run(message(3, 'NOK', 'test2.MOV')).payload.history.length, 3);
    assert.equal(run(message(0, 'NOK', 'test2.MOV')).payload.history.length, 4);
    assert.equal(run(message(0, 'NOK', 'test2.MOV')), null);
});
test('retained status does not count, first live result does', () => {
    const run = runner();
    assert.equal(run(message(0, 'OK', 'test1.MOV', true)).payload.history, undefined);
    assert.equal(run(message(1)).payload.history.length, 1);
});
test('refresh restores shared history and preserves requesting client', () => {
    const run = runner(); run(message(0)); run(message(1, 'NOK'));
    const reply = run({ topic: 'inspection/request', _client: { socketId: 'browser-2' }, ui_update: { format: 'untrusted' } });
    assert.equal(reply.payload.history.length, 2);
    assert.equal(reply.payload.current.status, 'NOK');
    assert.equal(reply._client.socketId, 'browser-2');
    assert.equal(reply.ui_update, undefined);
});
test('bad payloads ignored and history bounded to 300 entries', () => {
    const run = runner();
    for (const payload of [null, {}, { status: 'NOK' }, { status: 'INDETERMINE', timestamp: new Date().toISOString() }]) {
        assert.equal(run({ topic: 'factory/inspection/status', payload }), null);
    }
    for (let i = 0; i < 400; i++) run(message(i, i % 2 ? 'NOK' : 'OK'));
    const history = run({ topic: 'inspection/request' }).payload.history;
    assert.equal(history.length, 300);
    assert.equal(history[0].frame_index, 399);
    assert.equal(history[299].frame_index, 100);
});
test('IACom imports OK and NOK, groups repeats, skips invalid lines and bounds memory', () => {
    const parse = component.methods.parseJournal;
    const result = parse('# Commentaire\r\n[2026-09-26 10:00:00] - Test OK\r\n[2026-09-26 10:00:01] - Test OK\r\n[2026-09-26 10:00:02] - Test NOK: Clip touche/proche du connecteur : 0.0 px\r\ninvalid');
    assert.equal(result.matched, 3);
    assert.equal(result.skipped, 1);
    assert.equal(result.rows.length, 2);
    assert.equal(result.rows[0].status, 'NOK');
    assert.equal(result.rows[0].detail, 'Clip touche/proche du connecteur : 0.0 px');
    const large = Array.from({ length: 1000 }, (_, i) => `[2026-09-26 10:00:00] - Test ${i % 2 ? 'NOK' : 'OK'}`).join('\n');
    assert.equal(parse(large).rows.length, 300);
    assert.equal(parse('\uFEFF[2026-09-26 10:00:00] - Test OK').matched, 1);
    assert.equal(parse('[2026-09-26 10:00:00] - Test NOK: ' + 'x'.repeat(600)).rows[0].detail.length, 512);
});
test('old results and disconnects are not presented as live', () => {
    const now = Date.now();
    const recent = value => component.methods.recent.call({ now }, value);
    assert.equal(recent(new Date(now - 16000).toISOString()), false);
    assert.equal(recent(new Date(now).toISOString()), true);
    assert.equal(recent('bad'), false);
    assert.equal(component.computed.live.call({ connected: false, recent, current: { timestamp: new Date(now).toISOString() } }), false);
});
test('Windows and Mac source paths expose only the filename', () => {
    assert.equal(component.methods.basename('C:\\Users\\name\\test1.MOV'), 'test1.MOV');
    assert.equal(component.methods.basename('/Users/name/test2.MOV'), 'test2.MOV');
});
