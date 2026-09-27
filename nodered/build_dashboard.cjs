// Embed readable sources into the standalone importable flow.
const fs = require('node:fs');
const path = require('node:path');
const file = path.join(__dirname, 'wire_harness_dashboard_flow.json');
const flow = JSON.parse(fs.readFileSync(file, 'utf8'));
flow.find(n => n.id === 'whi_dashboard').format = fs.readFileSync(path.join(__dirname, 'dashboard.vue'), 'utf8');
flow.find(n => n.id === 'whi_history').func = fs.readFileSync(path.join(__dirname, 'inspection_history.js'), 'utf8');
fs.writeFileSync(file, JSON.stringify(flow, null, 2) + '\n');
console.log('Flow FlowFuse synchronise : ' + file);
