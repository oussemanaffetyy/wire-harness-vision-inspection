const path = require('path');

module.exports = {
    uiHost: '127.0.0.1',
    flowFile: path.join(__dirname, 'wire_harness_dashboard_flow.json'),
    flowFilePretty: true,
    editorTheme: { projects: { enabled: false } }
};
