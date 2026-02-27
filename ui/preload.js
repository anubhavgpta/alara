const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('alara', {
  send: (message) => ipcRenderer.send('alara:send', message),
  onMessage: (callback) => {
    ipcRenderer.on('alara:message', (_event, message) => callback(message));
  },
});
