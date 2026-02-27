const { app, BrowserWindow, globalShortcut, ipcMain, screen } = require('electron');
const path = require('path');

let win = null;
let socket = null;
let reconnectTimer = null;
let registeredShortcut = null;
let reconnectScheduled = false;

function createWindow() {
  const display = screen.getPrimaryDisplay();
  const width = 700;
  const height = 60;
  const x = Math.round((display.workArea.width - width) / 2 + display.workArea.x);
  const y = display.workArea.y + 40;

  win = new BrowserWindow({
    width,
    height,
    x,
    y,
    frame: false,
    transparent: true,
    resizable: false,
    movable: false,
    alwaysOnTop: true,
    skipTaskbar: true,
    show: false,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  win.loadFile(path.join(__dirname, 'index.html'));
}

function sendToRenderer(message) {
  if (win && !win.isDestroyed()) {
    win.webContents.send('alara:message', message);
  }
}

function showOverlay() {
  if (!win || win.isDestroyed()) return;
  win.show();
  win.focus();
  sendToRenderer({ type: 'overlay_shown' });
}

function hideOverlay() {
  if (!win || win.isDestroyed()) return;
  sendToRenderer({ type: 'overlay_hidden' });
  win.hide();
}

function toggleOverlay() {
  if (!win) return;
  if (win.isVisible()) {
    hideOverlay();
  } else {
    showOverlay();
  }
}

function connectBackend() {
  if (socket && (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING)) {
    return;
  }

  if (reconnectTimer) {
    clearTimeout(reconnectTimer);
    reconnectTimer = null;
  }
  reconnectScheduled = false;

  if (!globalThis.WebSocket) {
    console.error('WebSocket API unavailable in Electron main process');
    return;
  }

  socket = new WebSocket('ws://localhost:8765');

  socket.onopen = () => {
    console.log('Connected to ALARA backend WebSocket');
  };

  socket.onmessage = (event) => {
    try {
      const payload = JSON.parse(event.data);
      if (payload.type === 'wake') {
        showOverlay();
        sendToRenderer({ type: 'state', state: 'listening' });
      }
      sendToRenderer(payload);
    } catch (err) {
      console.error('Invalid backend message:', err);
    }
  };

  socket.onclose = () => {
    socket = null;
    if (!reconnectScheduled) {
      reconnectScheduled = true;
      reconnectTimer = setTimeout(() => {
        reconnectScheduled = false;
        connectBackend();
      }, 2000);
    }
  };

  socket.onerror = (err) => {
    console.warn('Backend WebSocket error:', err?.message || err);
    // Do not close() here; onclose handles reconnect scheduling.
  };
}

function sendToBackend(message) {
  if (message && message.type === 'hide_overlay') {
    hideOverlay();
    return;
  }

  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify(message));
  }
}

app.whenReady().then(() => {
  createWindow();
  connectBackend();

  const preferred = 'Control+Space';
  const preferredOk = globalShortcut.register(preferred, () => {
    toggleOverlay();
  });
  if (preferredOk) {
    registeredShortcut = preferred;
    console.log(`Overlay hotkey registered: ${registeredShortcut}`);
  } else {
    const fallback = 'Control+Shift+Space';
    const fallbackOk = globalShortcut.register(fallback, () => {
      toggleOverlay();
    });
    if (fallbackOk) {
      registeredShortcut = fallback;
      console.warn(`Could not register ${preferred}. Using fallback: ${registeredShortcut}`);
      // Ensure the UI is discoverable when preferred shortcut is blocked.
      showOverlay();
    } else {
      console.error(`Could not register global hotkeys: ${preferred}, ${fallback}`);
      // Last-resort discoverability.
      showOverlay();
    }
  }

  ipcMain.on('alara:send', (_event, message) => {
    sendToBackend(message);
  });

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('will-quit', () => {
  globalShortcut.unregisterAll();
  if (socket) {
    socket.close();
  }
});
