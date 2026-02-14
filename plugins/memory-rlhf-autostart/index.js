/**
 * SillyTavern Server Plugin - Auto-starts the Memory & RLHF Python backend
 * Location: SillyTavern/plugins/memory-rlhf-autostart/index.js
 */
const { spawn } = require('child_process');
const path = require('path');
const http = require('http');

let backendProcess = null;

/**
 * Check if backend is already running
 */
function checkBackendRunning() {
    return new Promise((resolve) => {
        const req = http.get('http://127.0.0.1:5125/health', (res) => {
            resolve(res.statusCode === 200);
        });
        req.on('error', () => resolve(false));
        req.setTimeout(2000, () => {
            req.destroy();
            resolve(false);
        });
    });
}

/**
 * Start the Python backend process
 */
async function startBackend() {
    // Check if already running
    const running = await checkBackendRunning();
    if (running) {
        console.log('[Memory-RLHF AutoStart] Backend already running on port 5125');
        return true;
    }

    console.log('[Memory-RLHF AutoStart] Starting Python backend...');

    // Path to the backend
    const extensionDir = path.join(
        __dirname, '..', '..', 'public', 'scripts', 'extensions',
        'third-party', 'Advanced_Memory_and_RLHF', 'backend'
    );

    const startScript = path.join(extensionDir, 'start_backend.py');

    // Try different Python commands
    const pythonCommands = process.platform === 'win32'
        ? ['py', 'python', 'python3']
        : ['python3', 'python'];

    for (const pyCmd of pythonCommands) {
        try {
            backendProcess = spawn(pyCmd, [startScript], {
                cwd: extensionDir,
                stdio: ['pipe', 'pipe', 'pipe'],
                detached: false,
                env: { ...process.env, PYTHONUNBUFFERED: '1' },
                windowsHide: true,
            });

            // Log stdout
            backendProcess.stdout.on('data', (data) => {
                const lines = data.toString().trim().split('\n');
                lines.forEach(line => {
                    if (line.trim()) {
                        console.log(`[Memory-RLHF Backend] ${line.trim()}`);
                    }
                });
            });

            // Log stderr (but filter noise)
            backendProcess.stderr.on('data', (data) => {
                const msg = data.toString().trim();
                // Filter out common non-error messages from uvicorn
                if (msg && !msg.includes('INFO:') && !msg.includes('WARN')) {
                    console.error(`[Memory-RLHF Backend] ${msg}`);
                }
            });

            backendProcess.on('error', (err) => {
                console.error(`[Memory-RLHF AutoStart] Process error: ${err.message}`);
                backendProcess = null;
            });

            backendProcess.on('close', (code) => {
                if (code !== 0 && code !== null) {
                    console.log(`[Memory-RLHF AutoStart] Backend exited with code ${code}`);
                }
                backendProcess = null;
            });

            // Wait for backend to become ready
            console.log(`[Memory-RLHF AutoStart] Waiting for backend (using ${pyCmd})...`);
            let retries = 30; // 30 seconds max
            while (retries > 0) {
                await new Promise(r => setTimeout(r, 1000));
                const ready = await checkBackendRunning();
                if (ready) {
                    console.log('[Memory-RLHF AutoStart] Backend is ready!');
                    return true;
                }
                // Check if process died
                if (backendProcess === null || backendProcess.killed) {
                    break;
                }
                retries--;
            }

            // If we get here, this python command didn't work
            if (backendProcess && !backendProcess.killed) {
                backendProcess.kill();
                backendProcess = null;
            }

        } catch (err) {
            // This python command not found, try next
            continue;
        }
    }

    console.error('[Memory-RLHF AutoStart] Failed to start backend. Start manually: py backend/start_backend.py');
    return false;
}

/**
 * Stop the backend process
 */
function stopBackend() {
    if (backendProcess && !backendProcess.killed) {
        console.log('[Memory-RLHF AutoStart] Stopping backend...');
        if (process.platform === 'win32') {
            spawn('taskkill', ['/pid', String(backendProcess.pid), '/f', '/t']);
        } else {
            backendProcess.kill('SIGTERM');
        }
        backendProcess = null;
    }
}

// ============================================================
// SillyTavern Plugin Interface
// ============================================================

module.exports = {
    init: async function (router) {
        console.log('[Memory-RLHF AutoStart] Plugin loading...');
        await startBackend();
        return true;
    },

    exit: async function () {
        stopBackend();
    },

    info: {
        id: 'memory-rlhf-autostart',
        name: 'Memory & RLHF Backend AutoStart',
        description: 'Automatically starts the Python backend for the Advanced Memory & RLHF extension',
    },
};