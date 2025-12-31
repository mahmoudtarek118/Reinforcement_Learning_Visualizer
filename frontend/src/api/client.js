/**
 * API client for the RL Learning Tool backend.
 */

// Use environment variable for production, fallback to localhost for development
const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';
const API_ROOT = import.meta.env.VITE_API_URL
    ? import.meta.env.VITE_API_URL.replace('/api', '')
    : 'http://localhost:8000';

/**
 * Make API request with error handling.
 */
async function apiRequest(endpoint, options = {}) {
    const url = `${API_BASE}${endpoint}`;

    try {
        const response = await fetch(url, {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        });

        if (!response.ok) {
            const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
            throw new Error(error.detail || `HTTP ${response.status}`);
        }

        return await response.json();
    } catch (error) {
        if (error.name === 'TypeError' && error.message.includes('fetch')) {
            throw new Error('Cannot connect to backend. Is the server running?');
        }
        throw error;
    }
}

/**
 * Create a new training session.
 */
export async function createSession(environment, algorithm) {
    return apiRequest('/sessions', {
        method: 'POST',
        body: JSON.stringify({
            environment: {
                type: environment.type || 'gridworld',
                rows: environment.rows || 4,
                cols: environment.cols || 4,
                max_steps: environment.maxSteps || 100
            },
            algorithm: {
                type: algorithm.type,
                params: algorithm.params || {}
            }
        })
    });
}

/**
 * Get session status.
 */
export async function getSession(sessionId) {
    return apiRequest(`/sessions/${sessionId}`);
}

/**
 * Execute one step.
 */
export async function stepSession(sessionId) {
    return apiRequest(`/sessions/${sessionId}/step`, {
        method: 'POST'
    });
}

/**
 * Run multiple steps.
 */
export async function runSession(sessionId, steps = 100) {
    return apiRequest(`/sessions/${sessionId}/run?steps=${steps}`, {
        method: 'POST'
    });
}

/**
 * Pause session.
 */
export async function pauseSession(sessionId) {
    return apiRequest(`/sessions/${sessionId}/pause`, {
        method: 'POST'
    });
}

/**
 * Resume session.
 */
export async function resumeSession(sessionId) {
    return apiRequest(`/sessions/${sessionId}/resume`, {
        method: 'POST'
    });
}

/**
 * Reset session.
 */
export async function resetSession(sessionId) {
    return apiRequest(`/sessions/${sessionId}/reset`, {
        method: 'POST'
    });
}

/**
 * Delete session.
 */
export async function deleteSession(sessionId) {
    return apiRequest(`/sessions/${sessionId}`, {
        method: 'DELETE'
    });
}

/**
 * Get diagnostics.
 */
export async function getDiagnostics(sessionId) {
    return apiRequest(`/sessions/${sessionId}/diagnostics`);
}

/**
 * Get available environments.
 */
export async function getEnvironments() {
    return apiRequest('/environments');
}

/**
 * Get available algorithms.
 */
export async function getAlgorithms() {
    return apiRequest('/algorithms');
}

/**
 * Health check.
 */
export async function healthCheck() {
    try {
        const response = await fetch(`${API_ROOT}/`);
        return response.ok;
    } catch {
        return false;
    }
}

