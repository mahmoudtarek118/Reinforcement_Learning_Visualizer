/**
 * Environment configuration for the RL Learning Tool.
 * Defines available environments and their compatibility with algorithms.
 */

// Available environments
export const ENVIRONMENTS = {
    gridworld: {
        id: 'gridworld',
        name: 'GridWorld',
        description: 'Simple deterministic grid navigation',
        defaultConfig: {
            rows: 4,
            cols: 4,
            maxSteps: 100
        },
        configFields: [
            { name: 'rows', label: 'Rows', type: 'number', min: 2, max: 10, default: 4 },
            { name: 'cols', label: 'Columns', type: 'number', min: 2, max: 10, default: 4 },
            { name: 'maxSteps', label: 'Max Steps', type: 'number', min: 10, max: 500, default: 100 }
        ]
    },
    cliffwalking: {
        id: 'cliffwalking',
        name: 'Cliff Walking',
        description: 'Grid with dangerous cliff region',
        available: false, // Coming soon
        defaultConfig: {
            width: 12,
            height: 4
        },
        configFields: [
            { name: 'width', label: 'Width', type: 'number', min: 6, max: 20, default: 12 },
            { name: 'height', label: 'Height', type: 'number', min: 3, max: 10, default: 4 }
        ]
    },
    frozenlake: {
        id: 'frozenlake',
        name: 'Frozen Lake',
        description: 'Slippery frozen lake with holes',
        available: true,  // Now enabled!
        defaultConfig: {
            rows: 4,
            cols: 4,
            maxSteps: 100,
            mapName: '4x4',
            isSlippery: true
        },
        configFields: [
            { name: 'isSlippery', label: 'Slippery', type: 'boolean', default: true }
        ],
        // FrozenLake-specific: define holes and goal positions
        holes: [5, 7, 11, 12],  // States that are holes
        goal: 15                 // Goal state
    }
};

// Algorithm categories and their compatible environments
export const ALGORITHM_ENVIRONMENTS = {
    // Dynamic Programming (model-based) - require known dynamics
    'policy-evaluation': {
        compatible: ['gridworld', 'frozenlake'],
        requiresModel: true,
        description: 'Requires known transition dynamics'
    },
    'policy-iteration': {
        compatible: ['gridworld', 'frozenlake'],
        requiresModel: true,
        description: 'Requires known transition dynamics'
    },
    'value-iteration': {
        compatible: ['gridworld', 'frozenlake'],
        requiresModel: true,
        description: 'Requires known transition dynamics'
    },

    // Model-free methods - learn from experience
    'monte-carlo': {
        compatible: ['gridworld', 'cliffwalking', 'frozenlake'],
        requiresModel: false,
        requiresEpisodic: true,
        description: 'Works with any episodic environment'
    },
    'td-learning': {
        compatible: ['gridworld', 'cliffwalking', 'frozenlake'],
        requiresModel: false,
        description: 'Works with any environment'
    },
    'sarsa': {
        compatible: ['gridworld', 'cliffwalking', 'frozenlake'],
        requiresModel: false,
        description: 'On-policy, works with any environment'
    },
    'q-learning': {
        compatible: ['gridworld', 'cliffwalking', 'frozenlake'],
        requiresModel: false,
        description: 'Off-policy, works with any environment'
    }
};

/**
 * Get compatible environments for an algorithm.
 * @param {string} algorithmId - The algorithm identifier
 * @param {boolean} onlyAvailable - If true, filter out unavailable environments
 * @returns {Array} List of compatible environment objects
 */
export function getCompatibleEnvironments(algorithmId, onlyAvailable = false) {
    const algorithmConfig = ALGORITHM_ENVIRONMENTS[algorithmId];

    if (!algorithmConfig) {
        console.warn(`Unknown algorithm: ${algorithmId}`);
        return [];
    }

    return algorithmConfig.compatible
        .map(envId => ENVIRONMENTS[envId])
        .filter(env => env && (!onlyAvailable || env.available !== false));
}

/**
 * Check if an environment is compatible with an algorithm.
 * @param {string} algorithmId - The algorithm identifier
 * @param {string} environmentId - The environment identifier
 * @returns {boolean} True if compatible
 */
export function isEnvironmentCompatible(algorithmId, environmentId) {
    const algorithmConfig = ALGORITHM_ENVIRONMENTS[algorithmId];

    if (!algorithmConfig) {
        return false;
    }

    return algorithmConfig.compatible.includes(environmentId);
}

/**
 * Validate environment configuration.
 * @param {string} environmentId - The environment identifier
 * @param {Object} config - The configuration to validate
 * @returns {Object} { valid: boolean, errors: string[] }
 */
export function validateEnvironmentConfig(environmentId, config) {
    const env = ENVIRONMENTS[environmentId];

    if (!env) {
        return { valid: false, errors: [`Unknown environment: ${environmentId}`] };
    }

    const errors = [];

    for (const field of env.configFields) {
        const value = config[field.name];

        if (value === undefined) {
            continue; // Will use default
        }

        if (field.type === 'number') {
            if (typeof value !== 'number' || isNaN(value)) {
                errors.push(`${field.label} must be a number`);
            } else if (field.min !== undefined && value < field.min) {
                errors.push(`${field.label} must be at least ${field.min}`);
            } else if (field.max !== undefined && value > field.max) {
                errors.push(`${field.label} must be at most ${field.max}`);
            }
        }

        if (field.type === 'boolean' && typeof value !== 'boolean') {
            errors.push(`${field.label} must be true or false`);
        }
    }

    return { valid: errors.length === 0, errors };
}

/**
 * Get default configuration for an environment.
 * @param {string} environmentId - The environment identifier
 * @returns {Object} Default configuration
 */
export function getDefaultConfig(environmentId) {
    const env = ENVIRONMENTS[environmentId];
    return env ? { ...env.defaultConfig } : {};
}
