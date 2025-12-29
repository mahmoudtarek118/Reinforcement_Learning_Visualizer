/**
 * Algorithm parameters configuration.
 * Defines parameter metadata for dynamic rendering.
 * This structure mirrors what the backend will provide.
 */

// Common parameter definitions (reusable)
const COMMON_PARAMS = {
    gamma: {
        name: 'gamma',
        label: 'Discount Factor (γ)',
        description: 'How much to value future rewards vs immediate rewards',
        type: 'float',
        min: 0,
        max: 1,
        step: 0.01,
        default: 0.99
    },
    theta: {
        name: 'theta',
        label: 'Convergence Threshold (θ)',
        description: 'Stop when max value change is below this threshold',
        type: 'float',
        min: 0.000001,
        max: 0.1,
        step: 0.0001,
        default: 0.0001,
        precision: 6
    },
    alpha: {
        name: 'alpha',
        label: 'Learning Rate (α)',
        description: 'Step size for value updates',
        type: 'float',
        min: 0.01,
        max: 1,
        step: 0.01,
        default: 0.1
    },
    epsilon: {
        name: 'epsilon',
        label: 'Exploration Rate (ε)',
        description: 'Probability of taking a random action',
        type: 'float',
        min: 0,
        max: 1,
        step: 0.01,
        default: 0.1
    },
    numEpisodes: {
        name: 'numEpisodes',
        label: 'Number of Episodes',
        description: 'How many episodes to run',
        type: 'int',
        min: 1,
        max: 10000,
        step: 1,
        default: 100
    },
    maxIterations: {
        name: 'maxIterations',
        label: 'Max Iterations',
        description: 'Maximum iterations before stopping',
        type: 'int',
        min: 1,
        max: 10000,
        step: 1,
        default: 1000
    }
};

// Algorithm-specific parameter definitions
export const ALGORITHM_PARAMS = {
    'policy-evaluation': {
        name: 'Policy Evaluation',
        category: 'dynamic-programming',
        parameters: [
            { ...COMMON_PARAMS.gamma },
            { ...COMMON_PARAMS.theta },
            { ...COMMON_PARAMS.maxIterations }
        ]
    },
    'policy-iteration': {
        name: 'Policy Iteration',
        category: 'dynamic-programming',
        parameters: [
            { ...COMMON_PARAMS.gamma },
            { ...COMMON_PARAMS.theta },
            {
                name: 'maxPolicyIterations',
                label: 'Max Policy Iterations',
                description: 'Maximum number of policy improvement steps',
                type: 'int',
                min: 1,
                max: 100,
                step: 1,
                default: 100
            }
        ]
    },
    'value-iteration': {
        name: 'Value Iteration',
        category: 'dynamic-programming',
        parameters: [
            { ...COMMON_PARAMS.gamma },
            { ...COMMON_PARAMS.theta },
            { ...COMMON_PARAMS.maxIterations }
        ]
    },
    'monte-carlo': {
        name: 'Monte Carlo Prediction',
        category: 'model-free',
        parameters: [
            { ...COMMON_PARAMS.gamma },
            { ...COMMON_PARAMS.numEpisodes }
        ]
    },
    'td-learning': {
        name: 'TD Learning',
        category: 'model-free',
        parameters: [
            {
                name: 'tdType',
                label: 'TD Type',
                description: 'Choose between TD(0) and n-step TD',
                type: 'select',
                options: [
                    { value: 'td0', label: 'TD(0)' },
                    { value: 'nstep', label: 'n-step TD' }
                ],
                default: 'td0'
            },
            {
                name: 'nSteps',
                label: 'n (steps)',
                description: 'Number of steps for n-step TD',
                type: 'int',
                min: 1,
                max: 20,
                step: 1,
                default: 3,
                dependsOn: { field: 'tdType', value: 'nstep' }
            },
            { ...COMMON_PARAMS.alpha },
            { ...COMMON_PARAMS.gamma },
            { ...COMMON_PARAMS.numEpisodes }
        ]
    },
    'sarsa': {
        name: 'SARSA',
        category: 'model-free',
        parameters: [
            { ...COMMON_PARAMS.alpha },
            { ...COMMON_PARAMS.gamma },
            { ...COMMON_PARAMS.epsilon },
            { ...COMMON_PARAMS.numEpisodes }
        ]
    },
    'q-learning': {
        name: 'Q-Learning',
        category: 'model-free',
        parameters: [
            { ...COMMON_PARAMS.alpha },
            { ...COMMON_PARAMS.gamma },
            { ...COMMON_PARAMS.epsilon },
            { ...COMMON_PARAMS.numEpisodes }
        ]
    }
};

/**
 * Get parameters for an algorithm.
 * @param {string} algorithmId - The algorithm identifier
 * @returns {Object} Algorithm config with parameters
 */
export function getAlgorithmParams(algorithmId) {
    return ALGORITHM_PARAMS[algorithmId] || null;
}

/**
 * Get default values for an algorithm's parameters.
 * @param {string} algorithmId - The algorithm identifier
 * @returns {Object} Map of parameter name to default value
 */
export function getDefaultParamValues(algorithmId) {
    const config = ALGORITHM_PARAMS[algorithmId];
    if (!config) return {};

    const defaults = {};
    for (const param of config.parameters) {
        defaults[param.name] = param.default;
    }
    return defaults;
}

/**
 * Validate parameter values.
 * @param {string} algorithmId - The algorithm identifier
 * @param {Object} values - Current parameter values
 * @returns {Object} { valid: boolean, errors: { [paramName]: string } }
 */
export function validateParams(algorithmId, values) {
    const config = ALGORITHM_PARAMS[algorithmId];
    if (!config) return { valid: false, errors: { _general: 'Unknown algorithm' } };

    const errors = {};

    for (const param of config.parameters) {
        const value = values[param.name];

        // Skip validation if param depends on another field that isn't set
        if (param.dependsOn) {
            if (values[param.dependsOn.field] !== param.dependsOn.value) {
                continue;
            }
        }

        if (value === undefined || value === null) continue;

        if (param.type === 'int' || param.type === 'float') {
            if (typeof value !== 'number' || isNaN(value)) {
                errors[param.name] = `${param.label} must be a number`;
            } else if (param.min !== undefined && value < param.min) {
                errors[param.name] = `${param.label} must be at least ${param.min}`;
            } else if (param.max !== undefined && value > param.max) {
                errors[param.name] = `${param.label} must be at most ${param.max}`;
            }
        }

        if (param.type === 'select') {
            const validValues = param.options.map(o => o.value);
            if (!validValues.includes(value)) {
                errors[param.name] = `Invalid value for ${param.label}`;
            }
        }
    }

    return { valid: Object.keys(errors).length === 0, errors };
}
