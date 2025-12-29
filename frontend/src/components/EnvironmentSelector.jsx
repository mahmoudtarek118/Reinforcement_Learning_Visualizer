import { useState, useEffect } from 'react';
import {
    ENVIRONMENTS,
    getCompatibleEnvironments,
    getDefaultConfig,
    validateEnvironmentConfig
} from '../config/environments';
import './EnvironmentSelector.css';

/**
 * Environment selector component with dynamic configuration fields.
 * Shows only environments compatible with the current algorithm.
 */
function EnvironmentSelector({
    algorithmId,
    value,
    config,
    onChange,
    onConfigChange
}) {
    const [errors, setErrors] = useState([]);

    // Get compatible environments for this algorithm
    const compatibleEnvs = getCompatibleEnvironments(algorithmId, false);
    const availableEnvs = getCompatibleEnvironments(algorithmId, true);

    // Get current environment details
    const currentEnv = ENVIRONMENTS[value];

    // Handle environment selection change
    const handleEnvironmentChange = (e) => {
        const newEnvId = e.target.value;
        onChange(newEnvId);

        // Reset config to defaults for new environment
        const newConfig = getDefaultConfig(newEnvId);
        onConfigChange(newConfig);
        setErrors([]);
    };

    // Handle config field change
    const handleConfigChange = (fieldName, fieldValue, fieldType) => {
        let parsedValue = fieldValue;

        if (fieldType === 'number') {
            parsedValue = Number(fieldValue);
        } else if (fieldType === 'boolean') {
            parsedValue = fieldValue === 'true' || fieldValue === true;
        }

        const newConfig = { ...config, [fieldName]: parsedValue };
        onConfigChange(newConfig);

        // Validate
        const validation = validateEnvironmentConfig(value, newConfig);
        setErrors(validation.errors);
    };

    // Initialize config if empty
    useEffect(() => {
        if (!config || Object.keys(config).length === 0) {
            onConfigChange(getDefaultConfig(value));
        }
    }, [value]);

    return (
        <div className="environment-selector">
            <div className="control-group">
                <label>Environment</label>
                <select value={value} onChange={handleEnvironmentChange}>
                    {compatibleEnvs.map(env => (
                        <option
                            key={env.id}
                            value={env.id}
                            disabled={env.available === false}
                        >
                            {env.name} {env.available === false ? '(coming soon)' : ''}
                        </option>
                    ))}
                </select>
                {currentEnv && (
                    <span className="env-description">{currentEnv.description}</span>
                )}
            </div>

            {currentEnv && currentEnv.available !== false && currentEnv.configFields && (
                <div className="config-fields">
                    {currentEnv.configFields.map(field => (
                        <div key={field.name} className="control-group">
                            <label>{field.label}</label>
                            {field.type === 'number' && (
                                <input
                                    type="number"
                                    min={field.min}
                                    max={field.max}
                                    value={config[field.name] ?? field.default}
                                    onChange={(e) => handleConfigChange(field.name, e.target.value, 'number')}
                                />
                            )}
                            {field.type === 'boolean' && (
                                <select
                                    value={config[field.name] ?? field.default}
                                    onChange={(e) => handleConfigChange(field.name, e.target.value, 'boolean')}
                                >
                                    <option value="true">Yes</option>
                                    <option value="false">No</option>
                                </select>
                            )}
                        </div>
                    ))}
                </div>
            )}

            {errors.length > 0 && (
                <div className="validation-errors">
                    {errors.map((error, i) => (
                        <span key={i} className="error">{error}</span>
                    ))}
                </div>
            )}

            {availableEnvs.length === 0 && (
                <div className="no-environments">
                    No environments available for this algorithm yet.
                </div>
            )}
        </div>
    );
}

export default EnvironmentSelector;
