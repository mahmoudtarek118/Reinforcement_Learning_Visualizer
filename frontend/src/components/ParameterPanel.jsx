import { useState, useEffect } from 'react';
import { getAlgorithmParams, getDefaultParamValues, validateParams } from '../config/algorithmParams';
import './ParameterPanel.css';

/**
 * Dynamic parameter panel component.
 * Renders algorithm parameters as sliders + numeric inputs based on metadata.
 */
function ParameterPanel({ algorithmId, values, onChange }) {
    const [errors, setErrors] = useState({});

    // Get algorithm parameter config
    const algorithmConfig = getAlgorithmParams(algorithmId);

    // Initialize with defaults if empty
    useEffect(() => {
        if (!values || Object.keys(values).length === 0) {
            onChange(getDefaultParamValues(algorithmId));
        }
    }, [algorithmId]);

    // Handle parameter change
    const handleChange = (paramName, rawValue, paramType) => {
        let value = rawValue;

        if (paramType === 'int') {
            value = parseInt(rawValue, 10);
        } else if (paramType === 'float') {
            value = parseFloat(rawValue);
        }

        const newValues = { ...values, [paramName]: value };
        onChange(newValues);

        // Validate
        const validation = validateParams(algorithmId, newValues);
        setErrors(validation.errors);
    };

    // Check if a parameter should be visible (based on dependencies)
    const isParamVisible = (param) => {
        if (!param.dependsOn) return true;
        return values[param.dependsOn.field] === param.dependsOn.value;
    };

    if (!algorithmConfig) {
        return <div className="parameter-panel-error">Unknown algorithm</div>;
    }

    return (
        <div className="parameter-panel">
            {algorithmConfig.parameters.map(param => {
                if (!isParamVisible(param)) return null;

                const value = values[param.name] ?? param.default;
                const error = errors[param.name];

                return (
                    <div key={param.name} className={`param-group ${error ? 'has-error' : ''}`}>
                        <div className="param-header">
                            <label htmlFor={param.name}>{param.label}</label>
                            {param.description && (
                                <span className="param-tooltip" title={param.description}>?</span>
                            )}
                        </div>

                        {/* Numeric types: slider + input */}
                        {(param.type === 'int' || param.type === 'float') && (
                            <div className="param-input-group">
                                <input
                                    type="range"
                                    id={`${param.name}-slider`}
                                    min={param.min}
                                    max={param.max}
                                    step={param.step}
                                    value={value}
                                    onChange={(e) => handleChange(param.name, e.target.value, param.type)}
                                    className="param-slider"
                                />
                                <input
                                    type="number"
                                    id={param.name}
                                    min={param.min}
                                    max={param.max}
                                    step={param.step}
                                    value={value}
                                    onChange={(e) => handleChange(param.name, e.target.value, param.type)}
                                    className="param-number"
                                />
                            </div>
                        )}

                        {/* Select type: dropdown */}
                        {param.type === 'select' && (
                            <select
                                id={param.name}
                                value={value}
                                onChange={(e) => handleChange(param.name, e.target.value, param.type)}
                                className="param-select"
                            >
                                {param.options.map(option => (
                                    <option key={option.value} value={option.value}>
                                        {option.label}
                                    </option>
                                ))}
                            </select>
                        )}

                        {error && <span className="param-error">{error}</span>}
                    </div>
                );
            })}
        </div>
    );
}

export default ParameterPanel;
