import './SpeedControl.css';

/**
 * Animation speed control slider.
 */
function SpeedControl({ value, onChange, disabled = false }) {
    const speedLabels = {
        0.25: 'Very Slow',
        0.5: 'Slow',
        1: 'Normal',
        1.5: 'Fast',
        2: 'Very Fast',
        3: 'Instant'
    };

    // Find closest label
    const getLabel = (val) => {
        const keys = Object.keys(speedLabels).map(Number);
        const closest = keys.reduce((a, b) =>
            Math.abs(b - val) < Math.abs(a - val) ? b : a
        );
        return speedLabels[closest];
    };

    return (
        <div className={`speed-control ${disabled ? 'disabled' : ''}`}>
            <div className="speed-header">
                <label>Animation Speed</label>
                <span className="speed-value">{getLabel(value)}</span>
            </div>
            <input
                type="range"
                min="0.25"
                max="3"
                step="0.25"
                value={value}
                onChange={(e) => onChange(Number(e.target.value))}
                disabled={disabled}
                className="speed-slider"
            />
            <div className="speed-marks">
                <span>Slow</span>
                <span>Fast</span>
            </div>
        </div>
    );
}

export default SpeedControl;
