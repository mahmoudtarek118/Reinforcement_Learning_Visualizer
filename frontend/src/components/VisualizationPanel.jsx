import GridWorldCanvas from './GridWorldCanvas';
import LearningCurve from './LearningCurve';
import './VisualizationPanel.css';

/**
 * Unified visualization panel for RL algorithms.
 * Combines GridWorld, value heatmaps, policy arrows, and learning curves.
 * All visuals are driven by diagnostics data.
 * 
 * Props:
 *   rows, cols - Grid dimensions
 *   agentPosition - Current agent position
 *   animationSpeed - Animation speed
 *   diagnostics - Algorithm diagnostics object
 *   algorithmType - Type of algorithm (for display customization)
 *   holes - Array of state IDs that are holes (for FrozenLake)
 *   environmentType - 'gridworld' or 'frozenlake'
 */
function VisualizationPanel({
    rows = 4,
    cols = 4,
    agentPosition = { row: 0, col: 0 },
    animationSpeed = 1,
    diagnostics = null,
    algorithmType = 'q-learning',
    holes = [],
    environmentType = 'gridworld'
}) {
    // Extract values from diagnostics
    const extractValues = () => {
        if (!diagnostics) return null;

        // Q-learning and SARSA: max Q per state
        if (diagnostics.q_values) {
            return diagnostics.q_values.map(qs => Math.max(...qs));
        }

        // Value-based methods
        if (diagnostics.values) {
            return diagnostics.values;
        }

        // From value_table
        if (diagnostics.value_table) {
            return diagnostics.value_table;
        }

        return null;
    };

    // Extract policy from diagnostics
    const extractPolicy = () => {
        if (!diagnostics) return null;

        // Direct policy array
        if (diagnostics.policy && Array.isArray(diagnostics.policy)) {
            return diagnostics.policy;
        }

        // From optimal_policy
        if (diagnostics.optimal_policy) {
            return diagnostics.optimal_policy;
        }

        return null;
    };

    // Extract TD errors for learning curve
    const extractTDErrors = () => {
        if (!diagnostics) return [];

        // Direct td_errors array
        if (diagnostics.td_errors && Array.isArray(diagnostics.td_errors)) {
            // Sample if too many points
            const errors = diagnostics.td_errors;
            if (errors.length > 200) {
                const step = Math.floor(errors.length / 200);
                return errors.filter((_, i) => i % step === 0).map(Math.abs);
            }
            return errors.map(Math.abs);
        }

        return [];
    };

    // Extract delta history for DP algorithms
    const extractDeltaHistory = () => {
        if (!diagnostics) return [];

        if (diagnostics.delta_history) {
            return diagnostics.delta_history;
        }

        return [];
    };

    const values = extractValues();
    const policy = extractPolicy();
    const tdErrors = extractTDErrors();
    const deltaHistory = extractDeltaHistory();

    // Choose which learning curve to show
    const curveData = tdErrors.length > 0 ? tdErrors : deltaHistory;
    const curveLabel = tdErrors.length > 0 ? '|TD Error|' : 'ΔV';
    const curveColor = tdErrors.length > 0 ? '#00d9ff' : '#00ff88';

    return (
        <div className="visualization-panel">
            <div className="viz-main">
                <div className="viz-grid">
                    <h3>Environment</h3>
                    <GridWorldCanvas
                        rows={rows}
                        cols={cols}
                        agentPosition={agentPosition}
                        animationSpeed={animationSpeed}
                        values={values}
                        policy={policy}
                        holes={holes}
                        environmentType={environmentType}
                        cellSize={Math.min(60, 280 / Math.max(rows, cols))}
                    />
                </div>

                {values && (
                    <div className="viz-values">
                        <h3>State Values</h3>
                        <div className="value-grid" style={{
                            gridTemplateColumns: `repeat(${cols}, 1fr)`
                        }}>
                            {values.map((v, i) => (
                                <div
                                    key={i}
                                    className={`value-cell ${v >= 0 ? 'positive' : 'negative'}`}
                                    style={{
                                        backgroundColor: v >= 0
                                            ? `rgba(0, 255, 136, ${Math.min(Math.abs(v) / (Math.max(...values.map(Math.abs)) || 1), 1) * 0.5})`
                                            : `rgba(255, 107, 107, ${Math.min(Math.abs(v) / (Math.max(...values.map(Math.abs)) || 1), 1) * 0.5})`
                                    }}
                                >
                                    {v.toFixed(1)}
                                </div>
                            ))}
                        </div>
                        <div className="value-legend">
                            <span className="legend-negative">Low</span>
                            <span className="legend-positive">High</span>
                        </div>
                    </div>
                )}
            </div>

            {curveData.length > 0 && (
                <div className="viz-curve">
                    <h3>Learning Curve</h3>
                    <LearningCurve
                        data={curveData}
                        label={curveLabel}
                        color={curveColor}
                        width={350}
                        height={150}
                    />
                </div>
            )}

            {policy && (
                <div className="viz-policy">
                    <h3>Policy</h3>
                    <div className="policy-grid" style={{
                        gridTemplateColumns: `repeat(${cols}, 1fr)`
                    }}>
                        {policy.map((action, i) => {
                            const isGoal = i === rows * cols - 1;
                            const arrows = ['↑', '↓', '←', '→'];
                            return (
                                <div
                                    key={i}
                                    className={`policy-cell ${isGoal ? 'goal' : ''}`}
                                >
                                    {isGoal ? 'G' : arrows[action] || '?'}
                                </div>
                            );
                        })}
                    </div>
                </div>
            )}
        </div>
    );
}

export default VisualizationPanel;
