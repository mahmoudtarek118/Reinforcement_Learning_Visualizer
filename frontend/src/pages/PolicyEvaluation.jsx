import { useState, useCallback, useEffect, useRef } from 'react';
import EnvironmentSelector from '../components/EnvironmentSelector';
import ParameterPanel from '../components/ParameterPanel';
import VisualizationPanel from '../components/VisualizationPanel';
import SpeedControl from '../components/SpeedControl';
import { getDefaultConfig, ENVIRONMENTS } from '../config/environments';
import { getDefaultParamValues } from '../config/algorithmParams';
import * as api from '../api/client';
import './AlgorithmPage.css';

const ALGORITHM_ID = 'policy-evaluation';

function PolicyEvaluation() {
    const [environment, setEnvironment] = useState('gridworld');
    const [envConfig, setEnvConfig] = useState(getDefaultConfig('gridworld'));
    const [params, setParams] = useState(getDefaultParamValues(ALGORITHM_ID));
    const [agentPos] = useState({ row: 0, col: 0 });
    const [animationSpeed, setAnimationSpeed] = useState(1);

    const [sessionId, setSessionId] = useState(null);
    const [isRunning, setIsRunning] = useState(false);
    const [isPaused, setIsPaused] = useState(false);
    const [error, setError] = useState(null);
    const [diagnostics, setDiagnostics] = useState(null);
    const [iteration, setIteration] = useState(0);
    const [converged, setConverged] = useState(false);

    const runningRef = useRef(false);
    const intervalRef = useRef(null);
    const sessionRef = useRef(null);  // Store session ID in ref for closure

    const rows = envConfig.rows || 4;
    const cols = envConfig.cols || 4;

    useEffect(() => {
        return () => {
            if (intervalRef.current) clearInterval(intervalRef.current);
            if (sessionRef.current) api.deleteSession(sessionRef.current).catch(() => { });
        };
    }, []);

    // Step function that both Run and Step use
    const doStep = useCallback(async () => {
        if (!sessionRef.current) return false;

        try {
            const result = await api.stepSession(sessionRef.current);
            setIteration(result.step);
            setDiagnostics(result.diagnostics);

            if (result.done || result.status === 'converged') {
                setConverged(true);
                return true; // Signal to stop
            }
            return false;
        } catch (err) {
            setError(err.message);
            return true; // Signal to stop on error
        }
    }, []);

    const handleStart = useCallback(async () => {
        setError(null);
        setConverged(false);

        try {
            // Create session with selected environment
            const envType = environment;
            const session = await api.createSession(
                { type: envType, rows, cols, maxSteps: envConfig.maxSteps || 100, mapName: envConfig.mapName || '4x4', isSlippery: envConfig.isSlippery !== false },
                { type: ALGORITHM_ID, params }
            );

            sessionRef.current = session.session_id;
            setSessionId(session.session_id);
            setIsRunning(true);
            setIsPaused(false);
            runningRef.current = true;

            // Start interval
            const runInterval = () => {
                intervalRef.current = setInterval(async () => {
                    if (!runningRef.current) return;

                    const shouldStop = await doStep();
                    if (shouldStop) {
                        clearInterval(intervalRef.current);
                        intervalRef.current = null;
                        runningRef.current = false;
                        setIsRunning(false);
                    }
                }, 300 / animationSpeed);
            };

            runInterval();

        } catch (err) {
            setError(err.message);
            setIsRunning(false);
        }
    }, [rows, cols, envConfig.maxSteps, params, animationSpeed, doStep]);

    const handleStop = useCallback(() => {
        runningRef.current = false;
        if (intervalRef.current) {
            clearInterval(intervalRef.current);
            intervalRef.current = null;
        }
        setIsRunning(false);
        setIsPaused(false);
    }, []);

    const handlePauseResume = useCallback(async () => {
        if (isPaused) {
            runningRef.current = true;
            setIsPaused(false);
            intervalRef.current = setInterval(async () => {
                if (!runningRef.current) return;
                const shouldStop = await doStep();
                if (shouldStop) handleStop();
            }, 300 / animationSpeed);
        } else {
            runningRef.current = false;
            if (intervalRef.current) clearInterval(intervalRef.current);
            setIsPaused(true);
        }
    }, [isPaused, animationSpeed, doStep, handleStop]);

    const handleStep = useCallback(async () => {
        setError(null);

        if (!sessionRef.current) {
            try {
                const envType = environment;
                const session = await api.createSession(
                    { type: envType, rows, cols, maxSteps: envConfig.maxSteps || 100, mapName: envConfig.mapName || '4x4', isSlippery: envConfig.isSlippery !== false },
                    { type: ALGORITHM_ID, params }
                );
                sessionRef.current = session.session_id;
                setSessionId(session.session_id);
            } catch (err) {
                setError(err.message);
                return;
            }
        }

        await doStep();
    }, [rows, cols, envConfig.maxSteps, params, doStep]);

    const handleReset = useCallback(async () => {
        handleStop();
        if (sessionRef.current) {
            try { await api.deleteSession(sessionRef.current); } catch (e) { }
        }
        sessionRef.current = null;
        setSessionId(null);
        setIteration(0);
        setDiagnostics(null);
        setConverged(false);
        setError(null);
    }, [handleStop]);

    return (
        <div className="algorithm-page">
            <header className="page-header">
                <h1>Policy Evaluation</h1>
                <p>Evaluate a given policy using iterative Bellman expectation updates</p>
            </header>

            <div className="algorithm-description">
                <h2>Algorithm Description</h2>
                <p>Policy Evaluation computes the state-value function V<sup>π</sup>(s) for a given policy π:</p>
                <div className="formula">V(s) = Σ<sub>a</sub> π(a|s) × [R(s,a) + γ × V(s')]</div>
                <ul className="key-points">
                    <li><strong>Input:</strong> A policy π (uniform random)</li>
                    <li><strong>Output:</strong> Value function V<sup>π</sup>(s)</li>
                    <li><strong>Convergence:</strong> When max|ΔV| &lt; θ</li>
                </ul>
            </div>

            <div className="page-content">
                <section className="controls-section">
                    <h2>Configuration</h2>
                    <EnvironmentSelector algorithmId={ALGORITHM_ID} value={environment} config={envConfig}
                        onChange={setEnvironment} onConfigChange={setEnvConfig} />
                    <ParameterPanel algorithmId={ALGORITHM_ID} values={params} onChange={setParams} />

                    <SpeedControl value={animationSpeed} onChange={setAnimationSpeed} disabled={isRunning && !isPaused} />

                    <div className="control-buttons">
                        {!isRunning ? (
                            <button className="run-button" onClick={handleStart} disabled={converged}>
                                {converged ? '✓ Converged' : '▶ Run Policy Evaluation'}
                            </button>
                        ) : (
                            <button className="run-button pause" onClick={handlePauseResume}>
                                {isPaused ? '▶ Resume' : '⏸ Pause'}
                            </button>
                        )}
                        <div className="button-row">
                            <button className="step-button" onClick={handleStep} disabled={(isRunning && !isPaused) || converged}>Step</button>
                            <button className="reset-button" onClick={handleReset}>Reset</button>
                            {isRunning && <button className="stop-button" onClick={handleStop}>⏹ Stop</button>}
                        </div>
                    </div>

                    {error && <div className="error-message">{error}</div>}
                </section>

                <section className="visualization-section">
                    <h2>Visualization</h2>
                    <VisualizationPanel
                        rows={rows}
                        cols={cols}
                        agentPosition={agentPos}
                        animationSpeed={animationSpeed}
                        diagnostics={diagnostics}
                        algorithmType={ALGORITHM_ID}
                        holes={ENVIRONMENTS[environment]?.holes || []}
                        environmentType={environment}
                    />
                </section>

                <section className="diagnostics-section">
                    <h2>Diagnostics</h2>
                    <div className="diagnostics-placeholder">
                        <div className="diagnostic-item">
                            <span className="label">Iterations</span>
                            <span className="value">{iteration}</span>
                        </div>
                        <div className="diagnostic-item">
                            <span className="label">Final ΔV</span>
                            <span className="value">{diagnostics?.max_delta?.toFixed(6) || '-'}</span>
                        </div>
                        <div className="diagnostic-item">
                            <span className="label">Converged</span>
                            <span className="value">{converged ? '✓ Yes' : 'No'}</span>
                        </div>
                    </div>
                </section>
            </div>
        </div>
    );
}

export default PolicyEvaluation;
