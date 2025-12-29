import { useState, useCallback, useEffect, useRef } from 'react';
import EnvironmentSelector from '../components/EnvironmentSelector';
import ParameterPanel from '../components/ParameterPanel';
import VisualizationPanel from '../components/VisualizationPanel';
import SpeedControl from '../components/SpeedControl';
import { getDefaultConfig, ENVIRONMENTS } from '../config/environments';
import { getDefaultParamValues } from '../config/algorithmParams';
import * as api from '../api/client';
import './AlgorithmPage.css';

const ALGORITHM_ID = 'monte-carlo';

function MonteCarlo() {
    const [environment, setEnvironment] = useState('gridworld');
    const [envConfig, setEnvConfig] = useState(getDefaultConfig('gridworld'));
    const [params, setParams] = useState(getDefaultParamValues(ALGORITHM_ID));
    const [agentPos, setAgentPos] = useState({ row: 0, col: 0 });
    const [animationSpeed, setAnimationSpeed] = useState(1);

    const [sessionId, setSessionId] = useState(null);
    const [isRunning, setIsRunning] = useState(false);
    const [isPaused, setIsPaused] = useState(false);
    const [error, setError] = useState(null);
    const [diagnostics, setDiagnostics] = useState(null);
    const [step, setStep] = useState(0);
    const [episode, setEpisode] = useState(0);

    const runningRef = useRef(false);
    const intervalRef = useRef(null);
    const sessionRef = useRef(null);

    const rows = envConfig.rows || 4;
    const cols = envConfig.cols || 4;

    useEffect(() => {
        return () => {
            if (intervalRef.current) clearInterval(intervalRef.current);
            if (sessionRef.current) api.deleteSession(sessionRef.current).catch(() => { });
        };
    }, []);

    const doStep = useCallback(async () => {
        if (!sessionRef.current) return false;

        try {
            const result = await api.stepSession(sessionRef.current);
            setAgentPos(result.agent_position);
            setStep(result.step);
            setEpisode(result.episode || 0);
            setDiagnostics(result.diagnostics);

            if (result.episode >= (params.numEpisodes || 100)) {
                return true;
            }
            return false;
        } catch (err) {
            setError(err.message);
            return true;
        }
    }, [params.numEpisodes]);

    const handleStart = useCallback(async () => {
        setError(null);

        try {
            const envType = environment;
            const session = await api.createSession(
                { type: envType, rows, cols, maxSteps: envConfig.maxSteps || 100, mapName: envConfig.mapName || '4x4', isSlippery: envConfig.isSlippery !== false },
                { type: ALGORITHM_ID, params }
            );

            sessionRef.current = session.session_id;
            setSessionId(session.session_id);
            setAgentPos(session.agent_position);
            setIsRunning(true);
            setIsPaused(false);
            runningRef.current = true;

            intervalRef.current = setInterval(async () => {
                if (!runningRef.current) return;

                const shouldStop = await doStep();
                if (shouldStop) {
                    clearInterval(intervalRef.current);
                    intervalRef.current = null;
                    runningRef.current = false;
                    setIsRunning(false);
                }
            }, 100 / animationSpeed);

        } catch (err) {
            setError(err.message);
            setIsRunning(false);
        }
    }, [rows, cols, envConfig.maxSteps, params, animationSpeed, doStep]);

    const handleStop = useCallback(() => {
        runningRef.current = false;
        if (intervalRef.current) clearInterval(intervalRef.current);
        intervalRef.current = null;
        setIsRunning(false);
        setIsPaused(false);
    }, []);

    const handlePauseResume = useCallback(() => {
        if (isPaused) {
            runningRef.current = true;
            setIsPaused(false);
            intervalRef.current = setInterval(async () => {
                if (!runningRef.current) return;
                const shouldStop = await doStep();
                if (shouldStop) handleStop();
            }, 100 / animationSpeed);
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
                setAgentPos(session.agent_position);
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
            try { await api.deleteSession(sessionRef.current); } catch { }
        }
        sessionRef.current = null;
        setSessionId(null);
        setAgentPos({ row: 0, col: 0 });
        setStep(0);
        setEpisode(0);
        setDiagnostics(null);
        setError(null);
    }, [handleStop]);

    return (
        <div className="algorithm-page">
            <header className="page-header">
                <h1>Monte Carlo Prediction</h1>
                <p>Learn value function from complete episode returns</p>
            </header>

            <div className="algorithm-description">
                <h2>Algorithm Description</h2>
                <p>First-Visit MC averages returns from complete episodes:</p>
                <div className="formula">V(s) = average(G<sub>t</sub> | S<sub>t</sub> = s)</div>
                <ul className="key-points">
                    <li><strong>Model-free:</strong> No knowledge of environment dynamics</li>
                    <li><strong>Episodic:</strong> Requires complete episodes</li>
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
                            <button className="run-button" onClick={handleStart}>▶ Run Monte Carlo</button>
                        ) : (
                            <button className="run-button pause" onClick={handlePauseResume}>
                                {isPaused ? '▶ Resume' : '⏸ Pause'}
                            </button>
                        )}
                        <div className="button-row">
                            <button className="step-button" onClick={handleStep} disabled={isRunning && !isPaused}>Step</button>
                            <button className="reset-button" onClick={handleReset}>Reset</button>
                            {isRunning && <button className="stop-button" onClick={handleStop}>⏹ Stop</button>}
                        </div>
                    </div>

                    {error && <div className="error-message">{error}</div>}
                </section>

                <section className="visualization-section">
                    <h2>Visualization</h2>
                    <VisualizationPanel rows={rows} cols={cols} agentPosition={agentPos}
                        animationSpeed={animationSpeed} diagnostics={diagnostics} algorithmType={ALGORITHM_ID}
                        holes={ENVIRONMENTS[environment]?.holes || []} environmentType={environment} />
                </section>

                <section className="diagnostics-section">
                    <h2>Diagnostics</h2>
                    <div className="diagnostics-placeholder">
                        <div className="diagnostic-item"><span className="label">Step</span><span className="value">{step}</span></div>
                        <div className="diagnostic-item"><span className="label">Episode</span><span className="value">{episode}</span></div>
                    </div>
                </section>
            </div>
        </div>
    );
}

export default MonteCarlo;
