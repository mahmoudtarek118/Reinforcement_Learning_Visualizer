import { useRef, useEffect, useState, useCallback } from 'react';
import './GridWorldCanvas.css';

/**
 * GridWorld Canvas Component with Animation.
 * Renders GridWorld environment using HTML Canvas with smooth agent movement.
 * Also supports FrozenLake with holes.
 * 
 * Props:
 *   rows - Number of rows in the grid
 *   cols - Number of columns in the grid
 *   agentPosition - Target agent position { row, col }
 *   goalPosition - Goal position { row, col } (defaults to bottom-right)
 *   holes - Array of state IDs that are holes (for FrozenLake)
 *   values - Optional value array for heatmap display
 *   policy - Optional policy array for arrow display
 *   cellSize - Size of each cell in pixels (default: 60)
 *   animationSpeed - Animation speed multiplier (0.1 to 3, default: 1)
 *   environmentType - 'gridworld' or 'frozenlake'
 */
function GridWorldCanvas({
    rows = 4,
    cols = 4,
    agentPosition = { row: 0, col: 0 },
    goalPosition = null,
    holes = [],
    values = null,
    policy = null,
    cellSize = 60,
    animationSpeed = 1,
    environmentType = 'gridworld'
}) {
    const canvasRef = useRef(null);
    const animationRef = useRef(null);

    // Animated agent position (interpolated)
    const [animatedPos, setAnimatedPos] = useState({
        row: agentPosition.row,
        col: agentPosition.col
    });

    // Previous target for animation
    const targetRef = useRef({ row: agentPosition.row, col: agentPosition.col });
    const currentPosRef = useRef({ row: agentPosition.row, col: agentPosition.col });

    // Default goal to bottom-right
    const goal = goalPosition || { row: rows - 1, col: cols - 1 };

    // Canvas dimensions
    const width = cols * cellSize;
    const height = rows * cellSize;

    // Colors
    const COLORS = {
        background: '#1a1a2e',
        gridLine: '#333',
        startCell: 'rgba(0, 217, 255, 0.15)',
        goalCell: 'rgba(0, 255, 136, 0.25)',
        goalBorder: '#00ff88',
        agentFill: '#00d9ff',
        agentBorder: '#fff',
        text: '#fff',
        // FrozenLake colors
        holeCell: 'rgba(30, 60, 90, 0.9)',      // Dark icy blue for holes
        holeBorder: '#4a90d9',                    // Lighter blue border
        frozenCell: 'rgba(100, 180, 255, 0.1)',  // Light icy tint
        icePattern: 'rgba(200, 230, 255, 0.15)' // Ice crack pattern
    };

    // Arrow directions for policy visualization
    const ARROWS = {
        0: { dx: 0, dy: -0.3 },  // UP
        1: { dx: 0, dy: 0.3 },   // DOWN
        2: { dx: -0.3, dy: 0 },  // LEFT
        3: { dx: 0.3, dy: 0 }    // RIGHT
    };

    // Animation loop
    const animate = useCallback(() => {
        const target = targetRef.current;
        const current = currentPosRef.current;

        // Calculate distance to target
        const dx = target.col - current.col;
        const dy = target.row - current.row;
        const distance = Math.sqrt(dx * dx + dy * dy);

        // If close enough, snap to target
        if (distance < 0.01) {
            current.row = target.row;
            current.col = target.col;
            setAnimatedPos({ row: current.row, col: current.col });
            return; // Stop animation
        }

        // Interpolate towards target
        // Speed: base speed * animationSpeed multiplier
        const speed = 0.08 * animationSpeed;
        const moveAmount = Math.min(speed, distance);

        current.col += (dx / distance) * moveAmount;
        current.row += (dy / distance) * moveAmount;

        setAnimatedPos({ row: current.row, col: current.col });

        // Continue animation
        animationRef.current = requestAnimationFrame(animate);
    }, [animationSpeed]);

    // Start animation when target changes
    useEffect(() => {
        targetRef.current = { row: agentPosition.row, col: agentPosition.col };

        // Cancel any existing animation
        if (animationRef.current) {
            cancelAnimationFrame(animationRef.current);
        }

        // Start new animation
        animationRef.current = requestAnimationFrame(animate);

        return () => {
            if (animationRef.current) {
                cancelAnimationFrame(animationRef.current);
            }
        };
    }, [agentPosition.row, agentPosition.col, animate]);

    // Draw canvas
    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const ctx = canvas.getContext('2d');

        // Clear canvas
        ctx.fillStyle = COLORS.background;
        ctx.fillRect(0, 0, width, height);

        // Calculate value range for heatmap
        let minValue = 0, maxValue = 0;
        if (values && values.length > 0) {
            minValue = Math.min(...values);
            maxValue = Math.max(...values);
        }

        // Draw cells
        for (let row = 0; row < rows; row++) {
            for (let col = 0; col < cols; col++) {
                const x = col * cellSize;
                const y = row * cellSize;
                const stateId = row * cols + col;

                // Draw value heatmap if available
                if (values && values.length > stateId) {
                    const value = values[stateId];
                    const range = Math.max(Math.abs(minValue), Math.abs(maxValue), 0.01);
                    const intensity = Math.min(Math.abs(value) / range, 1) * 0.6;

                    if (value >= 0) {
                        ctx.fillStyle = `rgba(0, 255, 136, ${intensity})`;
                    } else {
                        ctx.fillStyle = `rgba(255, 107, 107, ${intensity})`;
                    }
                    ctx.fillRect(x + 1, y + 1, cellSize - 2, cellSize - 2);
                }

                // Draw start cell highlight (top-left)
                if (row === 0 && col === 0) {
                    ctx.fillStyle = COLORS.startCell;
                    ctx.fillRect(x + 1, y + 1, cellSize - 2, cellSize - 2);
                }

                // Draw hole cells (for FrozenLake)
                const isHole = holes.includes(stateId);
                if (isHole) {
                    // Dark icy blue hole
                    ctx.fillStyle = COLORS.holeCell;
                    ctx.fillRect(x + 1, y + 1, cellSize - 2, cellSize - 2);

                    // Ice crack pattern
                    ctx.strokeStyle = COLORS.icePattern;
                    ctx.lineWidth = 1;
                    ctx.beginPath();
                    ctx.moveTo(x + cellSize * 0.2, y + cellSize * 0.3);
                    ctx.lineTo(x + cellSize * 0.5, y + cellSize * 0.5);
                    ctx.lineTo(x + cellSize * 0.8, y + cellSize * 0.4);
                    ctx.moveTo(x + cellSize * 0.5, y + cellSize * 0.5);
                    ctx.lineTo(x + cellSize * 0.4, y + cellSize * 0.8);
                    ctx.stroke();

                    // Border
                    ctx.strokeStyle = COLORS.holeBorder;
                    ctx.lineWidth = 2;
                    ctx.strokeRect(x + 2, y + 2, cellSize - 4, cellSize - 4);

                    // 'H' label
                    ctx.fillStyle = '#ff6b6b';
                    ctx.font = 'bold 14px sans-serif';
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'middle';
                    ctx.fillText('H', x + cellSize / 2, y + cellSize / 2 - (values ? 8 : 0));
                }

                // Draw goal cell
                if (row === goal.row && col === goal.col) {
                    ctx.fillStyle = COLORS.goalCell;
                    ctx.fillRect(x + 1, y + 1, cellSize - 2, cellSize - 2);

                    ctx.strokeStyle = COLORS.goalBorder;
                    ctx.lineWidth = 2;
                    ctx.strokeRect(x + 2, y + 2, cellSize - 4, cellSize - 4);
                }

                // Draw value text if available
                if (values && values.length > stateId) {
                    const value = values[stateId];
                    ctx.fillStyle = COLORS.text;
                    ctx.font = '12px monospace';
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'middle';
                    ctx.fillText(value.toFixed(1), x + cellSize / 2, y + cellSize / 2 + 12);
                }

                // Draw policy arrow if available
                if (policy && policy.length > stateId && !(row === goal.row && col === goal.col)) {
                    const action = policy[stateId];
                    const arrow = ARROWS[action];
                    if (arrow) {
                        const centerX = x + cellSize / 2;
                        const centerY = y + cellSize / 2 - 5;
                        const arrowLen = cellSize * 0.25;

                        ctx.strokeStyle = '#00d9ff';
                        ctx.lineWidth = 2;
                        ctx.beginPath();
                        ctx.moveTo(centerX - arrow.dx * arrowLen, centerY - arrow.dy * arrowLen);
                        ctx.lineTo(centerX + arrow.dx * arrowLen, centerY + arrow.dy * arrowLen);
                        ctx.stroke();

                        const headLen = 6;
                        const angle = Math.atan2(arrow.dy, arrow.dx);
                        const tipX = centerX + arrow.dx * arrowLen;
                        const tipY = centerY + arrow.dy * arrowLen;

                        ctx.beginPath();
                        ctx.moveTo(tipX, tipY);
                        ctx.lineTo(
                            tipX - headLen * Math.cos(angle - Math.PI / 6),
                            tipY - headLen * Math.sin(angle - Math.PI / 6)
                        );
                        ctx.moveTo(tipX, tipY);
                        ctx.lineTo(
                            tipX - headLen * Math.cos(angle + Math.PI / 6),
                            tipY - headLen * Math.sin(angle + Math.PI / 6)
                        );
                        ctx.stroke();
                    }
                }
            }
        }

        // Draw grid lines
        ctx.strokeStyle = COLORS.gridLine;
        ctx.lineWidth = 1;

        for (let col = 0; col <= cols; col++) {
            ctx.beginPath();
            ctx.moveTo(col * cellSize, 0);
            ctx.lineTo(col * cellSize, height);
            ctx.stroke();
        }

        for (let row = 0; row <= rows; row++) {
            ctx.beginPath();
            ctx.moveTo(0, row * cellSize);
            ctx.lineTo(width, row * cellSize);
            ctx.stroke();
        }

        // Draw labels
        ctx.font = 'bold 14px sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';

        ctx.fillStyle = '#00d9ff';
        ctx.fillText('S', cellSize / 2, cellSize / 2 - (values ? 8 : 0));

        ctx.fillStyle = '#00ff88';
        ctx.fillText('G', goal.col * cellSize + cellSize / 2, goal.row * cellSize + cellSize / 2 - (values ? 8 : 0));

        // Draw agent at animated position
        const agentX = animatedPos.col * cellSize + cellSize / 2;
        const agentY = animatedPos.row * cellSize + cellSize / 2;
        const agentRadius = cellSize * 0.25;

        // Agent glow
        const gradient = ctx.createRadialGradient(agentX, agentY, 0, agentX, agentY, agentRadius * 1.5);
        gradient.addColorStop(0, 'rgba(0, 217, 255, 0.4)');
        gradient.addColorStop(1, 'rgba(0, 217, 255, 0)');
        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(agentX, agentY, agentRadius * 1.5, 0, Math.PI * 2);
        ctx.fill();

        // Agent body
        ctx.fillStyle = COLORS.agentFill;
        ctx.beginPath();
        ctx.arc(agentX, agentY, agentRadius, 0, Math.PI * 2);
        ctx.fill();

        // Agent border
        ctx.strokeStyle = COLORS.agentBorder;
        ctx.lineWidth = 2;
        ctx.stroke();

        // Agent inner highlight
        ctx.fillStyle = 'rgba(255, 255, 255, 0.3)';
        ctx.beginPath();
        ctx.arc(agentX - agentRadius * 0.3, agentY - agentRadius * 0.3, agentRadius * 0.3, 0, Math.PI * 2);
        ctx.fill();

    }, [rows, cols, animatedPos, goal, values, policy, cellSize, width, height, holes]);

    return (
        <div className="gridworld-canvas-container">
            <canvas
                ref={canvasRef}
                width={width}
                height={height}
                className="gridworld-canvas"
            />
            <div className="gridworld-legend">
                <span className="legend-item">
                    <span className="legend-color start"></span> Start
                </span>
                <span className="legend-item">
                    <span className="legend-color goal"></span> Goal
                </span>
                {holes.length > 0 && (
                    <span className="legend-item">
                        <span className="legend-color hole"></span> Hole
                    </span>
                )}
                <span className="legend-item">
                    <span className="legend-color agent"></span> Agent
                </span>
            </div>
        </div>
    );
}

export default GridWorldCanvas;
