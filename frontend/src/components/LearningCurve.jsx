import { useRef, useEffect } from 'react';
import './LearningCurve.css';

/**
 * Learning Curve visualization using Canvas.
 * Displays TD errors, rewards, or other metrics over time.
 * 
 * Props:
 *   data - Array of data points to plot
 *   label - Label for the y-axis
 *   color - Line color (default: cyan)
 *   width - Canvas width (default: 400)
 *   height - Canvas height (default: 200)
 *   showGrid - Whether to show grid lines
 */
function LearningCurve({
    data = [],
    label = 'Value',
    color = '#00d9ff',
    width = 400,
    height = 200,
    showGrid = true
}) {
    const canvasRef = useRef(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        const padding = { top: 20, right: 20, bottom: 30, left: 50 };
        const plotWidth = width - padding.left - padding.right;
        const plotHeight = height - padding.top - padding.bottom;

        // Clear canvas
        ctx.fillStyle = '#1a1a2e';
        ctx.fillRect(0, 0, width, height);

        if (data.length === 0) {
            ctx.fillStyle = '#666';
            ctx.font = '14px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('No data yet', width / 2, height / 2);
            return;
        }

        // Calculate ranges
        const minVal = Math.min(...data);
        const maxVal = Math.max(...data);
        const range = maxVal - minVal || 1;

        // Draw grid
        if (showGrid) {
            ctx.strokeStyle = '#333';
            ctx.lineWidth = 1;

            // Horizontal grid lines
            for (let i = 0; i <= 4; i++) {
                const y = padding.top + (plotHeight / 4) * i;
                ctx.beginPath();
                ctx.moveTo(padding.left, y);
                ctx.lineTo(width - padding.right, y);
                ctx.stroke();
            }

            // Vertical grid lines
            for (let i = 0; i <= 4; i++) {
                const x = padding.left + (plotWidth / 4) * i;
                ctx.beginPath();
                ctx.moveTo(x, padding.top);
                ctx.lineTo(x, height - padding.bottom);
                ctx.stroke();
            }
        }

        // Draw axes
        ctx.strokeStyle = '#555';
        ctx.lineWidth = 2;

        // Y-axis
        ctx.beginPath();
        ctx.moveTo(padding.left, padding.top);
        ctx.lineTo(padding.left, height - padding.bottom);
        ctx.stroke();

        // X-axis
        ctx.beginPath();
        ctx.moveTo(padding.left, height - padding.bottom);
        ctx.lineTo(width - padding.right, height - padding.bottom);
        ctx.stroke();

        // Draw labels
        ctx.fillStyle = '#888';
        ctx.font = '11px sans-serif';
        ctx.textAlign = 'right';

        // Y-axis labels
        for (let i = 0; i <= 4; i++) {
            const val = maxVal - (range / 4) * i;
            const y = padding.top + (plotHeight / 4) * i;
            ctx.fillText(val.toFixed(2), padding.left - 5, y + 4);
        }

        // X-axis labels
        ctx.textAlign = 'center';
        ctx.fillText('0', padding.left, height - padding.bottom + 15);
        ctx.fillText(String(data.length), width - padding.right, height - padding.bottom + 15);

        // Y-axis label
        ctx.save();
        ctx.translate(12, height / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.textAlign = 'center';
        ctx.fillText(label, 0, 0);
        ctx.restore();

        // X-axis label
        ctx.fillText('Step', width / 2, height - 5);

        // Draw line
        if (data.length > 1) {
            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.beginPath();

            for (let i = 0; i < data.length; i++) {
                const x = padding.left + (i / (data.length - 1)) * plotWidth;
                const y = padding.top + ((maxVal - data[i]) / range) * plotHeight;

                if (i === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
            }

            ctx.stroke();

            // Draw area fill
            ctx.lineTo(padding.left + plotWidth, height - padding.bottom);
            ctx.lineTo(padding.left, height - padding.bottom);
            ctx.closePath();
            ctx.fillStyle = color.replace(')', ', 0.1)').replace('rgb', 'rgba');
            if (color.startsWith('#')) {
                ctx.fillStyle = color + '1a';
            }
            ctx.fill();
        }

        // Draw current value indicator
        if (data.length > 0) {
            const lastVal = data[data.length - 1];
            const x = padding.left + plotWidth;
            const y = padding.top + ((maxVal - lastVal) / range) * plotHeight;

            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.arc(x, y, 4, 0, Math.PI * 2);
            ctx.fill();

            // Value text
            ctx.fillStyle = '#fff';
            ctx.font = 'bold 12px monospace';
            ctx.textAlign = 'left';
            ctx.fillText(lastVal.toFixed(4), x + 8, y + 4);
        }

    }, [data, label, color, width, height, showGrid]);

    return (
        <div className="learning-curve">
            <canvas
                ref={canvasRef}
                width={width}
                height={height}
                className="learning-curve-canvas"
            />
        </div>
    );
}

export default LearningCurve;
