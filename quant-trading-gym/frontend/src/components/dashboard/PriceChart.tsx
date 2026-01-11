/**
 * Price chart component with candlesticks and line chart modes (V4.4).
 *
 * Renders OHLCV data as either candlestick or line chart using SVG.
 * No external charting library - pure React/CSS for minimal dependencies.
 *
 * Declarative: Data-driven rendering from CandlesResponse
 * Modular: Self-contained chart logic with configurable display type
 * SoC: Only handles price visualization
 */

import type { CandlesResponse, Candle } from '../../types/api';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** Chart display type */
export type ChartType = 'candlestick' | 'line';

export interface PriceChartProps {
  /** Candle data from API. */
  data: CandlesResponse | null;
  /** Loading state. */
  loading?: boolean;
  /** Error state. */
  error?: Error | null;
  /** Chart height in pixels. */
  height?: number;
  /** Number of candles to display. */
  maxCandles?: number;
  /** Chart type: 'candlestick' or 'line' (default: candlestick). */
  chartType?: ChartType;
  /** Callback when chart type changes. */
  onChartTypeChange?: (type: ChartType) => void;
  /** Whether to show chart type toggle (default: true). */
  showChartToggle?: boolean;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Calculate price range for scaling. */
function getPriceRange(candles: Candle[]): { min: number; max: number } {
  if (candles.length === 0) return { min: 0, max: 100 };

  let min = Infinity;
  let max = -Infinity;

  for (const c of candles) {
    if (c.low < min) min = c.low;
    if (c.high > max) max = c.high;
  }

  // Add 5% padding
  const padding = (max - min) * 0.05;
  return { min: min - padding, max: max + padding };
}

/** Scale price to pixel position. */
function scaleY(price: number, min: number, max: number, height: number): number {
  return height - ((price - min) / (max - min)) * height;
}

/** Format price for display. */
function formatPrice(price: number): string {
  return price.toFixed(2);
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

interface CandleBarProps {
  candle: Candle;
  index: number;
  priceMin: number;
  priceMax: number;
  height: number;
  candleWidth: number;
}

function CandleBar({ candle, index, priceMin, priceMax, height, candleWidth }: CandleBarProps) {
  const { open, high, low, close } = candle;
  const isGreen = close >= open;

  // Calculate positions
  const highY = scaleY(high, priceMin, priceMax, height);
  const lowY = scaleY(low, priceMin, priceMax, height);
  const openY = scaleY(open, priceMin, priceMax, height);
  const closeY = scaleY(close, priceMin, priceMax, height);

  // Body dimensions
  const bodyTop = Math.min(openY, closeY);
  const bodyHeight = Math.max(1, Math.abs(closeY - openY));

  // Wick X position
  const x = index * candleWidth + candleWidth / 2;

  const color = isGreen ? '#22c55e' : '#ef4444'; // green-500 / red-500

  return (
    <g>
      {/* Wick (high-low line) */}
      <line x1={x} y1={highY} x2={x} y2={lowY} stroke={color} strokeWidth={1} />
      {/* Body (open-close rect) */}
      <rect
        x={x - candleWidth * 0.35}
        y={bodyTop}
        width={candleWidth * 0.7}
        height={bodyHeight}
        fill={isGreen ? color : color}
        stroke={color}
        strokeWidth={1}
      />
    </g>
  );
}

interface PriceAxisProps {
  min: number;
  max: number;
  height: number;
  width: number;
}

function PriceAxis({ min, max, height, width }: PriceAxisProps) {
  // Generate ~5 tick marks
  const ticks: number[] = [];
  const step = (max - min) / 5;
  for (let i = 0; i <= 5; i++) {
    ticks.push(min + step * i);
  }

  return (
    <g className="text-gray-400 text-xs">
      {ticks.map((price) => {
        const y = scaleY(price, min, max, height);
        return (
          <g key={price}>
            <line
              x1={0}
              y1={y}
              x2={width}
              y2={y}
              stroke="#374151"
              strokeWidth={1}
              strokeDasharray="4,4"
            />
            <text x={width + 4} y={y + 4} fill="#9ca3af" fontSize={10}>
              {formatPrice(price)}
            </text>
          </g>
        );
      })}
    </g>
  );
}

// ---------------------------------------------------------------------------
// Line chart component
// ---------------------------------------------------------------------------

interface LineChartProps {
  candles: Candle[];
  priceMin: number;
  priceMax: number;
  height: number;
  candleWidth: number;
}

function LineChart({ candles, priceMin, priceMax, height, candleWidth }: LineChartProps) {
  if (candles.length === 0) return null;

  // Build path using close prices
  const points = candles.map((c, i) => {
    const x = i * candleWidth + candleWidth / 2;
    const y = scaleY(c.close, priceMin, priceMax, height);
    return `${x},${y}`;
  });

  const pathD = `M ${points.join(' L ')}`;

  // Determine trend color based on first vs last close
  const firstClose = candles[0].close;
  const lastClose = candles[candles.length - 1].close;
  const color = lastClose >= firstClose ? '#22c55e' : '#ef4444';

  // Create gradient fill under the line
  const gradientId = `lineGradient-${Date.now()}`;
  const areaPath = `${pathD} L ${(candles.length - 1) * candleWidth + candleWidth / 2},${height} L ${candleWidth / 2},${height} Z`;

  return (
    <g>
      {/* Gradient definition */}
      <defs>
        <linearGradient id={gradientId} x1="0%" y1="0%" x2="0%" y2="100%">
          <stop offset="0%" stopColor={color} stopOpacity={0.3} />
          <stop offset="100%" stopColor={color} stopOpacity={0.05} />
        </linearGradient>
      </defs>
      {/* Area fill */}
      <path d={areaPath} fill={`url(#${gradientId})`} />
      {/* Line */}
      <path d={pathD} fill="none" stroke={color} strokeWidth={2} strokeLinejoin="round" />
      {/* Current price dot */}
      <circle
        cx={(candles.length - 1) * candleWidth + candleWidth / 2}
        cy={scaleY(lastClose, priceMin, priceMax, height)}
        r={4}
        fill={color}
      />
    </g>
  );
}

// ---------------------------------------------------------------------------
// Chart type toggle
// ---------------------------------------------------------------------------

interface ChartToggleProps {
  chartType: ChartType;
  onChange: (type: ChartType) => void;
}

function ChartToggle({ chartType, onChange }: ChartToggleProps) {
  return (
    <div className="flex items-center gap-1 bg-gray-800 rounded p-0.5">
      <button
        className={`px-2 py-1 text-xs rounded transition-colors ${
          chartType === 'line' ? 'bg-primary-600 text-white' : 'text-gray-400 hover:text-white'
        }`}
        onClick={() => onChange('line')}
        title="Line Chart"
      >
        <svg
          className="w-4 h-4"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
        >
          <polyline points="4,16 8,12 12,14 16,8 20,10" />
        </svg>
      </button>
      <button
        className={`px-2 py-1 text-xs rounded transition-colors ${
          chartType === 'candlestick'
            ? 'bg-primary-600 text-white'
            : 'text-gray-400 hover:text-white'
        }`}
        onClick={() => onChange('candlestick')}
        title="Candlestick Chart"
      >
        <svg
          className="w-4 h-4"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
        >
          <line x1="8" y1="4" x2="8" y2="20" />
          <rect x="6" y="8" width="4" height="8" fill="currentColor" />
          <line x1="16" y1="6" x2="16" y2="18" />
          <rect x="14" y="10" width="4" height="6" fill="none" />
        </svg>
      </button>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export function PriceChart({
  data,
  loading = false,
  error = null,
  height = 300,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  maxCandles: _maxCandles = 200, // Reserved for future pagination/limiting
  chartType = 'candlestick',
  onChartTypeChange,
  showChartToggle = true,
}: PriceChartProps) {
  // Loading state
  if (loading && !data) {
    return (
      <div
        className="bg-gray-900 rounded-lg border border-gray-700 flex items-center justify-center"
        style={{ height }}
      >
        <div className="text-gray-400 animate-pulse">Loading chart...</div>
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <div
        className="bg-gray-900 rounded-lg border border-red-700 flex items-center justify-center"
        style={{ height }}
      >
        <div className="text-red-400">Error: {error.message}</div>
      </div>
    );
  }

  // No data state
  if (!data || data.candles.length === 0) {
    return (
      <div
        className="bg-gray-900 rounded-lg border border-gray-700 flex items-center justify-center"
        style={{ height }}
      >
        <div className="text-gray-500">No candle data</div>
      </div>
    );
  }

  // Use all candles (no slicing - show full history)
  const candles = data.candles;
  const { min, max } = getPriceRange(candles);

  // Calculate dimensions - responsive width with minimum candle size
  const axisWidth = 60;
  const minCandleWidth = 12; // Minimum pixels per candle for readability
  // Chart width scales with candle count, minimum 100% container width
  const chartWidth = Math.max(800, candles.length * minCandleWidth);
  const candleWidth = chartWidth / Math.max(candles.length, 1);
  const totalWidth = chartWidth + axisWidth;

  // Current price info
  const lastCandle = candles[candles.length - 1];
  const priceChange = lastCandle.close - lastCandle.open;
  const priceChangePercent = (priceChange / lastCandle.open) * 100;

  // Handle chart type toggle
  const handleChartTypeChange = (type: ChartType) => {
    onChartTypeChange?.(type);
  };

  return (
    <div className="bg-gray-900 rounded-lg border border-gray-700 p-4">
      {/* Header */}
      <div className="flex justify-between items-center mb-3">
        <div className="flex items-center gap-3">
          <span className="text-white font-semibold">{data.symbol}</span>
          <span className="text-gray-400 text-sm">{candles.length} candles</span>
        </div>
        <div className="flex items-center gap-3">
          {showChartToggle && onChartTypeChange && (
            <ChartToggle chartType={chartType} onChange={handleChartTypeChange} />
          )}
          <div className="flex items-center gap-2">
            <span className="text-white font-mono">{formatPrice(lastCandle.close)}</span>
            <span
              className={`text-sm font-mono ${priceChange >= 0 ? 'text-green-500' : 'text-red-500'}`}
            >
              {priceChange >= 0 ? '+' : ''}
              {formatPrice(priceChange)} ({priceChangePercent.toFixed(2)}%)
            </span>
          </div>
        </div>
      </div>

      {/* Scrollable chart container - fills width, scrolls when content exceeds */}
      <div
        className="overflow-x-auto scrollbar-thin scrollbar-thumb-gray-600 scrollbar-track-gray-800"
        style={{ height: height + 20 }}
      >
        <svg
          width={totalWidth}
          height={height}
          viewBox={`0 0 ${totalWidth} ${height}`}
          className="block"
        >
          {/* Grid and axis */}
          <PriceAxis min={min} max={max} height={height} width={chartWidth} />

          {/* Render based on chart type */}
          {chartType === 'line' ? (
            <LineChart
              candles={candles}
              priceMin={min}
              priceMax={max}
              height={height}
              candleWidth={candleWidth}
            />
          ) : (
            <g>
              {candles.map((candle, i) => (
                <CandleBar
                  key={candle.tick}
                  candle={candle}
                  index={i}
                  priceMin={min}
                  priceMax={max}
                  height={height}
                  candleWidth={candleWidth}
                />
              ))}
            </g>
          )}
        </svg>
      </div>
    </div>
  );
}
