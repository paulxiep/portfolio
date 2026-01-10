/**
 * Simulation page - placeholder for V4.4
 *
 * SoC: Will handle simulation dashboard in V4.4
 */

import { Link } from 'react-router-dom';
import { Button } from '../components';

export function SimulationPage() {
  return (
    <div className="min-h-screen bg-gray-950">
      {/* Header */}
      <header className="bg-gray-900 border-b border-gray-800">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <Link
            to="/"
            className="text-xl font-bold text-gray-100 hover:text-primary-400 transition-colors"
          >
            ← Quant Trading Gym
          </Link>
          <Link to="/config">
            <Button variant="secondary">Configure</Button>
          </Link>
        </div>
      </header>

      {/* Placeholder Content */}
      <main className="flex flex-col items-center justify-center min-h-[calc(100vh-80px)] px-8">
        <div className="text-center max-w-2xl">
          <div className="text-6xl mb-6">📊</div>
          <h1 className="text-3xl font-bold text-gray-100 mb-4">Simulation Dashboard</h1>
          <p className="text-xl text-gray-400 mb-8">
            Coming in V4.4 — Real-time visualization with price charts, order book depth,
            indicators, and agent explorer.
          </p>

          <div className="bg-gray-900 rounded-xl border border-gray-800 p-6 text-left">
            <h2 className="text-lg font-semibold text-gray-200 mb-3">Planned Features:</h2>
            <ul className="text-gray-400 space-y-2">
              <li>• Price chart (candlestick + line modes)</li>
              <li>• Order book depth heatmap</li>
              <li>• Indicator panel (RSI, MACD, Bollinger, ATR)</li>
              <li>• Factor gauges (momentum, value, volatility)</li>
              <li>• Risk dashboard (VaR, Sharpe, drawdown)</li>
              <li>• News feed with sentiment tags</li>
              <li>• Agent Explorer with sortable P&L table</li>
              <li>• Time controls (Pause/Play, Speed, Step)</li>
            </ul>
          </div>

          <div className="mt-8">
            <Link to="/config">
              <Button variant="secondary" className="px-6">
                ← Back to Config
              </Button>
            </Link>
          </div>
        </div>
      </main>
    </div>
  );
}
