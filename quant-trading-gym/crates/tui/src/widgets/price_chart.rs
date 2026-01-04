//! Price chart widget - displays price history as a line graph.
//!
//! # V2.3 Multi-Symbol Support
//!
//! The chart supports two modes:
//! - Single symbol: Shows one price line (cyan)
//! - Multi-symbol overlay: Shows multiple price lines with different colors

use std::collections::HashMap;

use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Style},
    symbols::Marker,
    text::Line,
    widgets::{Axis, Block, Borders, Chart, Dataset, GraphType, Widget},
};

use types::Symbol;

/// Colors for multi-symbol overlay mode.
const OVERLAY_COLORS: &[Color] = &[
    Color::Cyan,
    Color::Green,
    Color::Yellow,
    Color::Magenta,
    Color::Red,
    Color::Blue,
    Color::LightCyan,
    Color::LightGreen,
];

/// Type alias for multi-symbol price data references.
type MultiPriceData<'a> = (&'a HashMap<Symbol, Vec<f64>>, &'a [Symbol]);

/// Price chart widget displaying price history as a sparkline.
pub struct PriceChart<'a> {
    /// Single symbol price data.
    prices: &'a [f64],
    /// Multi-symbol price data (for overlay mode).
    multi_prices: Option<MultiPriceData<'a>>,
    /// Chart title.
    title: &'a str,
}

impl<'a> PriceChart<'a> {
    /// Create a new price chart widget for a single symbol.
    pub fn new(prices: &'a [f64]) -> Self {
        Self {
            prices,
            multi_prices: None,
            title: "Price",
        }
    }

    /// Create a new price chart widget for multiple symbols (overlay mode).
    pub fn multi(prices: &'a HashMap<Symbol, Vec<f64>>, symbols: &'a [Symbol]) -> Self {
        Self {
            prices: &[],
            multi_prices: Some((prices, symbols)),
            title: "Price (Overlay)",
        }
    }

    /// Set the chart title.
    pub fn title(mut self, title: &'a str) -> Self {
        self.title = title;
        self
    }
}

impl Widget for PriceChart<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        if let Some((prices_map, symbols)) = self.multi_prices {
            self.render_multi(prices_map, symbols, area, buf);
        } else {
            self.render_single(area, buf);
        }
    }
}

impl PriceChart<'_> {
    /// Render single symbol chart.
    fn render_single(self, area: Rect, buf: &mut Buffer) {
        if self.prices.is_empty() {
            // Render empty chart with "No data" message
            let block = Block::default()
                .title(self.title)
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::DarkGray));
            block.render(area, buf);
            return;
        }

        // Calculate bounds for Y axis
        let min_price = self.prices.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_price = self
            .prices
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        // Add some padding to Y bounds
        let y_padding = ((max_price - min_price) * 0.1).max(0.01);
        let y_min = min_price - y_padding;
        let y_max = max_price + y_padding;

        // Prepare data points: (x, y) where x is the index
        let data: Vec<(f64, f64)> = self
            .prices
            .iter()
            .enumerate()
            .map(|(i, &p)| (i as f64, p))
            .collect();

        let x_max = (self.prices.len().saturating_sub(1)) as f64;

        let dataset = Dataset::default()
            .marker(Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(Color::Cyan))
            .data(&data);

        let y_labels: Vec<Line> = vec![
            Line::from(format!("{:.2}", y_min)),
            Line::from(format!("{:.2}", (y_min + y_max) / 2.0)),
            Line::from(format!("{:.2}", y_max)),
        ];

        let chart = Chart::new(vec![dataset])
            .block(
                Block::default()
                    .title(self.title)
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::White)),
            )
            .x_axis(
                Axis::default()
                    .style(Style::default().fg(Color::Gray))
                    .bounds([0.0, x_max.max(1.0)])
                    .labels::<Vec<Line>>(vec![]),
            )
            .y_axis(
                Axis::default()
                    .style(Style::default().fg(Color::Gray))
                    .bounds([y_min, y_max])
                    .labels(y_labels),
            );

        chart.render(area, buf);
    }

    /// Render multi-symbol overlay chart.
    fn render_multi(
        self,
        prices_map: &HashMap<Symbol, Vec<f64>>,
        symbols: &[Symbol],
        area: Rect,
        buf: &mut Buffer,
    ) {
        // Collect all price data
        let all_prices: Vec<_> = symbols
            .iter()
            .filter_map(|s| prices_map.get(s))
            .filter(|v| !v.is_empty())
            .collect();

        if all_prices.is_empty() {
            let block = Block::default()
                .title(self.title)
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::DarkGray));
            block.render(area, buf);
            return;
        }

        // Calculate global Y bounds across all symbols
        let min_price = all_prices
            .iter()
            .flat_map(|v| v.iter())
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let max_price = all_prices
            .iter()
            .flat_map(|v| v.iter())
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        let y_padding = ((max_price - min_price) * 0.1).max(0.01);
        let y_min = min_price - y_padding;
        let y_max = max_price + y_padding;

        // Find max X
        let x_max = all_prices.iter().map(|v| v.len()).max().unwrap_or(1) as f64 - 1.0;

        // Build datasets with owned data
        // We need to store the data vectors to keep them alive for the chart
        let data_vecs: Vec<Vec<(f64, f64)>> = symbols
            .iter()
            .filter_map(|s| prices_map.get(s))
            .map(|prices| {
                prices
                    .iter()
                    .enumerate()
                    .map(|(i, &p)| (i as f64, p))
                    .collect()
            })
            .collect();

        let datasets: Vec<Dataset> = data_vecs
            .iter()
            .enumerate()
            .map(|(i, data)| {
                let color = OVERLAY_COLORS[i % OVERLAY_COLORS.len()];
                Dataset::default()
                    .marker(Marker::Braille)
                    .graph_type(GraphType::Line)
                    .style(Style::default().fg(color))
                    .data(data)
            })
            .collect();

        let y_labels: Vec<Line> = vec![
            Line::from(format!("{:.2}", y_min)),
            Line::from(format!("{:.2}", (y_min + y_max) / 2.0)),
            Line::from(format!("{:.2}", y_max)),
        ];

        let chart = Chart::new(datasets)
            .block(
                Block::default()
                    .title(self.title)
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::White)),
            )
            .x_axis(
                Axis::default()
                    .style(Style::default().fg(Color::Gray))
                    .bounds([0.0, x_max.max(1.0)])
                    .labels::<Vec<Line>>(vec![]),
            )
            .y_axis(
                Axis::default()
                    .style(Style::default().fg(Color::Gray))
                    .bounds([y_min, y_max])
                    .labels(y_labels),
            );

        chart.render(area, buf);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_price_chart_empty() {
        let prices: Vec<f64> = vec![];
        let chart = PriceChart::new(&prices);
        let area = Rect::new(0, 0, 40, 10);
        let mut buf = Buffer::empty(area);
        chart.render(area, &mut buf);
        // Should not panic
    }

    #[test]
    fn test_price_chart_with_data() {
        let prices = vec![100.0, 101.0, 99.5, 102.0, 100.5];
        let chart = PriceChart::new(&prices).title("Test Price");
        let area = Rect::new(0, 0, 60, 15);
        let mut buf = Buffer::empty(area);
        chart.render(area, &mut buf);
        // Should render without panic
    }

    #[test]
    fn test_price_chart_multi_symbol() {
        let mut prices = HashMap::new();
        prices.insert("AAPL".to_string(), vec![150.0, 151.0, 149.0]);
        prices.insert("GOOG".to_string(), vec![2800.0, 2810.0, 2790.0]);
        let symbols = vec!["AAPL".to_string(), "GOOG".to_string()];
        let chart = PriceChart::multi(&prices, &symbols);
        let area = Rect::new(0, 0, 60, 15);
        let mut buf = Buffer::empty(area);
        chart.render(area, &mut buf);
        // Should render without panic
    }
}
