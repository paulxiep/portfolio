//! Price chart widget - displays price history as a line graph.

use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Style},
    symbols::Marker,
    text::Line,
    widgets::{Axis, Block, Borders, Chart, Dataset, GraphType, Widget},
};

/// Price chart widget displaying price history as a sparkline.
pub struct PriceChart<'a> {
    /// Price history data (most recent last).
    prices: &'a [f64],
    /// Chart title.
    title: &'a str,
}

impl<'a> PriceChart<'a> {
    /// Create a new price chart widget.
    pub fn new(prices: &'a [f64]) -> Self {
        Self {
            prices,
            title: "Price",
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
}
