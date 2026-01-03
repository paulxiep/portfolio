//! Stats panel widget - displays simulation statistics.

use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Widget},
};
use types::Price;

/// Simulation statistics panel widget.
pub struct StatsPanel {
    /// Current tick.
    pub tick: u64,
    /// Last trade price.
    pub last_price: Option<Price>,
    /// Total trades executed.
    pub total_trades: u64,
    /// Total orders submitted.
    pub total_orders: u64,
    /// Number of agents.
    pub agent_count: usize,
    /// Spread (if available).
    pub spread: Option<Price>,
}

impl StatsPanel {
    /// Create a new stats panel.
    pub fn new() -> Self {
        Self {
            tick: 0,
            last_price: None,
            total_trades: 0,
            total_orders: 0,
            agent_count: 0,
            spread: None,
        }
    }

    /// Set the current tick.
    pub fn tick(mut self, tick: u64) -> Self {
        self.tick = tick;
        self
    }

    /// Set the last price.
    pub fn last_price(mut self, price: Option<Price>) -> Self {
        self.last_price = price;
        self
    }

    /// Set total trades.
    pub fn total_trades(mut self, trades: u64) -> Self {
        self.total_trades = trades;
        self
    }

    /// Set total orders.
    pub fn total_orders(mut self, orders: u64) -> Self {
        self.total_orders = orders;
        self
    }

    /// Set agent count.
    pub fn agent_count(mut self, count: usize) -> Self {
        self.agent_count = count;
        self
    }

    /// Set spread.
    pub fn spread(mut self, spread: Option<Price>) -> Self {
        self.spread = spread;
        self
    }
}

impl Default for StatsPanel {
    fn default() -> Self {
        Self::new()
    }
}

impl Widget for StatsPanel {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let price_str = match self.last_price {
            Some(p) => format!("${:.2}", p.to_float()),
            None => "—".to_string(),
        };

        let spread_str = match self.spread {
            Some(s) => format!("${:.4}", s.to_float()),
            None => "—".to_string(),
        };

        let lines = vec![
            Line::from(vec![
                Span::styled("Tick: ", Style::default().fg(Color::Gray)),
                Span::styled(
                    format!("{}", self.tick),
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                ),
            ]),
            Line::from(vec![
                Span::styled("Last Price: ", Style::default().fg(Color::Gray)),
                Span::styled(price_str, Style::default().fg(Color::White)),
            ]),
            Line::from(vec![
                Span::styled("Spread: ", Style::default().fg(Color::Gray)),
                Span::styled(spread_str, Style::default().fg(Color::Yellow)),
            ]),
            Line::from(vec![
                Span::styled("Trades: ", Style::default().fg(Color::Gray)),
                Span::styled(
                    format!("{}", self.total_trades),
                    Style::default().fg(Color::Green),
                ),
            ]),
            Line::from(vec![
                Span::styled("Orders: ", Style::default().fg(Color::Gray)),
                Span::styled(
                    format!("{}", self.total_orders),
                    Style::default().fg(Color::White),
                ),
            ]),
            Line::from(vec![
                Span::styled("Agents: ", Style::default().fg(Color::Gray)),
                Span::styled(
                    format!("{}", self.agent_count),
                    Style::default().fg(Color::Magenta),
                ),
            ]),
        ];

        let para = Paragraph::new(lines).block(
            Block::default()
                .title("Simulation")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::White)),
        );

        para.render(area, buf);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stats_panel_default() {
        let panel = StatsPanel::default();
        let area = Rect::new(0, 0, 30, 10);
        let mut buf = Buffer::empty(area);
        panel.render(area, &mut buf);
    }

    #[test]
    fn test_stats_panel_with_data() {
        let panel = StatsPanel::new()
            .tick(500)
            .last_price(Some(Price::from_float(100.25)))
            .total_trades(42)
            .total_orders(150)
            .agent_count(12)
            .spread(Some(Price::from_float(0.50)));

        let area = Rect::new(0, 0, 30, 10);
        let mut buf = Buffer::empty(area);
        panel.render(area, &mut buf);
    }
}
