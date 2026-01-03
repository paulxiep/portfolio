//! Agent P&L table widget - displays agent positions and profits.

use ratatui::{
    buffer::Buffer,
    layout::{Constraint, Rect},
    style::{Color, Modifier, Style},
    text::Span,
    widgets::{Block, Borders, Cell, Row, Table, Widget},
};

use super::update::AgentInfo;

/// Agent P&L summary table widget.
///
/// Sorts noise traders by equity (descending), market makers at bottom.
pub struct AgentTable {
    /// Agent information to display (sorted).
    agents: Vec<AgentInfo>,
    /// Scroll offset for the agent list.
    scroll_offset: usize,
}

impl AgentTable {
    /// Create a new agent table widget.
    ///
    /// Agents are sorted: noise traders by equity (desc), then market makers at bottom.
    pub fn new(agents: &[AgentInfo]) -> Self {
        let mut sorted = agents.to_vec();
        sorted.sort_by(|a, b| {
            // Market makers always go to bottom
            match (a.is_market_maker, b.is_market_maker) {
                (true, false) => std::cmp::Ordering::Greater,
                (false, true) => std::cmp::Ordering::Less,
                _ => {
                    // Within same category, sort by equity (descending)
                    b.equity
                        .partial_cmp(&a.equity)
                        .unwrap_or(std::cmp::Ordering::Equal)
                }
            }
        });
        Self {
            agents: sorted,
            scroll_offset: 0,
        }
    }

    /// Set scroll offset.
    pub fn scroll_offset(mut self, offset: usize) -> Self {
        self.scroll_offset = offset;
        self
    }
}

impl Widget for AgentTable {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let header_cells = ["Agent", "Position", "Cash", "Realized P&L"]
            .iter()
            .map(|h| Cell::from(*h).style(Style::default().add_modifier(Modifier::BOLD)));
        let header = Row::new(header_cells)
            .style(Style::default().fg(Color::Yellow))
            .height(1);

        // Calculate visible rows (area height - border - header)
        let visible_rows = (area.height.saturating_sub(3)) as usize;
        let scroll_offset = self.scroll_offset.min(self.agents.len().saturating_sub(1));

        let rows = self
            .agents
            .iter()
            .skip(scroll_offset)
            .take(visible_rows)
            .map(|agent| {
                // Color position based on direction
                let position_style = if agent.position > 0 {
                    Style::default().fg(Color::Green)
                } else if agent.position < 0 {
                    Style::default().fg(Color::Red)
                } else {
                    Style::default().fg(Color::Gray)
                };

                // Color P&L based on profit/loss
                let pnl_value = agent.realized_pnl.to_float();
                let pnl_style = if pnl_value > 0.0 {
                    Style::default().fg(Color::Green)
                } else if pnl_value < 0.0 {
                    Style::default().fg(Color::Red)
                } else {
                    Style::default().fg(Color::Gray)
                };

                Row::new(vec![
                    Cell::from(agent.name.clone()),
                    Cell::from(format!("{:>8}", agent.position)).style(position_style),
                    Cell::from(format!("${:>10.2}", agent.cash.to_float())),
                    Cell::from(format!("${:>10.2}", pnl_value)).style(pnl_style),
                ])
            });

        let table = Table::new(
            rows,
            [
                Constraint::Min(15),    // Agent name
                Constraint::Length(10), // Position
                Constraint::Length(14), // Cash
                Constraint::Length(14), // P&L
            ],
        )
        .header(header)
        .block(
            Block::default()
                .title(Span::styled(
                    " Agent P&L ",
                    Style::default().fg(Color::White),
                ))
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::White)),
        );

        Widget::render(table, area, buf);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use types::Cash;

    #[test]
    fn test_agent_table_empty() {
        let agents: Vec<AgentInfo> = vec![];
        let widget = AgentTable::new(&agents);
        let area = Rect::new(0, 0, 60, 15);
        let mut buf = Buffer::empty(area);
        widget.render(area, &mut buf);
    }

    #[test]
    fn test_agent_table_with_data() {
        let agents = vec![
            AgentInfo {
                name: "01-MarketMaker".to_string(),
                position: 50,
                realized_pnl: Cash::from_float(125.50),
                cash: Cash::from_float(10_125.50),
                is_market_maker: true,
                equity: 10_125.50 + 50.0 * 100.0,
            },
            AgentInfo {
                name: "02-NoiseTrader".to_string(),
                position: -20,
                realized_pnl: Cash::from_float(-45.00),
                cash: Cash::from_float(9_955.00),
                is_market_maker: false,
                equity: 9_955.00 - 20.0 * 100.0,
            },
        ];
        let widget = AgentTable::new(&agents);
        let area = Rect::new(0, 0, 60, 10);
        let mut buf = Buffer::empty(area);
        widget.render(area, &mut buf);
    }
}
