//! Agent P&L table widget - displays agent positions and profits.

use ratatui::{
    buffer::Buffer,
    layout::{Constraint, Rect},
    style::{Color, Modifier, Style},
    widgets::{Block, Borders, Cell, Row, Table, Widget},
};

use super::update::AgentInfo;

/// Agent P&L summary table widget.
pub struct AgentTable<'a> {
    /// Agent information to display.
    agents: &'a [AgentInfo],
}

impl<'a> AgentTable<'a> {
    /// Create a new agent table widget.
    pub fn new(agents: &'a [AgentInfo]) -> Self {
        Self { agents }
    }
}

impl Widget for AgentTable<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let header_cells = ["Agent", "Position", "Cash", "Realized P&L"]
            .iter()
            .map(|h| Cell::from(*h).style(Style::default().add_modifier(Modifier::BOLD)));
        let header = Row::new(header_cells)
            .style(Style::default().fg(Color::Yellow))
            .height(1);

        let rows = self.agents.iter().map(|agent| {
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
                .title("Agent P&L")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::White)),
        )
        .row_highlight_style(Style::default().add_modifier(Modifier::REVERSED));

        // Render the table using the Widget trait
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
                name: "MM-1".to_string(),
                position: 50,
                realized_pnl: Cash::from_float(125.50),
                cash: Cash::from_float(10_125.50),
            },
            AgentInfo {
                name: "Noise-1".to_string(),
                position: -20,
                realized_pnl: Cash::from_float(-45.00),
                cash: Cash::from_float(9_955.00),
            },
        ];
        let widget = AgentTable::new(&agents);
        let area = Rect::new(0, 0, 60, 10);
        let mut buf = Buffer::empty(area);
        widget.render(area, &mut buf);
    }
}
