//! Main TUI application - composes widgets and handles rendering loop.

use std::io::{self, Stdout};
use std::time::{Duration, Instant};

use crossbeam_channel::Receiver;
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{
    Frame, Terminal,
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::Paragraph,
};

use crate::widgets::{AgentTable, BookDepth, PriceChart, SimUpdate, StatsPanel};

/// TUI application state.
pub struct TuiApp {
    /// Channel receiver for simulation updates.
    receiver: Receiver<SimUpdate>,
    /// Latest simulation state.
    state: SimUpdate,
    /// Whether the simulation has finished.
    finished: bool,
    /// Target frame rate.
    frame_rate: u64,
}

impl TuiApp {
    /// Create a new TUI app with the given channel receiver.
    ///
    /// Uses `try_iter()` internally for non-blocking updates.
    pub fn new(receiver: Receiver<SimUpdate>) -> Self {
        Self {
            receiver,
            state: SimUpdate::default(),
            finished: false,
            frame_rate: 30, // 30 FPS
        }
    }

    /// Set the target frame rate (frames per second).
    pub fn frame_rate(mut self, fps: u64) -> Self {
        self.frame_rate = fps;
        self
    }

    /// Run the TUI event loop.
    ///
    /// Blocks until the user presses 'q' or the simulation finishes.
    pub fn run(mut self) -> io::Result<()> {
        // Setup terminal
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen)?;
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;

        let result = self.run_loop(&mut terminal);

        // Restore terminal
        disable_raw_mode()?;
        execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
        terminal.show_cursor()?;

        result
    }

    /// Main event loop.
    fn run_loop(&mut self, terminal: &mut Terminal<CrosstermBackend<Stdout>>) -> io::Result<()> {
        let tick_rate = Duration::from_millis(1000 / self.frame_rate);
        let mut last_tick = Instant::now();

        loop {
            // Draw current state
            terminal.draw(|f| self.draw(f))?;

            // Handle input with timeout
            let timeout = tick_rate
                .checked_sub(last_tick.elapsed())
                .unwrap_or_else(|| Duration::from_secs(0));

            if event::poll(timeout)?
                && let Event::Key(key) = event::read()?
                && key.kind == KeyEventKind::Press
            {
                match key.code {
                    KeyCode::Char('q') | KeyCode::Esc => return Ok(()),
                    _ => {}
                }
            }

            // Update state from channel (non-blocking)
            if last_tick.elapsed() >= tick_rate {
                self.poll_updates();
                last_tick = Instant::now();
            }

            // Exit if simulation finished and user has seen it
            if self.finished {
                // Keep showing until user presses q
            }
        }
    }

    /// Poll for updates from the simulation channel (non-blocking).
    fn poll_updates(&mut self) {
        // Drain all currently available updates, keep the latest
        // try_iter() is non-blocking - returns only items currently in the channel
        for update in self.receiver.try_iter() {
            if update.finished {
                self.finished = true;
            }
            self.state = update;
        }
    }

    /// Draw the UI.
    fn draw(&self, frame: &mut Frame) {
        let area = frame.area();

        // Main layout: header + content
        let main_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1), // Header
                Constraint::Min(0),    // Content
                Constraint::Length(1), // Footer
            ])
            .split(area);

        // Render header
        self.draw_header(frame, main_chunks[0]);

        // Content layout: left panel (stats + book) + right panel (chart + agents)
        let content_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(35), Constraint::Percentage(65)])
            .split(main_chunks[1]);

        // Left panel: stats + order book
        let left_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(9), Constraint::Min(0)])
            .split(content_chunks[0]);

        // Right panel: price chart + agent table
        let right_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(content_chunks[1]);

        // Draw widgets
        self.draw_stats(frame, left_chunks[0]);
        self.draw_book_depth(frame, left_chunks[1]);
        self.draw_price_chart(frame, right_chunks[0]);
        self.draw_agent_table(frame, right_chunks[1]);

        // Render footer
        self.draw_footer(frame, main_chunks[2]);
    }

    /// Draw the header bar.
    fn draw_header(&self, frame: &mut Frame, area: Rect) {
        let status = if self.finished {
            Span::styled(
                " FINISHED ",
                Style::default()
                    .fg(Color::Black)
                    .bg(Color::Green)
                    .add_modifier(Modifier::BOLD),
            )
        } else {
            Span::styled(
                " RUNNING ",
                Style::default()
                    .fg(Color::Black)
                    .bg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            )
        };

        let title = Line::from(vec![
            Span::styled(
                "Quant Trading Gym",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw(" │ "),
            status,
        ]);

        let header = Paragraph::new(title).style(Style::default().bg(Color::DarkGray));
        frame.render_widget(header, area);
    }

    /// Draw the footer bar.
    fn draw_footer(&self, frame: &mut Frame, area: Rect) {
        let footer = Paragraph::new(Line::from(vec![
            Span::styled(" q", Style::default().fg(Color::Yellow)),
            Span::raw(" Quit"),
        ]))
        .style(Style::default().bg(Color::DarkGray));
        frame.render_widget(footer, area);
    }

    /// Draw the stats panel.
    fn draw_stats(&self, frame: &mut Frame, area: Rect) {
        // Calculate spread from book data
        let spread = match (self.state.bids.first(), self.state.asks.first()) {
            (Some(bid), Some(ask)) => Some(ask.price - bid.price),
            _ => None,
        };

        let stats = StatsPanel::new()
            .tick(self.state.tick)
            .last_price(self.state.last_price)
            .total_trades(self.state.total_trades)
            .total_orders(self.state.total_orders)
            .agent_count(self.state.agents.len())
            .spread(spread);

        frame.render_widget(stats, area);
    }

    /// Draw the order book depth.
    fn draw_book_depth(&self, frame: &mut Frame, area: Rect) {
        let book = BookDepth::new(&self.state.bids, &self.state.asks).max_levels(10);
        frame.render_widget(book, area);
    }

    /// Draw the price chart.
    fn draw_price_chart(&self, frame: &mut Frame, area: Rect) {
        let title = match self.state.last_price {
            Some(p) => format!("Price: ${:.2}", p.to_float()),
            None => "Price".to_string(),
        };
        let chart = PriceChart::new(&self.state.price_history).title(&title);
        frame.render_widget(chart, area);
    }

    /// Draw the agent P&L table.
    fn draw_agent_table(&self, frame: &mut Frame, area: Rect) {
        let table = AgentTable::new(&self.state.agents);
        frame.render_widget(table, area);
    }
}

/// A simpler TUI that doesn't use channels - for direct integration.
pub struct SimpleTui {
    /// Current state to display.
    state: SimUpdate,
}

impl SimpleTui {
    /// Create a new simple TUI.
    pub fn new() -> Self {
        Self {
            state: SimUpdate::default(),
        }
    }

    /// Update the display state.
    pub fn update(&mut self, state: SimUpdate) {
        self.state = state;
    }

    /// Draw a single frame to the given terminal.
    pub fn draw(&self, frame: &mut Frame) {
        // Create a dummy channel just for drawing (never used)
        let (_tx, rx) = crossbeam_channel::unbounded::<SimUpdate>();
        let app = TuiApp {
            receiver: rx,
            state: self.state.clone(),
            finished: self.state.finished,
            frame_rate: 30,
        };
        app.draw(frame);
    }

    /// Initialize the terminal for TUI rendering.
    pub fn init() -> io::Result<Terminal<CrosstermBackend<Stdout>>> {
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen)?;
        let backend = CrosstermBackend::new(stdout);
        Terminal::new(backend)
    }

    /// Restore the terminal to normal state.
    pub fn restore(terminal: &mut Terminal<CrosstermBackend<Stdout>>) -> io::Result<()> {
        disable_raw_mode()?;
        execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
        terminal.show_cursor()?;
        Ok(())
    }
}

impl Default for SimpleTui {
    fn default() -> Self {
        Self::new()
    }
}

/// Check if a key was pressed (non-blocking).
pub fn check_quit() -> io::Result<bool> {
    if event::poll(Duration::from_millis(0))?
        && let Event::Key(key) = event::read()?
        && key.kind == KeyEventKind::Press
        && (key.code == KeyCode::Char('q') || key.code == KeyCode::Esc)
    {
        return Ok(true);
    }
    Ok(false)
}
