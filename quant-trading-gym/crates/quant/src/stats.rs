//! Statistical utilities for quantitative analysis.
//!
//! This module provides common statistical functions used across
//! indicator calculations, risk metrics, and factor scoring.

/// Calculate the mean of a slice of values.
pub fn mean(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    Some(values.iter().sum::<f64>() / values.len() as f64)
}

/// Calculate the variance of a slice of values (population variance).
pub fn variance(values: &[f64]) -> Option<f64> {
    let n = values.len();
    if n < 2 {
        return None;
    }

    let mean_val = mean(values)?;
    let sum_sq: f64 = values.iter().map(|v| (v - mean_val).powi(2)).sum();
    Some(sum_sq / n as f64)
}

/// Calculate the standard deviation (population).
pub fn std_dev(values: &[f64]) -> Option<f64> {
    variance(values).map(|v| v.sqrt())
}

/// Calculate the sample variance (n-1 denominator).
pub fn sample_variance(values: &[f64]) -> Option<f64> {
    let n = values.len();
    if n < 2 {
        return None;
    }

    let mean_val = mean(values)?;
    let sum_sq: f64 = values.iter().map(|v| (v - mean_val).powi(2)).sum();
    Some(sum_sq / (n - 1) as f64)
}

/// Calculate the sample standard deviation (n-1 denominator).
pub fn sample_std_dev(values: &[f64]) -> Option<f64> {
    sample_variance(values).map(|v| v.sqrt())
}

/// Calculate returns from a price series.
/// Returns (price[i] - price[i-1]) / price[i-1] for each consecutive pair.
pub fn returns(prices: &[f64]) -> Vec<f64> {
    if prices.len() < 2 {
        return vec![];
    }

    prices
        .windows(2)
        .filter_map(|w| {
            if w[0] != 0.0 {
                Some((w[1] - w[0]) / w[0])
            } else {
                None
            }
        })
        .collect()
}

/// Calculate log returns from a price series.
/// Returns ln(price[i] / price[i-1]) for each consecutive pair.
pub fn log_returns(prices: &[f64]) -> Vec<f64> {
    if prices.len() < 2 {
        return vec![];
    }

    prices
        .windows(2)
        .filter_map(|w| {
            if w[0] > 0.0 && w[1] > 0.0 {
                Some((w[1] / w[0]).ln())
            } else {
                None
            }
        })
        .collect()
}

/// Calculate covariance between two series.
pub fn covariance(x: &[f64], y: &[f64]) -> Option<f64> {
    if x.len() != y.len() || x.len() < 2 {
        return None;
    }

    let mean_x = mean(x)?;
    let mean_y = mean(y)?;
    let n = x.len();

    let sum: f64 = x
        .iter()
        .zip(y.iter())
        .map(|(xi, yi)| (xi - mean_x) * (yi - mean_y))
        .sum();

    Some(sum / n as f64)
}

/// Calculate Pearson correlation coefficient.
pub fn correlation(x: &[f64], y: &[f64]) -> Option<f64> {
    let cov = covariance(x, y)?;
    let std_x = std_dev(x)?;
    let std_y = std_dev(y)?;

    if std_x == 0.0 || std_y == 0.0 {
        return None;
    }

    Some(cov / (std_x * std_y))
}

/// Calculate beta (slope) of y with respect to x using linear regression.
pub fn beta(x: &[f64], y: &[f64]) -> Option<f64> {
    let cov = covariance(x, y)?;
    let var_x = variance(x)?;

    if var_x == 0.0 {
        return None;
    }

    Some(cov / var_x)
}

/// Calculate percentile value from a sorted slice.
/// Percentile should be between 0.0 and 1.0 (e.g., 0.95 for 95th percentile).
pub fn percentile(sorted_values: &[f64], pct: f64) -> Option<f64> {
    if sorted_values.is_empty() || !(0.0..=1.0).contains(&pct) {
        return None;
    }

    let n = sorted_values.len();
    if n == 1 {
        return Some(sorted_values[0]);
    }

    let idx = pct * (n - 1) as f64;
    let lower = idx.floor() as usize;
    let upper = idx.ceil() as usize;
    let frac = idx - lower as f64;

    if upper >= n {
        Some(sorted_values[n - 1])
    } else {
        Some(sorted_values[lower] * (1.0 - frac) + sorted_values[upper] * frac)
    }
}

/// Calculate exponential weighted moving average.
/// Alpha is the smoothing factor (higher = more weight on recent values).
pub fn ewma(values: &[f64], alpha: f64) -> Option<f64> {
    if values.is_empty() || !(0.0..=1.0).contains(&alpha) {
        return None;
    }

    let initial = values[0];
    Some(
        values
            .iter()
            .skip(1)
            .fold(initial, |prev, curr| alpha * curr + (1.0 - alpha) * prev),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean() {
        assert_eq!(mean(&[1.0, 2.0, 3.0, 4.0, 5.0]), Some(3.0));
        assert_eq!(mean(&[]), None);
    }

    #[test]
    fn test_std_dev() {
        let values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let std = std_dev(&values).unwrap();
        assert!((std - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_returns() {
        let prices = [100.0, 110.0, 99.0, 121.0];
        let rets = returns(&prices);
        assert_eq!(rets.len(), 3);
        assert!((rets[0] - 0.1).abs() < 0.0001); // 10% gain
        assert!((rets[1] - (-0.1)).abs() < 0.0001); // 10% loss
    }

    #[test]
    fn test_correlation() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [2.0, 4.0, 6.0, 8.0, 10.0];
        let corr = correlation(&x, &y).unwrap();
        assert!((corr - 1.0).abs() < 0.0001); // Perfect positive correlation
    }

    #[test]
    fn test_percentile() {
        let sorted = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        assert!((percentile(&sorted, 0.5).unwrap() - 5.5).abs() < 0.0001);
        assert!((percentile(&sorted, 0.0).unwrap() - 1.0).abs() < 0.0001);
        assert!((percentile(&sorted, 1.0).unwrap() - 10.0).abs() < 0.0001);
    }
}
