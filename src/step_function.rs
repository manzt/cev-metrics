use std::cmp::Ordering;

#[derive(Debug, PartialEq)]
pub enum Side {
    Left,
    Right,
}

pub struct StepFunction {
    x: Vec<f64>,
    y: Vec<f64>,
    side: Side,
}

impl StepFunction {
    pub fn new(x: &[f64], y: &[f64], side: Side) -> Self {
        let mut x = x.to_vec();
        let mut y = y.to_vec();
        x.insert(0, f64::NEG_INFINITY);
        y.insert(0, 0.0);
        StepFunction { x, y, side }
    }

    pub fn evaluate(&self, value: f64) -> f64 {
        let idx = match self.side {
            Side::Left => self.x.partition_point(|&x| x < value),
            Side::Right => self.x.partition_point(|&x| x <= value),
        };
        self.y[idx.saturating_sub(1)]
    }
}

pub struct ECDF {
    step_function: StepFunction,
}

impl ECDF {
    pub fn new(data: &[f64], side: Side) -> Self {
        let mut data = data.to_vec();
        data.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        let n = data.len() as f64;
        let y: Vec<_> = (1..=data.len()).map(|i| i as f64 / n).collect();
        let step_function = StepFunction::new(&data, &y, side);
        Self { step_function }
    }

    pub fn evaluate(&self, value: f64) -> f64 {
        self.step_function.evaluate(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_step_function() {
        let x = (0..20).map(|x| x as f64).collect::<Vec<_>>();
        let y = (0..20).map(|y| y as f64).collect::<Vec<_>>();
        let f = StepFunction::new(&x, &y, Side::Left);
        let input = vec![3.2, 4.5, 24.0, -3.1, 3.0, 4.0];
        let vals: Vec<_> = input.iter().map(|x| f.evaluate(*x)).collect();
        assert_eq!(vals, vec![3.0, 4.0, 19.0, 0.0, 2.0, 3.0]);
    }

    #[test]
    fn test_step_function_value_side_right() {
        let x = (0..20).map(|x| x as f64).collect::<Vec<_>>();
        let y = (0..20).map(|y| y as f64).collect::<Vec<_>>();
        let f = StepFunction::new(&x, &y, Side::Right);
        let input = vec![3.2, 4.5, 24.0, -3.1, 3.0, 4.0];
        let vals: Vec<_> = input.iter().map(|x| f.evaluate(*x)).collect();
        assert_eq!(vals, vec![3.0, 4.0, 19.0, 0.0, 3.0, 4.0]);
    }

    #[test]
    fn test_step_function_repeated_values1() {
        let x = vec![1., 1., 2., 2., 2., 3., 3., 3., 4., 5.];
        let y = vec![6., 7., 8., 9., 10., 11., 12., 13., 14., 15.];
        let f = StepFunction::new(&x, &y, Side::Left);
        let vals: Vec<_> = [1.0, 2.0, 3.0, 4.0, 5.0]
            .iter()
            .map(|x| f.evaluate(*x))
            .collect();
        assert_eq!(vals, vec![0.0, 7.0, 10.0, 13.0, 14.0]);
    }

    #[test]
    fn test_step_function_repeated_values2() {
        let x = vec![1., 1., 2., 2., 2., 3., 3., 3., 4., 5.];
        let y = vec![6., 7., 8., 9., 10., 11., 12., 13., 14., 15.];
        let f = StepFunction::new(&x, &y, Side::Right);
        let vals: Vec<_> = [1.0, 2.0, 3.0, 4.0, 5.0]
            .iter()
            .map(|x| f.evaluate(*x))
            .collect();
        assert_eq!(vals, vec![7.0, 10.0, 13.0, 14.0, 15.0]);
    }
}
