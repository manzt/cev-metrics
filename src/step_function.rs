#[derive(Copy, Clone, Debug)]
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
    pub fn new(x: &Vec<f64>, y: &Vec<f64>, ival: f64, sorted: bool, side: Side) -> Self {
        assert_eq!(x.len(), y.len(), "x and y do not have the same length");

        let mut x = x.clone();
        let mut y = y.clone();

        x.insert(0, f64::NEG_INFINITY);
        y.insert(0, ival);

        if !sorted {
            let mut indices: Vec<usize> = (0..x.len()).collect();
            indices.sort_unstable_by(|a, b| x[*a].partial_cmp(&x[*b]).unwrap());

            let mut sorted_x = vec![0.; x.len()];
            let mut sorted_y = vec![0.; y.len()];

            for (i, &idx) in indices.iter().enumerate() {
                sorted_x[i] = x[idx];
                sorted_y[i] = y[idx];
            }

            x = sorted_x;
            y = sorted_y;
        }

        StepFunction { x, y, side }
    }

    pub fn evaluate(&self, x: f64) -> f64 {
        let tind = match self.x.binary_search_by(|a| a.total_cmp(&x)) {
            Ok(i) => i - 1,
            Err(i) => match self.side {
                Side::Left => i - 1,
                Side::Right => {
                    let mut j = i;
                    while self.x[i] == self.x[j] && j < self.x.len() - 1 {
                        j += 1;
                    }
                    j
                }
            },
        };
        self.y[clamp(tind, 0, self.y.len() - 1)]
    }
}

fn clamp<T: PartialOrd>(x: T, min: T, max: T) -> T {
    if x < min {
        min
    } else if x > max {
        max
    } else {
        x
    }
}

struct ECDF {
    step_function: StepFunction,
}

impl ECDF {
    /// Constructs a new `ECDF` instance.
    ///
    /// # Arguments
    ///
    /// * `x` - Observations
    /// * `side` - Side of the step intervals. Side::Left corresponds to (a, b], Side::Right corresponds to [a, b).
    pub fn new(x: &Vec<f64>, side: Side) -> Self {
        let mut sorted_x = x.clone();
        sorted_x.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let nobs = sorted_x.len() as f64;
        let y: Vec<f64> = (1..=sorted_x.len()).map(|i| i as f64 / nobs).collect();
        let step_function = StepFunction::new(&sorted_x, &y, 0.0, false, side);
        ECDF { step_function }
    }

    pub fn evaluate(&self, point: f64) -> f64 {
        self.step_function.evaluate(point)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_step_function() {
        let x = (0..20).map(|i| i as f64).collect::<Vec<_>>();
        let y = x.clone();
        let f = StepFunction::new(&x, &y, 0.0, true, Side::Left);
        let vals = [3.2, 4.5, 24., -3.1, 3.0, 4.0];
        assert_eq!(
            &vals
                .iter()
                .map(|&point| f.evaluate(point) as i32)
                .collect::<Vec<_>>(),
            &[3, 4, 19, 0, 2, 3],
        )
    }

    #[test]
    #[should_panic(expected = "x and y do not have the same length")]
    fn test_step_function_bad_shape() {
        let x = (0..20).map(|i| i as f64).collect::<Vec<_>>();
        let y = (0..21).map(|i| i as f64).collect::<Vec<_>>();
        StepFunction::new(&x, &y, 0.0, true, Side::Left);
    }

    #[test]
    fn test_step_function_value_side_right() {
        let x = (0..20).map(|i| i as f64).collect::<Vec<_>>();
        let y = x.clone();
        let f = StepFunction::new(&x, &y, 0.0, true, Side::Right);
        let vals = [3.2, 4.5, 24., -3.1, 3.0, 4.0];
        assert_eq!(
            &vals
                .iter()
                .map(|&point| f.evaluate(point) as i32)
                .collect::<Vec<_>>(),
            &[3, 4, 19, 0, 3, 4],
        )
    }

    #[test]
    fn test_step_function_repeated_values() {
        let x = vec![1., 1., 2., 2., 2., 3., 3., 3., 4., 5.];
        let y = vec![6., 7., 8., 9., 10., 11., 12., 13., 14., 15.];
        let points = vec![1., 2., 3., 4., 5.];

        let f = StepFunction::new(&x, &y, 0.0, true, Side::Left);
        assert_eq!(
            &points
                .iter()
                .map(|&point| f.evaluate(point) as i32)
                .collect::<Vec<_>>(),
            &[0, 7, 10, 13, 14],
        );

        let f2 = StepFunction::new(&x, &y, 0.0, true, Side::Right);
        assert_eq!(
            &points
                .iter()
                .map(|&point| f2.evaluate(point) as i32)
                .collect::<Vec<_>>(),
            &[7, 10, 13, 14, 15],
        );
    }
}
