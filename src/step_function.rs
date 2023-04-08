use std::cmp::Ordering;

pub struct StepFunction<T: PartialOrd + Copy> {
    x: Vec<T>,
    y: Vec<f64>,
    side: Side,
}

impl<T: PartialOrd + Copy> StepFunction<T> {
    pub fn new(x: &[T], y: &[f64], side: Side) -> Self {
        let mut x_y_pairs: Vec<_> = x.iter().cloned().zip(y.iter().cloned()).collect();
        x_y_pairs.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
        let x: Vec<_> = x_y_pairs.iter().map(|pair| pair.0).collect();
        let y: Vec<_> = x_y_pairs.iter().map(|pair| pair.1).collect();
        Self { x, y, side }
    }

    pub fn call(&self, value: T) -> f64 {
        match self.side {
            Side::Left => {
                let index = match self
                    .x
                    .binary_search_by(|probe| probe.partial_cmp(&value).unwrap())
                {
                    Ok(i) => i,
                    Err(i) => i.saturating_sub(1),
                };
                self.y[index]
            }
            Side::Right => {
                let index = match self
                    .x
                    .binary_search_by(|probe| probe.partial_cmp(&value).unwrap())
                {
                    Ok(i) => i.saturating_sub(1),
                    Err(i) => i.saturating_sub(1),
                };
                self.y[index]
            }
        }
    }
}

pub enum Side {
    Left,
    Right,
}

pub struct ECDF<T: PartialOrd + Copy> {
    step_function: StepFunction<T>,
}

impl<T: PartialOrd + Copy> ECDF<T> {
    pub fn new(data: &[T], side: Side) -> Self {
        let mut data = data.to_vec();
        data.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        let n = data.len() as f64;
        let y: Vec<_> = (1..=data.len()).map(|i| i as f64 / n).collect();
        let step_function = StepFunction::new(&data, &y, side);
        Self { step_function }
    }

    pub fn evaluate(&self, value: T) -> f64 {
        self.step_function.call(value)
    }
}
