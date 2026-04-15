use super::*;

pub(crate) struct Lfm2ShortConv {
    pub(super) conv_weight: Array,
    pub(super) in_proj: QuantizedLinear,
    pub(super) out_proj: QuantizedLinear,
    pub(super) hidden_size: i32,
    pub(super) conv_l_cache: i32,
}

impl Lfm2ShortConv {
    pub(super) fn forward_no_cache(&self, x: &Array) -> Result<Array> {
        let bcx = self.in_proj.forward(x)?;
        let hidden = self.hidden_size;
        let b = bcx.index((std::ops::RangeFull, std::ops::RangeFull, 0..hidden));
        let c = bcx.index((
            std::ops::RangeFull,
            std::ops::RangeFull,
            hidden..(hidden * 2),
        ));
        let x_proj = bcx.index((
            std::ops::RangeFull,
            std::ops::RangeFull,
            (hidden * 2)..(hidden * 3),
        ));
        let bx = b.multiply(&x_proj)?;
        let bx = pad(
            &bx,
            &[(0, 0), (self.conv_l_cache - 1, 0), (0, 0)],
            None::<Array>,
            None::<mlx_rs::ops::PadMode>,
        )?;
        let conv_out = conv1d(
            &bx,
            &self.conv_weight,
            None::<i32>,
            None::<i32>,
            None::<i32>,
            Some(self.hidden_size),
        )?;
        let y = c.multiply(&conv_out)?;
        self.out_proj.forward(&y)
    }
}
