use crate::{quantization::{self, Requant}, tensor::Conv_Data};
use ff_ext::ExtensionField;
use itertools::Itertools;
use tract_onnx::tract_core::ops::cnn::conv;

use crate::{Element, tensor::Tensor};

#[derive(Clone, Debug)]
pub struct Convolution {
    pub filter: Tensor<Element>,
    pub bias: Tensor<Element>,
}


impl Convolution {
    pub fn new(filter: Tensor<Element>, bias: Tensor<Element>) -> Self {
        assert_eq!(filter.kw(), bias.dims()[0]);
        Self { filter, bias }
    }
    pub fn add_bias(&self, conv_out : &Tensor<Element>) -> Tensor<Element>{
        let mut arr = conv_out.data.clone();
        assert_eq!(conv_out.data.len(),conv_out.kw()*conv_out.filter_size());
        for i in 0..conv_out.kw(){
            for j in 0..conv_out.filter_size(){
                arr[i*conv_out.filter_size() + j] += self.bias.data[i];

            }
        }
        Tensor::new(conv_out.get_shape(),arr)
    }

    pub fn op<E:ExtensionField>(&self, input: &Tensor<Element>) -> (Tensor<Element>, Conv_Data<E>) {
        let (output, proving_data) = self.filter.fft_conv(input);
        (self.add_bias(&output), proving_data)
    }

    pub fn get_shape(&self) -> Vec<usize>{
        self.filter.get_shape()
    }

    pub fn kw(&self)-> usize{
        self.filter.kw()
    }
    
    pub fn kx(&self)-> usize{
        self.filter.kx()
    }
    
    pub fn nw(&self)-> usize{
        self.filter.nw()
    }

    pub fn ncols_2d(&self)->usize{
        self.filter.ncols_2d()
    }
    
    pub fn nrows_2d(&self)->usize{
        self.filter.nrows_2d()
    }
    pub fn filter_size(&self)->usize{
        self.filter.filter_size()
    }
    pub fn requant_info(&self) -> Requant {
        let ind_range = (*quantization::MAX as i64 - *quantization::MIN as i64) as usize;
        let max_range = (2*ind_range + self.filter.ncols_2d() as usize * ind_range).next_power_of_two();
        let shift = (max_range.ilog2() as usize) - (*quantization::BIT_LEN as usize);
        Requant {
            range: max_range,
            right_shift: shift,
            after_range: 1 << *quantization::BIT_LEN,
        }
    }
}

