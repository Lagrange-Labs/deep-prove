use std::{cmp::max, marker::PhantomData};

use multilinear_extensions::virtual_poly::VPAuxInfo;

/// List of list of MLEs num_vars (f1*f2 + f1*f3*f4 + ... )
pub fn from_mle_list_dimensions<E>(product_list: &[Vec<usize>]) -> VPAuxInfo<E> {
    let mut max_num_vars = 0;
    let mut max_degree = 0;

    for product in product_list {
        max_degree = max(max_degree, product.len());
        max_num_vars = max(
            max_num_vars,
            *product
                .iter()
                .max()
                .expect("At least one MLE in the product is required"),
        );
    }

    VPAuxInfo {
        max_degree,
        max_num_variables: max_num_vars,
        phantom: PhantomData,
    }
}
