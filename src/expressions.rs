#![allow(clippy::unused_unit)]
#![allow(unused_imports)]

use std::hash::Hash;
use std::ops::{Add, Div, Mul, Sub};

use polars::export::num::{NumCast, Zero};
use polars::prelude::arity::{
    binary_elementwise_into_string_amortized, broadcast_binary_elementwise,
};
use polars::prelude::*;
use polars_arrow::bitmap::MutableBitmap;
use polars_core::series::amortized_iter::AmortSeries;
use polars_core::utils::align_chunks_binary;
use pyo3_polars::derive::polars_expr;
use pyo3_polars::export::polars_core::export::num::Signed;
use pyo3_polars::export::polars_core::utils::arrow::array::PrimitiveArray;
use pyo3_polars::export::polars_core::utils::CustomIterTools;
use serde::Deserialize;
use rand::seq::SliceRandom;
use rand::thread_rng;

use std::collections::HashSet;

// fn same_output_type(input_fields: &[Field]) -> PolarsResult<Field> {
//     let field = &input_fields[0];
//     Ok(field.clone())
// }

fn list_idx_dtype(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = Field::new(input_fields[0].name.clone(), DataType::List(Box::new(IDX_DTYPE)));
    Ok(field.clone())
}
#[derive(Deserialize)]
struct Index{
    val: i64
}

#[derive(Deserialize)]
struct Maestra {
    sample_from: Vec<i64>,
    neg_ratio: usize
}

#[polars_expr(output_type_func=list_idx_dtype)]
fn non_val_indices(inputs: &[Series], kwargs: Index) -> PolarsResult<Series> {
    let ca = inputs[0].list()?;
    polars_ensure!(
        ca.dtype() == &DataType::List(Box::new(DataType::Int64)),
        ComputeError: "Expexted 'List(Int64)' got: {}", ca.dtype()
    );
    
    let out: ListChunked = ca.apply_amortized(|s|{
        let s: &Series = s.as_ref();
        let casted_series = s.cast(&DataType::Int64).unwrap();
        let ca = casted_series.i64().unwrap();
        let out: IdxCa = ca
            .iter()
            .enumerate()
            .filter(|(_idx, opt_val)| opt_val != &Some(kwargs.val))
            .map(|(idx, _opt_val)| Some(idx as IdxSize))
            .collect_ca(PlSmallStr::EMPTY);
        out.into_series()
    });
    Ok(out.into_series())
}



#[polars_expr(output_type_func=list_idx_dtype)]
fn neg_sample(inputs: &[Series], kwargs: Maestra) -> PolarsResult<Series> {
    let ca = inputs[0].list()?;
    polars_ensure!(
        ca.dtype() == &DataType::List(Box::new(DataType::Int64)) || ca.dtype() == &DataType::List(Box::new(DataType::Int32)),
        ComputeError: "Expexted 'List(Int64)' got: {}", ca.dtype()
    );

    let out: ListChunked = ca.apply_amortized(|s|{
        let s: &Series = s.as_ref();
        let casted_series = s.cast(&DataType::Int64).expect("Expected List(i32) or List(I64)");

        let ca = casted_series.i64().unwrap();
        
        let mut rng = thread_rng();

        let subsampled_set: Vec<i64> = kwargs.sample_from
            .choose_multiple(&mut rng, ca.len() * 2 * kwargs.neg_ratio)
            .cloned()
            .collect();
        let ca_set: HashSet<_> = ca.to_vec().into_iter().collect();

        let out: Vec<_> = subsampled_set
            .into_iter()
            .filter(|x| !ca_set.contains(&Some(*x)))
            .take(ca.len() * kwargs.neg_ratio) // we only need the len of the original list
            .collect();

        Series::from_vec("neg_sample".into(), out)
    });
    Ok(out.into_series())
}
 

