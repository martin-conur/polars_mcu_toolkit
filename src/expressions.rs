#![allow(clippy::unused_unit)]
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

use std::collections::HashSet;

fn same_output_type(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = &input_fields[0];
    Ok(field.clone())
}

fn list_idx_dtype(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = Field::new(input_fields[0].name.clone(), DataType::List(Box::new(IDX_DTYPE)));
    Ok(field.clone())
}

fn neg_sample(inputs: &[Series], _set: &[i32]) -> PolarsResult<Series> {
    todo!();
}

#[derive(Deserialize)]
struct Index{
    val: i64
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
        let ca = s.i64().unwrap();
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
