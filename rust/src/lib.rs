use anyhow::{Context, Result};
use ndarray::Array2;
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyModule;

#[path = "deep_filter/lib.rs"]
mod deep_filter;

use crate::deep_filter::tract::{DfParams, DfTract, ReduceMask, RuntimeParams};

fn to_py_err(err: anyhow::Error) -> PyErr {
    let details = err
        .chain()
        .map(|cause| cause.to_string())
        .collect::<Vec<_>>()
        .join(": ");
    PyRuntimeError::new_err(details)
}

#[pyclass(unsendable)]
struct DeepFilterNetRealtime {
    model: DfTract,
    pending: Vec<f32>,
    samples_to_drop: usize,
    closed: bool,
}

#[pymethods]
impl DeepFilterNetRealtime {
    #[new]
    #[pyo3(signature = (
        model_path=None,
        atten_lim=100.0,
        log_level=None,
        compensate_delay=true,
        post_filter_beta=0.0,
        min_db_thresh=-15.0,
        max_db_erb_thresh=35.0,
        max_db_df_thresh=35.0
    ))]
    fn new(
        model_path: Option<String>,
        atten_lim: f32,
        log_level: Option<String>,
        compensate_delay: bool,
        post_filter_beta: f32,
        min_db_thresh: f32,
        max_db_erb_thresh: f32,
        max_db_df_thresh: f32,
    ) -> PyResult<Self> {
        let _ = log_level;
        let model = create_model(
            model_path,
            atten_lim,
            post_filter_beta,
            min_db_thresh,
            max_db_erb_thresh,
            max_db_df_thresh,
        )
        .map_err(to_py_err)?;
        let samples_to_drop = if compensate_delay {
            model.fft_size - model.hop_size + model.lookahead * model.hop_size
        } else {
            0
        };
        Ok(Self {
            model,
            pending: Vec::new(),
            samples_to_drop,
            closed: false,
        })
    }

    #[getter]
    fn sample_rate(&self) -> usize {
        self.model.sr
    }

    #[getter]
    fn frame_length(&self) -> usize {
        self.model.hop_size
    }

    fn process_chunk<'py>(
        &mut self,
        py: Python<'py>,
        audio: PyReadonlyArray1<'py, f32>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        if self.closed {
            return Err(PyRuntimeError::new_err(
                "DeepFilterNetRealtime processor is closed",
            ));
        }
        let input = audio.as_slice()?;
        self.pending.extend_from_slice(input);
        let output = self.process_available_frames().map_err(to_py_err)?;
        Ok(PyArray1::from_vec_bound(py, output))
    }

    fn finalize<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f32>>> {
        if self.closed {
            return Ok(PyArray1::from_vec_bound(py, Vec::new()));
        }
        let output = if self.pending.is_empty() {
            Vec::new()
        } else {
            let hop_size = self.model.hop_size;
            self.pending.resize(hop_size, 0.0);
            self.process_available_frames().map_err(to_py_err)?
        };
        self.pending.clear();
        self.closed = true;
        Ok(PyArray1::from_vec_bound(py, output))
    }

    fn close(&mut self) {
        self.pending.clear();
        self.closed = true;
    }
}

impl DeepFilterNetRealtime {
    fn process_available_frames(&mut self) -> Result<Vec<f32>> {
        let hop_size = self.model.hop_size;
        let mut output = Vec::new();
        while self.pending.len() >= hop_size {
            let frame: Vec<f32> = self.pending.drain(..hop_size).collect();
            let noisy = Array2::from_shape_vec((1, hop_size), frame)
                .context("Could not create DeepFilterNet input frame")?;
            let mut enhanced = Array2::<f32>::zeros((1, hop_size));
            self.model
                .process(noisy.view(), enhanced.view_mut())
                .context("DeepFilterNet frame processing failed")?;
            let mut frame_output = enhanced.into_raw_vec();
            if self.samples_to_drop > 0 {
                let drop = self.samples_to_drop.min(frame_output.len());
                frame_output.drain(..drop);
                self.samples_to_drop -= drop;
            }
            output.extend(frame_output);
        }
        Ok(output)
    }
}

fn create_model(
    model_path: Option<String>,
    atten_lim: f32,
    post_filter_beta: f32,
    min_db_thresh: f32,
    max_db_erb_thresh: f32,
    max_db_df_thresh: f32,
) -> Result<DfTract> {
    if post_filter_beta < 0.0 {
        return Err(anyhow::anyhow!("post_filter_beta must be >= 0"));
    }
    let r_params = RuntimeParams::default_with_ch(1)
        .with_atten_lim(atten_lim)
        .with_thresholds(min_db_thresh, max_db_erb_thresh, max_db_df_thresh)
        .with_post_filter(post_filter_beta)
        .with_mask_reduce(ReduceMask::NONE);
    let df_params = match model_path {
        Some(path) => DfParams::new(path.into()).context("Could not load DeepFilterNet model")?,
        None => DfParams::default(),
    };
    DfTract::new(df_params, &r_params).context("Could not initialize DeepFilterNet runtime")
}

#[pymodule]
fn _native(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DeepFilterNetRealtime>()?;
    Ok(())
}
