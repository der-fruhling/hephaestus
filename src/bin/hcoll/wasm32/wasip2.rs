#![cfg(feature = "wasm32-wasip2")]

use hephaestus::Module;

use crate::CollectorError;

pub fn collect(module: Module) -> Result<Vec<u8>, CollectorError> {
    todo!()
}
