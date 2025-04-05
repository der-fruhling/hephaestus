use clap::Parser;
use hephaestus::Module;

mod wasm32;

#[derive(clap::Parser)]
struct Cli {}

#[derive(clap::ValueEnum, Clone, Debug)]
pub enum Collector {
    #[cfg(feature = "wasm32-wasip2")]
    #[value(id = "wasm32/component!wasip2")]
    Wasm32WasiP2,
}

pub struct CollectorError {}

impl Collector {
    pub fn collect(self, module: Module) -> Result<Vec<u8>, CollectorError> {
        match self {
            #[cfg(feature = "wasm32-wasip2")]
            Self::Wasm32WasiP2 => wasm32::wasip2::collect(module),
        }
    }
}

fn main() {
    let cli = Cli::parse();
}
