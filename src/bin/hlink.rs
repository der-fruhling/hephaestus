use std::{error::Error, fs, path::PathBuf};

use bytes::{Bytes, BytesMut};
use clap::Parser;
use hephaestus::{
    linker::{LinkError, Linker},
    BinaryEncodable, Module, Target,
};

#[derive(clap::Parser)]
struct Cli {
    #[clap(long)]
    allow_partial: bool,
    #[clap(long)]
    #[cfg(feature = "command-logging")]
    verbose: bool,

    #[clap(short = 'x', long, default_value = "ambiguous")]
    target: Target,

    #[clap(short, long)]
    output: PathBuf,
    sources: Vec<PathBuf>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();

    #[cfg(feature = "command-logging")]
    {
        if cli.verbose {
            simple_log::quick().unwrap();
            log::info!("Verbose mode enabled");
        }
    }

    let mut linker = Linker::new(cli.target);

    for src in cli.sources {
        let mut bytes = Bytes::from(fs::read(&src).unwrap());
        let module = Module::decode(&mut bytes).unwrap();
        linker.put_module(module)?;
    }

    let module = linker.link()?;

    let mut bytes = BytesMut::new();
    module.encode(&mut bytes);
    fs::write(&cli.output, bytes).unwrap();
    Ok(())
}
