use std::{fs, path::PathBuf, process::ExitCode};

use clap::Parser;
use hephaestus::asm::parse_text_asm;

#[derive(Default, clap::ValueEnum, Clone, Eq, PartialEq)]
enum OutputMode {
    #[default] Normal,
    DebugParser,
    DebugModule
}

#[derive(clap::Parser)]
struct Cli {
    #[clap(long, default_value = "normal")]
    output_mode: OutputMode,
    
    #[clap(short, long = "output")]
    output_file: PathBuf,
    input_file: PathBuf
}

fn main() -> ExitCode {
    let cli = Cli::parse();

    let input = fs::read_to_string(&cli.input_file).unwrap();
    let asm = match parse_text_asm(&input) {
        Ok(value) => value,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE
        },
    };
    
    if cli.output_mode == OutputMode::DebugParser {
        fs::write(&cli.output_file, format!("{:#?}", asm)).unwrap();
        return ExitCode::SUCCESS
    }
    
    let module = asm.finish();
    
    if cli.output_mode == OutputMode::DebugModule {
        fs::write(&cli.output_file, format!("{:#?}", module)).unwrap();
        return ExitCode::SUCCESS
    }

    fs::write(cli.output_file, module.as_bytes()).unwrap();

    ExitCode::SUCCESS
}
