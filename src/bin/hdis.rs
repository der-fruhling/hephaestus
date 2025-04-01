use bytes::{Buf, Bytes};
use clap::Parser;
use hephaestus::{
    BinaryEncodable, ImportedModule, Item, LinkageFlags, MetadataContentKind, Module, ModuleFlags,
    Source, StructField, Target,
};
use sha2::Digest;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::fs;
use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::PathBuf;

#[derive(clap::Parser)]
struct Cli {
    #[clap(short, long = "output")]
    output_file: Option<PathBuf>,
    input_file: PathBuf,
}

fn version_to_asm_ver(ver: u16) -> &'static str {
    match ver {
        0.. => "0.1.0",
    }
}

struct NameOrIdx<'a, const CH: char>(Option<&'a str>, usize);

impl<const CH: char> Display for NameOrIdx<'_, CH> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match &self.0 {
            Some(s) => write!(f, "{}", s),
            None => write!(f, "{CH}%{}", self.1),
        }
    }
}

fn global_name(name: Option<&str>, idx: usize) -> NameOrIdx<'_, '@'> {
    NameOrIdx(name, idx)
}

fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();

    let mut input = File::open(&cli.input_file).unwrap();
    let mut out: BufWriter<Box<dyn Write>> =
        BufWriter::new(if let Some(output_file) = &cli.output_file {
            Box::new(File::create(output_file).unwrap())
        } else {
            Box::new(std::io::stdout())
        });

    let mut bytes = Vec::new();
    input.read_to_end(&mut bytes).unwrap();
    drop(input);

    let module = Module::decode(&mut Bytes::from(bytes)).unwrap();

    writeln!(out, "%hephaestus {}", version_to_asm_ver(module.version))?;
    writeln!(out, "# assembled version 0x{:04x}", module.version)?;
    writeln!(out)?;
    writeln!(out, "[!heph::target({})]", module.target)?;

    if module.flags.contains(ModuleFlags::SUPPORTS_WIT_COMPONENTS) {
        writeln!(out, "[!heph::use_wit_components]")?;
    }

    if module.flags.contains(ModuleFlags::PARTIAL) {
        writeln!(out, "[!heph::partial]")?;
    }

    for custom_feature in &module.metadata {
        let disp: Box<dyn Display> = match custom_feature.kind {
            MetadataContentKind::String => Box::new(
                String::from_utf8(custom_feature.content.clone().into())
                    .unwrap_or_else(|e| format!("<parsing error: {e}>")),
            ),
            MetadataContentKind::Bytes => Box::new(hex::encode(&custom_feature.content)),
            #[cfg(feature = "cbor-features")]
            MetadataContentKind::CBOR => {
                use itertools::Itertools;
                use rustc_serialize::json::{Json, ToJson};

                let mut dec = cbor::Decoder::from_bytes(custom_feature.content.clone());
                let cbors: Vec<cbor::Cbor> = dec.items().try_collect()?;

                match cbors[..] {
                    [] => Box::new("<nothing>"),
                    [ref cbor] => Box::new(cbor.to_json()),
                    _ => Box::new(cbor::Cbor::Array(cbors).to_json()),
                }
            }
            MetadataContentKind::Target => {
                Box::new(Target::from_u16(custom_feature.content.clone().get_u16()))
            }
            MetadataContentKind::DWARF => Box::new("<DWARF debug information>"),
            _other => Box::new(hex::encode(&custom_feature.content)),
        };
        writeln!(
            out,
            "[!metadata({}, {}, {})]",
            custom_feature.name, custom_feature.kind, disp
        )?;
    }

    writeln!(out)?;

    for (i, global) in module.all_globals().enumerate() {
        match global {
            Item::Global(g) => {
                if g.name.is_some() {
                    writeln!(out, "# index = @%{i}")?;
                }
                pretty_print_linkage_as_attributes(&mut out, g.linkage, &g.source)?;
                write!(out, "globl ")?;
                if g.mutable {
                    write!(out, "mutable ")?;
                }

                write!(out, "{}: {}", global_name(g.name.as_deref(), i), g.ty)?;

                if let Some(init) = &g.initializer {
                    write!(out, " = {init}")?;
                }
            }
            Item::Function(f) => {
                if f.name.is_some() {
                    writeln!(out, "# index = @%{i}")?;
                }
                pretty_print_linkage_as_attributes(&mut out, f.linkage, &f.source)?;
                write!(out, "funct {}(", global_name(f.name.as_deref(), i))?;

                let mut is_first_param = true;
                for (name, ty) in &f.params {
                    if !is_first_param {
                        write!(out, ", ")?;
                    }
                    write!(out, "{name}: {ty}")?;
                    is_first_param = false;
                }

                write!(out, ") -> ")?;
                match &f.return_type {
                    Some(ty) => write!(out, "{ty}")?,
                    None => write!(out, "void")?,
                }

                if let Some(b) = &f.block {
                    writeln!(out)?;
                    write!(out, "{}", b)?;
                }
            }
            Item::ImportedModule(ImportedModule::StaticLibrary(name)) => {
                write!(out, "import static \"{name}\" as @%{i}")?;
            }
            Item::ImportedModule(ImportedModule::DynamicLibrary(name)) => {
                write!(out, "import dynamic \"{name}\" as @%{i}")?;
            }
            Item::ImportedModule(ImportedModule::WitComponent(name, content)) => {
                let mut hasher = sha2::Sha256::default();
                hasher.write_all(content)?;
                let hash = hex::encode(hasher.finalize());
                let canon_path = cli.output_file
                    .as_ref().ok_or("file contains a WIT component and an output file must be provided to save it")?
                    .canonicalize()?;
                let canon_parent = canon_path.parent().ok_or("no parent could be found")?;
                let path = canon_parent
                    .join(".components")
                    .join(hash)
                    .with_extension("wasm");
                let _ = fs::create_dir_all(path.parent().unwrap());
                fs::write(&path, content)?;
                write!(
                    out,
                    "import world \"{name}\" from component \"{}\" as @%{i}",
                    path.strip_prefix(canon_parent)?.to_string_lossy()
                )?;
            }
            Item::Struct(s) => {
                if s.name.is_some() {
                    writeln!(out, "# index = @%{i}")?;
                }
                pretty_print_linkage_as_attributes(&mut out, s.linkage, &s.source)?;
                writeln!(out, "struct {} {{", global_name(s.name.as_deref(), i))?;

                for value in &s.values {
                    match value {
                        StructField::Data {
                            layout: _, /* TODO implement layout */
                            ty,
                            name,
                        } => {
                            writeln!(out, "    {name}: {ty},")?;
                        }

                        StructField::EmptySpace(padding) => {
                            writeln!(out, "    !pad {padding},")?;
                        }
                    }
                }

                write!(out, "}}")?;
            }
            Item::Enum(e) => {
                if e.name.is_some() {
                    writeln!(out, "# index = @%{i}")?;
                }
                pretty_print_linkage_as_attributes(&mut out, e.linkage, &e.source)?;
                writeln!(
                    out,
                    "enum {}: {} {{",
                    global_name(e.name.as_deref(), i),
                    e.backing_type
                )?;

                for (name, value) in &e.values {
                    writeln!(out, "    {name} = {value},")?;
                }

                write!(out, "}}")?;
            }
        }
        write!(out, "\n\n")?;
    }

    Ok(())
}

fn pretty_print_linkage_as_attributes(
    out: &mut BufWriter<Box<dyn Write>>,
    linkage: LinkageFlags,
    source: &Option<Source>,
) -> Result<(), Box<dyn Error>> {
    if linkage.contains(LinkageFlags::EXPORTED) {
        write!(out, "export ")?;

        if linkage.contains(LinkageFlags::ACCESSIBLE) {
            write!(out, "accessible ")?;
        } else {
            write!(out, "noaccessible ")?;
        }
        if linkage.contains(LinkageFlags::DOCUMENTED) {
            write!(out, "document ")?;
        } else {
            write!(out, "nodocument ")?;
        }
    }

    if linkage.contains(LinkageFlags::INTERNAL) {
        write!(out, "internal ")?;

        if linkage.contains(LinkageFlags::FORCE_INLINE) {
            write!(out, "inline(always) ")?;
        } else if linkage.contains(LinkageFlags::INLINE) {
            write!(out, "inline ")?;
        }
    }

    if linkage.contains(LinkageFlags::IMPORTED) {
        let Some(src) = source else { unreachable!() };
        write!(out, "from @%{} / {} import ", src.module, src.item)?;
    }

    if linkage.contains(LinkageFlags::EXPECTED) {
        write!(out, "expect ")?;
    }

    Ok(())
}
