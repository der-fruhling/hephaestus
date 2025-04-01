pub mod asm;
pub mod instruction;
pub mod linker;

use bitflags::bitflags;
use bytes::{Buf, BufMut, Bytes, BytesMut};
use instruction::{Instruction, Op};
use itertools::Itertools;
use lazy_static::lazy_static;
use ordered_hash_map::OrderedHashMap;
use std::collections::BTreeMap;
use std::iter;
use std::str::FromStr;
use std::string::FromUtf8Error;
use std::{
    collections::HashMap,
    fmt::{Debug, Display, Formatter},
    mem,
};
use thiserror::Error;

#[derive(Default, Debug, Clone)]
pub struct Module<Global = u32> {
    pub version: u16,
    pub flags: ModuleFlags,
    pub target: Target,
    pub metadata: Vec<MetadataDeclaration>,
    globals: Vec<Item<Global>>,
    imported_modules: HashMap<String, usize>,
    functions: HashMap<String, usize>,
    global_vars: HashMap<String, usize>,
    structs: HashMap<String, usize>,
    enums: HashMap<String, usize>,
}

impl<Global> Module<Global> {
    pub fn new() -> Self
    where
        Global: Default,
    {
        Self {
            version: CURRENT_VERSION,
            ..Default::default()
        }
    }

    pub fn add_item(&mut self, item: Item<Global>) -> u32 {
        let idx = self.globals.len() as u32;
        self.globals.push(item);
        idx
    }

    pub fn map_globals<T: Default>(self, f: impl Fn(Global) -> T + Copy) -> Module<T> {
        let Module {
            version,
            flags,
            target,
            metadata,
            globals,
            ..
        } = self;

        let mut m = Module::<T> {
            version,
            flags,
            target,
            metadata,
            globals: globals.into_iter().map(|i| i.map_globals::<T>(f)).collect(),
            ..Default::default()
        };

        m.recalculate();

        m
    }

    pub fn globals(&self) -> impl IntoIterator<Item = &Global> {
        self.globals.iter().flat_map(|v| v.needed_globals())
    }
}

#[derive(Debug, Clone)]
pub struct MetadataDeclaration {
    pub name: String,
    pub kind: MetadataContentKind,
    pub content: Bytes,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
#[repr(u8)]
#[non_exhaustive]
pub enum MetadataContentKind {
    /// UTF-8 encoded string. Has predefined meaning.
    String = 0x01,

    /// Raw binary. Has predefined meaning.
    Bytes = 0x02,

    /// Arbitrary CBOR encoded table. Has predefined meaning.
    #[cfg(feature = "cbor-features")]
    CBOR = 0x10,

    /// Describes a [Target]. Its usage is defined by surrounding information.
    Target = 0x11,

    /// Debug information encoded in the DWARF format.
    DWARF = 0x80,
}

impl Display for MetadataContentKind {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            MetadataContentKind::String => f.write_str("string"),
            MetadataContentKind::Bytes => f.write_str("bytes"),
            #[cfg(feature = "cbor-features")]
            MetadataContentKind::CBOR => f.write_str("table"),
            MetadataContentKind::Target => f.write_str("target"),
            MetadataContentKind::DWARF => f.write_str("dwarf"),
        }
    }
}

impl MetadataContentKind {
    pub fn into_u8(self) -> u8 {
        self as u8
    }

    pub fn from_u8(value: u8) -> Self {
        unsafe { mem::transmute(value) }
    }
}

impl<Global> Module<Global> {
    pub(crate) fn recalculate(&mut self) {
        self.functions.clear();
        self.global_vars.clear();

        for (i, item) in self.globals.iter().enumerate() {
            match item {
                Item::Function(f) => {
                    if let Some(name) = &f.name {
                        self.functions.insert(name.clone(), i);
                    }
                }
                Item::Global(f) => {
                    if let Some(name) = &f.name {
                        self.global_vars.insert(name.clone(), i);
                    }
                }
                Item::ImportedModule(m) => {
                    let n = match m {
                        ImportedModule::DynamicLibrary(n) => n,
                        ImportedModule::StaticLibrary(n) => n,
                        ImportedModule::WitComponent(n, _) => n,
                    };

                    self.imported_modules.insert(n.clone(), i);
                }
                Item::Struct(s) => {
                    if let Some(name) = &s.name {
                        self.structs.insert(name.clone(), i);
                    }
                }
                Item::Enum(e) => {
                    if let Some(name) = &e.name {
                        self.enums.insert(name.clone(), i);
                    }
                }
            };
        }
    }

    pub fn all_globals(&self) -> impl Iterator<Item = &Item<Global>> {
        self.globals.iter()
    }
}

impl Module {
    pub fn as_bytes(&self) -> Bytes {
        let mut bytes = BytesMut::new();
        self.encode(&mut bytes);
        bytes.freeze()
    }

    pub fn get_function(&self, name: &str) -> Option<&Function> {
        let index = *self.functions.get(name)?;
        self.globals.get(index).and_then(|v| match v {
            Item::Function(f) => Some(f),
            _other => None,
        })
    }

    pub fn get_global(&self, name: &str) -> Option<&Global> {
        let index = *self.global_vars.get(name)?;
        self.globals.get(index).and_then(|v| match v {
            Item::Global(g) => Some(g),
            _other => None,
        })
    }
}

#[derive(Debug, Clone)]
pub enum ImportedModule {
    /// Refers to a native library. Actual resolved path is system dependent.
    ///
    /// The found library is linked dynamically.
    ///
    /// Only supported on native platforms.
    DynamicLibrary(String),

    /// Refers to a native library. Actual resolved path is system dependent.
    ///
    /// The found library is linked statically.
    ///
    /// Only supported on native platforms and WebAssembly.
    StaticLibrary(String),

    /// Assumes a WASM component is implemented by the host. The provided
    /// bytes are assumed to be the component and will be included in the
    /// resulting component as an import.
    ///
    /// Only binary WASM components are supported. Textual WIT components will
    /// result in an error.
    ///
    /// Only supported on WebAssembly when targeting the component model.
    WitComponent(String, Bytes),
}

impl ImportedModule {
    const DYNAMIC_LIBRARY: u8 = 1;
    const STATIC_LIBRARY: u8 = 2;
    const WIT_COMPONENT: u8 = 3;
}

impl BinaryEncodable for ImportedModule {
    fn encode(&self, bytes: &mut BytesMut) {
        match self {
            ImportedModule::DynamicLibrary(name) => {
                bytes.put_u8(Self::DYNAMIC_LIBRARY);
                bytes.put_u16(name.len() as u16);
                bytes.put_slice(name.as_bytes());
            }
            ImportedModule::StaticLibrary(name) => {
                bytes.put_u8(Self::STATIC_LIBRARY);
                bytes.put_u16(name.len() as u16);
                bytes.put_slice(name.as_bytes());
            }
            ImportedModule::WitComponent(name, content) => {
                bytes.put_u8(Self::WIT_COMPONENT);
                bytes.put_u16(name.len() as u16);
                bytes.put_slice(name.as_bytes());
                bytes.put_u32(content.len() as u32);
                bytes.put(content.clone());
            }
        }
    }

    fn decode(bytes: &mut Bytes) -> Result<Self, DecodeError>
    where
        Self: Sized,
    {
        match bytes.get_u8() {
            Self::DYNAMIC_LIBRARY => {
                let name_len = bytes.get_u16() as usize;
                let name = String::from_utf8_lossy(&bytes.split_to(name_len)).into_owned();

                Ok(ImportedModule::DynamicLibrary(name))
            }
            Self::STATIC_LIBRARY => {
                let name_len = bytes.get_u16() as usize;
                let name = String::from_utf8_lossy(&bytes.split_to(name_len)).into_owned();

                Ok(ImportedModule::StaticLibrary(name))
            }
            Self::WIT_COMPONENT => {
                let name_len = bytes.get_u16() as usize;
                let name = String::from_utf8_lossy(&bytes.split_to(name_len)).into_owned();
                let content_len = bytes.get_u32() as usize;
                let content = bytes.split_to(content_len);

                Ok(ImportedModule::WitComponent(name, content))
            }
            _ => Err(DecodeError::InvalidImportedModuleType),
        }
    }
}

#[repr(u16)]
#[derive(Default, Debug, Eq, PartialEq, Ord, PartialOrd, Copy, Clone)]
pub enum Target {
    /// Indicates that the target is not known.
    #[default]
    AmbiguousTarget = 0,

    NativeLinuxX86_32 = 0x0100,
    NativeWindowsX86_32 = 0x0101,
    NativeDarwinX86_32 = 0x0102,

    NativeLinuxX86_64 = 0x0110,
    NativeWindowsX86_64 = 0x0111,
    NativeDarwinX86_64 = 0x0112,

    NativeLinuxArm32 = 0x0400,
    NativeLinuxArm64 = 0x0410,
    NativeDarwinArm64 = 0x0411,
    NativeWindowsArm64 = 0x0412,

    WebAssembly32Module = 0xD000,
    WebAssembly32PartialModule = 0xD001,
    WebAssembly32Component = 0xD010,
    WebAssembly32PartialComponent = 0xD011,
}

impl Target {
    #[allow(unreachable_code)]
    const fn get_self() -> Target {
        #[cfg(all(target_os = "linux", target_arch = "x86"))]
        return Target::NativeLinuxX86_32;
        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
        return Target::NativeLinuxX86_64;
        #[cfg(all(target_os = "linux", target_arch = "arm"))]
        return Target::NativeLinuxArm32;
        #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
        return Target::NativeLinuxArm64;
        #[cfg(all(target_os = "windows", target_arch = "x86"))]
        return Target::NativeWindowsX86_32;
        #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
        return Target::NativeWindowsX86_64;
        #[cfg(all(target_os = "windows", target_arch = "aarch64"))]
        return Target::NativeWindowsArm64;
        #[cfg(all(target_os = "macos", target_arch = "x86"))]
        return Target::NativeDarwinX86_32;
        #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
        return Target::NativeDarwinX86_64;
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        return Target::NativeDarwinArm64;
        #[cfg(all(target_arch = "wasm32", target_os = "wasi", target_env = "p1"))]
        return Target::WebAssembly32Module;
        #[cfg(all(target_arch = "wasm32", target_os = "wasi", target_env = "p2"))]
        return Target::WebAssembly32Component;
        Target::AmbiguousTarget
    }

    pub const SELF: Self = Self::get_self();
}

lazy_static! {
    static ref TARGETS: HashMap<&'static str, Target> = HashMap::from([
        ("ambiguous", Target::AmbiguousTarget),
        ("native/linux/x86", Target::NativeLinuxX86_32),
        ("native/windows/x86", Target::NativeWindowsX86_32),
        ("native/darwin/x86", Target::NativeDarwinX86_32),
        ("native/linux/x86_64", Target::NativeLinuxX86_64),
        ("native/windows/x86_64", Target::NativeWindowsX86_64),
        ("native/darwin/x86_64", Target::NativeDarwinX86_64),
        ("native/linux/arm32", Target::NativeLinuxArm32),
        ("native/linux/arm64", Target::NativeLinuxArm64),
        ("native/darwin/arm64", Target::NativeDarwinArm64),
        ("native/windows/arm64", Target::NativeWindowsArm64),
        ("wasm32/module", Target::WebAssembly32Module),
        ("wasm32/module_part", Target::WebAssembly32PartialModule),
        ("wasm32/component", Target::WebAssembly32Component),
        (
            "wasm32/component_part",
            Target::WebAssembly32PartialComponent
        ),
    ]);
    static ref TARGET_NAMES: BTreeMap<Target, &'static str> = (*TARGETS)
        .clone()
        .into_iter()
        .map(|(a, b)| (b, a))
        .collect::<BTreeMap<_, _>>();
}

#[derive(Error, Debug)]
#[error("invalid target name: {0}")]
pub struct InvalidTargetError(String);

impl FromStr for Target {
    type Err = InvalidTargetError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        TARGETS
            .get(s)
            .copied()
            .ok_or_else(|| InvalidTargetError(s.to_string()))
    }
}

impl Display for Target {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(TARGET_NAMES.get(self).unwrap())
    }
}

impl Target {
    pub fn into_u16(self) -> u16 {
        self as u16
    }

    pub fn from_u16(val: u16) -> Self {
        unsafe { std::mem::transmute(val) }
    }
}

bitflags! {
    #[derive(Debug, Clone, Copy, Default)]
    pub struct ModuleFlags: u32 {
        /// Declares that this module contains [Metadata] entries. Only used
        /// for encoding and decoding, this flag is automatically set.
        const HAS_CUSTOM_FEATURES = 1 << 0;

        /// Declares that this module contains a WIT component that must be
        /// loaded for the module to be coherent. This is present to allow
        /// tools to initialize any WebAssembly decoders and to provide an
        /// easy out to declare the module invalid if the tool does not
        /// support these components.
        const SUPPORTS_WIT_COMPONENTS = 1 << 1;

        /// Declares that the module contains items with
        /// [LinkageFlags::EXPECTED] linkage. A module with this flag will
        /// fail to transpile to actual target code unless that target
        /// supports partial modules (e.g. it's being compiled to an object
        /// file, or the partial-supporting WebAssembly targets are being used)
        const PARTIAL = 1 << 2;
    }
}

#[derive(Error, Debug)]
pub enum DecodeError {
    #[error("module missing magic number")]
    MissingMagic,
    #[error("version {0} of file is greater than the latest supported version {CURRENT_VERSION}")]
    UnsupportedVersion(u16),
    #[error("illegal imported module type")]
    InvalidImportedModuleType,
    #[error("invalid flags")]
    InvalidFlags,
    #[error("invalid variant")]
    InvalidVariant,
    #[error("invalid item type: {0}")]
    InvalidItemType(u8),
    #[error("invalid primitive type: 0x{0:02x}")]
    InvalidPrimitiveType(u8),
    #[error("attempted to decode invalid UTF8")]
    InvalidUTF8(#[from] FromUtf8Error),

    #[error("erroneous block end op: {0:?}")]
    BlockEnd(Op),
}

pub trait BinaryEncodable {
    fn encode(&self, bytes: &mut BytesMut);
    fn decode(bytes: &mut Bytes) -> Result<Self, DecodeError>
    where
        Self: Sized;
}

const MODULE_MAGIC: u32 = 0xCAFEEE77;
const CURRENT_VERSION: u16 = 0;

type ModuleDecoderFn = fn(&mut Bytes) -> Result<Module, DecodeError>;

static MODULE_DECODERS: &[ModuleDecoderFn] = &[|bytes| -> Result<Module, DecodeError> {
    let flags = ModuleFlags::from_bits_retain(bytes.get_u32());
    let target = Target::from_u16(bytes.get_u16());

    let custom_features = if flags.contains(ModuleFlags::HAS_CUSTOM_FEATURES) {
        let mut custom_features = vec![];
        let feature_len = bytes.get_u16() as usize;

        for _ in 0..feature_len {
            let name_len = bytes.get_u16() as usize;
            let name = String::from_utf8_lossy(&bytes.split_to(name_len)).into_owned();
            let kind = MetadataContentKind::from_u8(bytes.get_u8());
            let content_len = bytes.get_u16() as usize;
            let content = bytes.split_to(content_len);
            custom_features.push(MetadataDeclaration {
                name,
                kind,
                content,
            })
        }

        custom_features
    } else {
        vec![]
    };

    let global_len = bytes.get_u32() as usize;
    let mut globals = vec![];

    for _ in 0..global_len {
        globals.push(Item::decode(bytes)?);
    }

    let mut module = Module {
        version: 0,
        flags,
        target,
        metadata: custom_features,
        globals,
        imported_modules: Default::default(),
        functions: Default::default(),
        global_vars: Default::default(),
        structs: Default::default(),
        enums: Default::default(),
    };

    module.recalculate();

    Ok(module)
}];

impl BinaryEncodable for Module {
    fn encode(&self, bytes: &mut BytesMut) {
        bytes.put_u32(MODULE_MAGIC);
        bytes.put_u16(CURRENT_VERSION);

        // everything after this comment can be changed if a decoder is written
        // for the new version.
        let mut flags = self.flags;
        flags.set(ModuleFlags::HAS_CUSTOM_FEATURES, !self.metadata.is_empty());
        bytes.put_u32(flags.bits());

        bytes.put_u16(self.target.into_u16());

        if !self.metadata.is_empty() {
            bytes.put_u16(self.metadata.len() as u16);

            for f in &self.metadata {
                bytes.put_u16(f.name.len() as u16);
                bytes.put_slice(f.name.as_bytes());
                bytes.put_u8(f.kind.into_u8());
                bytes.put_u16(f.content.len() as u16);
                bytes.put(f.content.clone());
            }
        }

        bytes.put_u32(self.globals.len() as u32);

        for g in &self.globals {
            g.encode(bytes);
        }
    }

    fn decode(bytes: &mut Bytes) -> Result<Self, DecodeError>
    where
        Self: Sized,
    {
        let magic = bytes.get_u32();
        if magic != MODULE_MAGIC {
            return Err(DecodeError::MissingMagic);
        }

        let version = bytes.get_u16();
        if version > CURRENT_VERSION {
            return Err(DecodeError::UnsupportedVersion(version));
        }

        MODULE_DECODERS[version as usize](bytes)
    }
}

bitflags! {
    #[derive(Debug, Clone, Copy)]
    pub struct LinkageFlags: u16 {
        /// Whether this item should be importable for use.
        const EXPORTED = 1 << 0;
        /// If this flag is provided, this item has no implementation in this
        /// module and must be imported from outside. If provided, the name of
        /// the module precedes the item's name separated by a colon.
        ///
        /// E.g. `@global_var` becomes `@module:global_var`
        const IMPORTED = 1 << 1;
        /// Whether this item can be accessed from other modules.
        const ACCESSIBLE = 1 << 2;
        /// Whether this item was marked as internal. Tools may use this flag
        /// to hide or warn against using internal items.
        const INTERNAL = 1 << 3;
        /// Whether this item should be documented. If this flag is not
        /// provided, tools intending to display items contained in this module
        /// should skip this item.
        const DOCUMENTED = 1 << 4;
        /// If this flag is provided, this item has a block associated with it.
        /// It will be read by the decoder.
        /// Only global variables and functions may have this flag.
        const HAS_BLOCK = 1 << 5;
        /// If this flag is provided, this item is mutable to anything that
        /// can access it.
        /// Only global variables may have this flag.
        const IS_MUTABLE = 1 << 6;

        const FORCE_INLINE = 1 << 7;
        const INLINE = 1 << 8;

        /// Provides a hard dependency for this item, expecting that a linker
        /// will provide an implementation. Using this option requires that
        /// [ModuleFlags::PARTIAL] is also set.
        const EXPECTED = 1 << 9;
    }
}

impl LinkageFlags {
    pub fn should_be_named(&self) -> bool {
        self.contains(Self::EXPORTED)
            | self.contains(Self::IMPORTED)
            | self.contains(Self::EXPECTED)
    }

    pub fn should_cause_inclusion(&self) -> bool {
        self.contains(Self::EXPORTED) | self.contains(Self::IMPORTED)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u8)]
#[non_exhaustive]
pub enum Type {
    U8 = 1,
    U16 = 2,
    U32 = 3,
    U64 = 4,
    I8 = 5,
    I16 = 6,
    I32 = 7,
    I64 = 8,
    F32 = 9,
    F64 = 10,
    UPtr = 11,
    IPtr = 12,
}

impl Display for Type {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Type::U8 => f.write_str("u8"),
            Type::U16 => f.write_str("u16"),
            Type::U32 => f.write_str("u32"),
            Type::U64 => f.write_str("u64"),
            Type::I8 => f.write_str("i8"),
            Type::I16 => f.write_str("i16"),
            Type::I32 => f.write_str("i32"),
            Type::I64 => f.write_str("i64"),
            Type::F32 => f.write_str("f32"),
            Type::F64 => f.write_str("f64"),
            Type::UPtr => f.write_str("u_ptr"),
            Type::IPtr => f.write_str("i_ptr"),
        }
    }
}

#[derive(Copy, Clone, Debug)]
#[non_exhaustive]
pub enum Const {
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
}

impl BinaryEncodable for Const {
    fn encode(&self, bytes: &mut BytesMut) {
        bytes.put_u8(self.ty().into_u8());
        self.encode_value(bytes);
    }

    fn decode(bytes: &mut Bytes) -> Result<Self, DecodeError>
    where
        Self: Sized,
    {
        let ty = Type::from_u8(bytes.get_u8());
        Self::decode_value(bytes, ty)
    }
}

impl Const {
    pub fn ty(&self) -> Type {
        match self {
            Const::U8(_) => Type::U8,
            Const::U16(_) => Type::U16,
            Const::U32(_) => Type::U32,
            Const::U64(_) => Type::U64,
            Const::I8(_) => Type::I8,
            Const::I16(_) => Type::I16,
            Const::I32(_) => Type::I32,
            Const::I64(_) => Type::I64,
            Const::F32(_) => Type::F32,
            Const::F64(_) => Type::F64,
        }
    }

    pub fn decode_value(bytes: &mut Bytes, ty: Type) -> Result<Const, DecodeError> {
        match ty {
            Type::U8 => Ok(Const::U8(bytes.get_u8())),
            Type::U16 => Ok(Const::U16(bytes.get_u16())),
            Type::U32 => Ok(Const::U32(bytes.get_u32())),
            Type::U64 => Ok(Const::U64(bytes.get_u64())),
            Type::I8 => Ok(Const::I8(bytes.get_i8())),
            Type::I16 => Ok(Const::I16(bytes.get_i16())),
            Type::I32 => Ok(Const::I32(bytes.get_i32())),
            Type::I64 => Ok(Const::I64(bytes.get_i64())),
            Type::F32 => Ok(Const::F32(bytes.get_f32())),
            Type::F64 => Ok(Const::F64(bytes.get_f64())),
            other => Err(DecodeError::InvalidPrimitiveType(other.into_u8())),
        }
    }

    pub fn encode_value(&self, bytes: &mut BytesMut) {
        match self {
            Const::U8(v) => bytes.put_u8(*v),
            Const::U16(v) => bytes.put_u16(*v),
            Const::U32(v) => bytes.put_u32(*v),
            Const::U64(v) => bytes.put_u64(*v),
            Const::I8(v) => bytes.put_i8(*v),
            Const::I16(v) => bytes.put_i16(*v),
            Const::I32(v) => bytes.put_i32(*v),
            Const::I64(v) => bytes.put_i64(*v),
            Const::F32(v) => bytes.put_f32(*v),
            Const::F64(v) => bytes.put_f64(*v),
        }
    }
}

impl Display for Const {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Const::U8(val) => Display::fmt(val, f),
            Const::U16(val) => Display::fmt(val, f),
            Const::U32(val) => Display::fmt(val, f),
            Const::U64(val) => Display::fmt(val, f),
            Const::I8(val) => Display::fmt(val, f),
            Const::I16(val) => Display::fmt(val, f),
            Const::I32(val) => Display::fmt(val, f),
            Const::I64(val) => Display::fmt(val, f),
            Const::F32(val) => Display::fmt(val, f),
            Const::F64(val) => Display::fmt(val, f),
        }
    }
}

impl Type {
    pub fn into_u8(self) -> u8 {
        self as u8
    }

    pub fn from_u8(val: u8) -> Self {
        let val = val & 0x3F;
        if val == 0x3F {
            return Self::UPtr;
        }

        unsafe { mem::transmute(val) }
    }
}

#[derive(Debug, Clone)]
pub enum Item<GlobalTy = u32> {
    ImportedModule(ImportedModule),
    Global(Global<GlobalTy>),
    Function(Function<GlobalTy>),
    Struct(Struct<GlobalTy>),
    Enum(Enum<GlobalTy>),
}

impl<Global> Item<Global> {
    const GLOBAL: u8 = 1;
    const FUNCTION: u8 = 2;
    const IMPORTED_MODULE: u8 = 3;
    const STRUCT: u8 = 4;
    const ENUM: u8 = 5;

    pub fn name(&self) -> Option<&str> {
        match self {
            Item::ImportedModule(_) => None,
            Item::Global(global) => global.name.as_deref(),
            Item::Function(function) => function.name.as_deref(),
            Item::Struct(struct_ty) => struct_ty.name.as_deref(),
            Item::Enum(enum_ty) => enum_ty.name.as_deref(),
        }
    }

    pub fn linkage(&self) -> Option<LinkageFlags> {
        match self {
            Item::ImportedModule(_) => None,
            Item::Global(global) => Some(global.linkage),
            Item::Function(function) => Some(function.linkage),
            Item::Struct(struct_ty) => Some(struct_ty.linkage),
            Item::Enum(enum_ty) => Some(enum_ty.linkage),
        }
    }

    pub fn map_globals<T>(self, f: impl Fn(Global) -> T + Copy) -> Item<T> {
        match self {
            Item::ImportedModule(imported_module) => Item::ImportedModule(imported_module),
            Item::Global(global) => Item::Global(global.map_globals(f)),
            Item::Function(function) => Item::Function(function.map_globals(f)),
            Item::Struct(s) => Item::Struct(s.map_globals(f)),
            Item::Enum(e) => Item::Enum(e.map_globals(f)),
        }
    }

    pub fn needed_globals(&self) -> Vec<&Global> {
        match self {
            Item::ImportedModule(_) => vec![],
            Item::Global(global) => global.needed_globals().into_iter().collect(),
            Item::Function(function) => function.needed_globals().into_iter().collect(),
            Item::Struct(s) => s.needed_globals().into_iter().collect(),
            Item::Enum(e) => e.needed_globals().into_iter().collect(),
        }
    }
}

impl BinaryEncodable for Item {
    fn encode(&self, bytes: &mut BytesMut) {
        match self {
            Item::Global(g) => {
                bytes.put_u8(Self::GLOBAL);
                g.encode(bytes);
            }
            Item::Function(f) => {
                bytes.put_u8(Self::FUNCTION);
                f.encode(bytes);
            }
            Item::ImportedModule(m) => {
                bytes.put_u8(Self::IMPORTED_MODULE);
                m.encode(bytes);
            }
            Item::Struct(s) => {
                bytes.put_u8(Self::STRUCT);
                s.encode(bytes);
            }
            Item::Enum(e) => {
                bytes.put_u8(Self::ENUM);
                e.encode(bytes);
            }
        }
    }

    fn decode(bytes: &mut Bytes) -> Result<Self, DecodeError>
    where
        Self: Sized,
    {
        match bytes.get_u8() {
            Self::GLOBAL => Ok(Item::Global(Global::decode(bytes)?)),
            Self::FUNCTION => Ok(Item::Function(Function::decode(bytes)?)),
            Self::IMPORTED_MODULE => Ok(Item::ImportedModule(ImportedModule::decode(bytes)?)),
            Self::STRUCT => Ok(Item::Struct(Struct::decode(bytes)?)),
            Self::ENUM => Ok(Item::Enum(Enum::decode(bytes)?)),
            other => Err(DecodeError::InvalidItemType(other)),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Source<GlobalTy = u32> {
    pub module: GlobalTy,
    pub item: String,
}

#[derive(Debug, Clone)]
pub struct Global<GlobalTy = u32> {
    pub name: Option<String>,
    pub source: Option<Source<GlobalTy>>,
    pub linkage: LinkageFlags,
    pub ty: ComplexType<GlobalTy>,
    pub mutable: bool,
    pub initializer: Option<Block<GlobalTy>>,
}

impl BinaryEncodable for Global {
    fn encode(&self, bytes: &mut BytesMut) {
        let mut linkage = self.linkage;
        linkage.set(LinkageFlags::IS_MUTABLE, self.mutable);
        linkage.set(LinkageFlags::HAS_BLOCK, self.initializer.is_some());
        linkage.set(LinkageFlags::IMPORTED, self.source.is_some());

        bytes.put_u16(linkage.bits());
        self.ty.encode(bytes);

        if let Some(source) = &self.source {
            bytes.put_u32(source.module);
            bytes.put_u16(source.item.len() as u16);
            bytes.put_slice(source.item.as_bytes());
        }

        if let Some(name) = self.name.as_ref() {
            bytes.put_u16(name.len() as u16);
            bytes.put_slice(name.as_bytes());
        } else {
            bytes.put_u16(0);
        }

        if let Some(init) = self.initializer.as_ref() {
            bytes.put_u16(init.locals.len() as u16);
            for local in &init.locals {
                bytes.put_u8(local.into_u8());
            }

            for op in &init.ops {
                op.encode(bytes);
            }

            bytes.put_u8(Op::End.into_u8())
        }
    }

    fn decode(bytes: &mut Bytes) -> Result<Self, DecodeError>
    where
        Self: Sized,
    {
        let linkage = LinkageFlags::from_bits(bytes.get_u16()).ok_or(DecodeError::InvalidFlags)?;
        let ty = ComplexType::decode(bytes)?;

        let source = if linkage.contains(LinkageFlags::IMPORTED) {
            let module = bytes.get_u32();
            let item_len = bytes.get_u16() as usize;
            let item = String::from_utf8(bytes.split_to(item_len).into()).unwrap();
            Some(Source { module, item })
        } else {
            None
        };

        let name_len = bytes.get_u16() as usize;
        let name = if name_len > 0 {
            Some(String::from_utf8_lossy(&bytes.split_to(name_len).to_vec()).into_owned())
        } else {
            None
        };

        let initializer = if linkage.contains(LinkageFlags::HAS_BLOCK) {
            let local_count = bytes.get_u16() as usize;
            let locals = bytes
                .split_to(local_count)
                .into_iter()
                .map(Type::from_u8)
                .collect_vec();
            let mut instructions = Vec::<Instruction>::new();

            loop {
                match Instruction::decode(bytes) {
                    Ok(instruction) => instructions.push(instruction),
                    Err(DecodeError::BlockEnd(Op::End)) => break,
                    Err(e) => return Err(e),
                };
            }

            Some(Block {
                locals,
                ops: instructions,
            })
        } else {
            None
        };

        Ok(Global {
            name,
            source,
            linkage,
            ty,
            mutable: linkage.contains(LinkageFlags::IS_MUTABLE),
            initializer,
        })
    }
}

impl<GlobalTy> Global<GlobalTy> {
    pub fn map_globals<T>(self, f: impl Fn(GlobalTy) -> T + Copy) -> Global<T> {
        let Global {
            name,
            source,
            linkage,
            ty,
            mutable,
            initializer,
        } = self;

        Global::<T> {
            name,
            source: source.map(|s| Source {
                module: f(s.module),
                item: s.item,
            }),
            linkage,
            ty: ty.map_globals(f),
            mutable,
            initializer: initializer.map(|b| b.map_globals(f)),
        }
    }

    pub fn needed_globals(&self) -> impl IntoIterator<Item = &GlobalTy> {
        [self.source.as_ref().map(|v| &v.module)]
            .into_iter()
            .flatten()
            .chain(
                self.initializer
                    .as_ref()
                    .into_iter()
                    .flat_map(|v| v.ops.iter().filter_map(|v| v.referenced_global())),
            )
    }
}

#[derive(Debug, Clone)]
pub struct Function<Global = u32> {
    pub linkage: LinkageFlags,
    pub name: Option<String>,
    pub source: Option<Source<Global>>,
    pub return_type: Option<ComplexType<Global>>,
    pub params: OrderedHashMap<String, ComplexType<Global>>,
    pub block: Option<Block<Global>>,
}

impl BinaryEncodable for Function {
    fn encode(&self, bytes: &mut BytesMut) {
        let mut linkage = self.linkage;
        linkage.set(LinkageFlags::HAS_BLOCK, self.block.is_some());
        linkage.set(LinkageFlags::IMPORTED, self.source.is_some());
        bytes.put_u16(linkage.bits());

        if let Some(source) = &self.source {
            bytes.put_u32(source.module);
            bytes.put_u16(source.item.len() as u16);
            bytes.put_slice(source.item.as_bytes());
        }

        if let Some(name) = self.name.as_ref() {
            bytes.put_u16(name.len() as u16);
            bytes.put_slice(name.as_bytes());
        } else {
            bytes.put_u16(0);
        }

        match &self.return_type {
            None => bytes.put_u8(0xFF),
            Some(v) => v.encode(bytes),
        }

        bytes.put_u16(self.params.len() as u16);

        for (name, param) in &self.params {
            bytes.put_u16(name.len() as u16);
            bytes.put_slice(name.as_bytes());

            param.encode(bytes);
        }

        if let Some(block) = self.block.as_ref() {
            bytes.put_u16(block.locals.len() as u16);
            for local in &block.locals {
                bytes.put_u8(local.into_u8());
            }

            for op in &block.ops {
                op.encode(bytes);
            }

            bytes.put_u8(Op::End.into_u8())
        }
    }

    fn decode(bytes: &mut Bytes) -> Result<Self, DecodeError>
    where
        Self: Sized,
    {
        let linkage = LinkageFlags::from_bits(bytes.get_u16()).ok_or(DecodeError::InvalidFlags)?;

        let source = if linkage.contains(LinkageFlags::IMPORTED) {
            let module = bytes.get_u32();
            let item_len = bytes.get_u16() as usize;
            let item = String::from_utf8(bytes.split_to(item_len).into()).unwrap();
            Some(Source { module, item })
        } else {
            None
        };

        let name_len = bytes.get_u16() as usize;
        let name = if name_len > 0 {
            Some(String::from_utf8_lossy(&bytes.split_to(name_len).to_vec()).into_owned())
        } else {
            None
        };

        let return_type = if bytes[0] != 0xFF {
            Some(ComplexType::decode(bytes)?)
        } else {
            bytes.advance(1);
            None
        };

        let param_len = bytes.get_u16();
        let params = (0..param_len)
            .map(|_| -> Result<(String, ComplexType), DecodeError> {
                let name_len = bytes.get_u16() as usize;
                let name = String::from_utf8_lossy(&bytes.split_to(name_len)).into_owned();

                let ty = ComplexType::decode(bytes)?;

                Ok((name, ty))
            })
            .try_collect()?;

        let block = if linkage.contains(LinkageFlags::HAS_BLOCK) {
            let local_count = bytes.get_u16() as usize;
            let locals = bytes
                .split_to(local_count)
                .into_iter()
                .map(Type::from_u8)
                .collect_vec();
            let mut instructions = Vec::<Instruction>::new();

            loop {
                match Instruction::decode(bytes) {
                    Ok(instruction) => instructions.push(instruction),
                    Err(DecodeError::BlockEnd(Op::End)) => break,
                    Err(e) => return Err(e),
                };
            }

            Some(Block {
                locals,
                ops: instructions,
            })
        } else {
            None
        };

        Ok(Self {
            linkage,
            source,
            name,
            return_type,
            params,
            block,
        })
    }
}

impl<Global> Function<Global> {
    pub fn map_globals<T>(self, f: impl Fn(Global) -> T + Copy) -> Function<T> {
        let Function {
            linkage,
            name,
            source,
            return_type,
            params,
            block,
        } = self;

        Function::<T> {
            linkage,
            name,
            source: source.map(|s| Source {
                module: f(s.module),
                item: s.item,
            }),
            return_type: return_type.map(|v| v.map_globals(&f)),
            params: params
                .into_iter()
                .map(|(k, v)| (k, v.map_globals(&f)))
                .collect(),
            block: block.map(|b| b.map_globals(f)),
        }
    }

    pub fn needed_globals(&self) -> impl IntoIterator<Item = &Global> {
        [self.source.as_ref().map(|v| &v.module)]
            .into_iter()
            .flatten()
            .chain(self.params.values().flat_map(|p| p.needed_globals()))
            .chain(
                self.block
                    .as_ref()
                    .into_iter()
                    .flat_map(|v| v.ops.iter().filter_map(|v| v.referenced_global())),
            )
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Debug, Default)]
#[repr(u8)]
#[non_exhaustive]
pub enum StructLayout {
    #[default]
    Hephaestus = 0,
    CStyle = 1,
}

impl StructLayout {
    pub fn into_u8(self) -> u8 {
        self as u8
    }

    pub fn from_u8(value: u8) -> Self {
        unsafe { mem::transmute(value) }
    }
}

#[derive(Clone, Eq, PartialEq, Debug)]
pub enum ComplexType<Global = u32> {
    Primitive(Type),

    /// Internally considered a [u_ptr][Type::UPtr].
    StructRef(Global),

    /// Considered the same type as it's [backing type][Enum::backing_type].
    Enum(Type, Global),

    AliasedType(Box<ComplexType<Global>>, Global),
}

impl Display for ComplexType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ComplexType::Primitive(ty) => Display::fmt(ty, f),
            ComplexType::StructRef(g) => write!(f, "ref @%{g}"),
            ComplexType::Enum(ty, g) => write!(f, "enum /* {ty} */ @%{g}"),
            ComplexType::AliasedType(_, g) => write!(f, "type @%{g}"),
        }
    }
}

impl<Global> ComplexType<Global> {
    const STRUCT_REF: u8 = 0x3F;
    const ALIAS: u8 = 0x40;
    const ENUM: u8 = 0x80;

    pub fn ty(&self) -> Type {
        match self {
            ComplexType::Primitive(ty) => *ty,
            ComplexType::StructRef(_) => Type::UPtr,
            ComplexType::Enum(ty, _) => *ty,
            ComplexType::AliasedType(ty, _) => ty.ty(),
        }
    }

    pub fn map_globals<T>(self, f: impl Fn(Global) -> T + Copy) -> ComplexType<T> {
        match self {
            ComplexType::Primitive(ty) => ComplexType::Primitive(ty),
            ComplexType::StructRef(g) => ComplexType::StructRef(f(g)),
            ComplexType::Enum(ty, g) => ComplexType::Enum(ty, f(g)),
            ComplexType::AliasedType(complex_type, g) => {
                ComplexType::AliasedType(Box::new(complex_type.map_globals(f)), f(g))
            }
        }
    }

    pub fn needed_globals(&self) -> impl IntoIterator<Item = &Global> {
        match self {
            ComplexType::Primitive(_) => vec![],
            ComplexType::StructRef(g) => vec![g],
            ComplexType::Enum(_, g) => vec![g],
            ComplexType::AliasedType(complex_type, g) => [g]
                .into_iter()
                .chain(complex_type.needed_globals())
                .collect_vec(),
        }
    }
}

impl ComplexType {
    fn read_complex_type(bytes: &mut Bytes, b: u8) -> ComplexType {
        if (b & Self::ENUM) > 0 {
            Self::Enum(Type::from_u8(b & 0xF), bytes.get_u32())
        } else if b == Self::STRUCT_REF {
            Self::StructRef(bytes.get_u32())
        } else {
            Self::Primitive(Type::from_u8(b))
        }
    }
}

impl BinaryEncodable for ComplexType {
    fn encode(&self, bytes: &mut BytesMut) {
        match self {
            ComplexType::Primitive(ty) => bytes.put_u8(ty.into_u8()),
            ComplexType::StructRef(name) => {
                bytes.put_u8(Self::STRUCT_REF);
                bytes.put_u32(*name);
            }
            ComplexType::Enum(ty, name) => {
                bytes.put_u8(Self::ENUM | ty.into_u8());
                bytes.put_u32(*name);
            }
            ComplexType::AliasedType(ty, name) => {
                match **ty {
                    ComplexType::Primitive(t) => bytes.put_u8(Self::ALIAS | t.into_u8()),
                    ComplexType::StructRef(name) => {
                        bytes.put_u8(Self::STRUCT_REF | Self::ALIAS);
                        bytes.put_u32(name);
                    }
                    ComplexType::Enum(t, name) => {
                        bytes.put_u8(Self::ENUM | Self::ALIAS | t.into_u8());
                        bytes.put_u32(name);
                    }
                    ComplexType::AliasedType(_, _) => {
                        panic!("alias must be fully resolved to be encodable")
                    }
                }

                bytes.put_u32(*name);
            }
        }
    }

    fn decode(bytes: &mut Bytes) -> Result<Self, DecodeError>
    where
        Self: Sized,
    {
        let b = bytes.get_u8();

        if (b & Self::ALIAS) > 0 {
            let b = b & !Self::ALIAS;
            let src = Self::read_complex_type(bytes, b);

            Ok(Self::AliasedType(src.into(), bytes.get_u32()))
        } else {
            Ok(Self::read_complex_type(bytes, b))
        }
    }
}

#[derive(Debug, Clone)]
pub struct Struct<GlobalTy = u32> {
    pub layout: StructLayout,
    pub linkage: LinkageFlags,
    pub name: Option<String>,
    pub source: Option<Source<GlobalTy>>,
    pub values: Vec<StructField<GlobalTy>>,
}

impl BinaryEncodable for Struct {
    fn encode(&self, bytes: &mut BytesMut) {
        bytes.put_u8(self.layout.into_u8());

        let mut linkage = self.linkage;
        linkage.set(LinkageFlags::IMPORTED, self.source.is_some());
        bytes.put_u16(linkage.bits());

        if let Some(source) = &self.source {
            bytes.put_u32(source.module);
            bytes.put_u16(source.item.len() as u16);
            bytes.put_slice(source.item.as_bytes());
        }

        if let Some(name) = self.name.as_ref() {
            bytes.put_u16(name.len() as u16);
            bytes.put_slice(name.as_bytes());
        } else {
            bytes.put_u16(0);
        }

        bytes.put_u16(self.values.len() as u16);
        for value in &self.values {
            value.encode(bytes);
        }
    }

    fn decode(bytes: &mut Bytes) -> Result<Self, DecodeError>
    where
        Self: Sized,
    {
        let layout = StructLayout::from_u8(bytes.get_u8());
        let linkage = LinkageFlags::from_bits_retain(bytes.get_u16());

        let source = if linkage.contains(LinkageFlags::IMPORTED) {
            let module = bytes.get_u32();
            let item_len = bytes.get_u16() as usize;
            let item = String::from_utf8(bytes.split_to(item_len).into()).unwrap();
            Some(Source { module, item })
        } else {
            None
        };

        let name_len = bytes.get_u16() as usize;
        let name = if name_len > 0 {
            Some(String::from_utf8_lossy(&bytes.split_to(name_len).to_vec()).into_owned())
        } else {
            None
        };

        let value_len = bytes.get_u16() as usize;
        let values = (0..value_len)
            .map(|_| StructField::decode(bytes))
            .try_collect()?;
        Ok(Self {
            layout,
            linkage,
            source,
            name,
            values,
        })
    }
}

impl<Global> Struct<Global> {
    pub fn map_globals<T>(self, f: impl Fn(Global) -> T + Copy) -> Struct<T> {
        let Struct {
            layout,
            linkage,
            name,
            source,
            values,
        } = self;
        Struct::<T> {
            layout,
            linkage,
            name,
            source: source.map(|s| Source {
                module: f(s.module),
                item: s.item,
            }),
            values: values.into_iter().map(|v| v.map_globals(f)).collect(),
        }
    }

    pub fn needed_globals(&self) -> impl IntoIterator<Item = &Global> {
        self.source
            .iter()
            .map(|v| &v.module)
            .chain(self.values.iter().flat_map(|v| v.needed_globals()))
    }
}

#[derive(Debug, Clone)]
pub struct Enum<Global = u32> {
    pub backing_type: Type,
    pub linkage: LinkageFlags,
    pub name: Option<String>,
    pub source: Option<Source<Global>>,
    pub values: OrderedHashMap<String, Const>,
}

impl BinaryEncodable for Enum {
    fn encode(&self, bytes: &mut BytesMut) {
        bytes.put_u8(self.backing_type.into_u8());

        let mut linkage = self.linkage;
        linkage.set(LinkageFlags::IMPORTED, self.source.is_some());
        bytes.put_u16(linkage.bits());

        if let Some(source) = &self.source {
            bytes.put_u32(source.module);
            bytes.put_u16(source.item.len() as u16);
            bytes.put_slice(source.item.as_bytes());
        }

        if let Some(name) = self.name.as_ref() {
            bytes.put_u16(name.len() as u16);
            bytes.put_slice(name.as_bytes());
        } else {
            bytes.put_u16(0);
        }

        bytes.put_u16(self.values.len() as u16);

        for (name, cst) in &self.values {
            bytes.put_u16(name.len() as u16);
            bytes.put_slice(name.as_bytes());

            cst.encode_value(bytes);
        }
    }

    fn decode(bytes: &mut Bytes) -> Result<Self, DecodeError>
    where
        Self: Sized,
    {
        let backing_type = Type::from_u8(bytes.get_u8());
        let linkage = LinkageFlags::from_bits_retain(bytes.get_u16());

        let source = if linkage.contains(LinkageFlags::IMPORTED) {
            let module = bytes.get_u32();
            let item_len = bytes.get_u16() as usize;
            let item = String::from_utf8(bytes.split_to(item_len).into()).unwrap();
            Some(Source { module, item })
        } else {
            None
        };

        let name_len = bytes.get_u16() as usize;
        let name = if name_len > 0 {
            Some(String::from_utf8_lossy(&bytes.split_to(name_len).to_vec()).into_owned())
        } else {
            None
        };

        let value_len = bytes.get_u16() as usize;
        let values = (0..value_len)
            .map(|_| -> Result<(String, Const), DecodeError> {
                let name_len = bytes.get_u16() as usize;
                let name = String::from_utf8(bytes.split_to(name_len).into())?;
                let value = Const::decode_value(bytes, backing_type)?;
                Ok((name, value))
            })
            .try_collect()?;

        Ok(Self {
            backing_type,
            linkage,
            source,
            name,
            values,
        })
    }
}

impl<Global> Enum<Global> {
    pub fn map_globals<T>(self, f: impl Fn(Global) -> T + Copy) -> Enum<T> {
        let Enum {
            backing_type,
            linkage,
            name,
            source,
            values,
        } = self;

        Enum::<T> {
            backing_type,
            linkage,
            name,
            source: source.map(|s| Source {
                module: f(s.module),
                item: s.item,
            }),
            values,
        }
    }

    pub fn needed_globals(&self) -> impl IntoIterator<Item = &Global> {
        self.source.as_ref().map(|v| &v.module)
    }
}

#[derive(Debug, Clone)]
pub enum StructFieldLayout {
    Automatic,
    Align(u8),
    Custom(i16),
}

impl StructFieldLayout {
    const AUTO: u8 = 0;
    const ALIGN: u8 = 1;
    const CUSTOM: u8 = 2;
}

impl BinaryEncodable for StructFieldLayout {
    fn encode(&self, bytes: &mut BytesMut) {
        match self {
            StructFieldLayout::Automatic => bytes.put_u8(Self::AUTO),
            StructFieldLayout::Align(align) => {
                bytes.put_u8(Self::ALIGN);
                bytes.put_u8(*align);
            }
            StructFieldLayout::Custom(pos) => {
                bytes.put_u8(Self::CUSTOM);
                bytes.put_i16(*pos);
            }
        }
    }

    fn decode(bytes: &mut Bytes) -> Result<Self, DecodeError>
    where
        Self: Sized,
    {
        match bytes.get_u8() {
            Self::AUTO => Ok(Self::Automatic),
            Self::ALIGN => Ok(Self::Align(bytes.get_u8())),
            Self::CUSTOM => Ok(Self::Custom(bytes.get_i16())),
            _ => Err(DecodeError::InvalidVariant),
        }
    }
}

#[derive(Debug, Clone)]
pub enum StructField<Global = u32> {
    Data {
        layout: StructFieldLayout,
        name: String,
        ty: ComplexType<Global>,
    },
    EmptySpace(u16),
}

impl<Global> StructField<Global> {
    const DATA: u8 = 0x00;
    const EMPTY_SPACE: u8 = 0x01;

    pub fn map_globals<T>(self, f: impl Fn(Global) -> T + Copy) -> StructField<T> {
        match self {
            StructField::Data { layout, name, ty } => StructField::Data {
                layout,
                name,
                ty: ty.map_globals(f),
            },
            StructField::EmptySpace(i) => StructField::EmptySpace(i),
        }
    }

    pub fn needed_globals(&self) -> impl IntoIterator<Item = &Global> {
        match self {
            StructField::Data { ty, .. } => ty.needed_globals().into_iter().collect_vec(),
            StructField::EmptySpace(_) => vec![],
        }
    }
}

impl BinaryEncodable for StructField {
    fn encode(&self, bytes: &mut BytesMut) {
        match self {
            StructField::Data { layout, name, ty } => {
                bytes.put_u8(Self::DATA);
                layout.encode(bytes);
                ty.encode(bytes);

                bytes.put_u16(name.len() as u16);
                bytes.put_slice(name.as_bytes());
            }
            StructField::EmptySpace(padding) => {
                bytes.put_u8(Self::EMPTY_SPACE);
                bytes.put_u16(*padding);
            }
        }
    }

    fn decode(bytes: &mut Bytes) -> Result<Self, DecodeError>
    where
        Self: Sized,
    {
        match bytes.get_u8() {
            Self::DATA => Ok(Self::Data {
                layout: StructFieldLayout::decode(bytes)?,
                ty: ComplexType::decode(bytes)?,
                name: {
                    let len = bytes.get_u16() as usize;
                    String::from_utf8(bytes.split_to(len).into())?
                },
            }),
            Self::EMPTY_SPACE => Ok(Self::EmptySpace(bytes.get_u16())),
            _ => Err(DecodeError::InvalidVariant),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Block<Global = u32> {
    pub locals: Vec<Type>,
    pub ops: Vec<Instruction<Global>>,
}

impl<Global> Block<Global> {
    pub fn map_globals<T>(self, f: impl Fn(Global) -> T + Copy) -> Block<T> {
        let Block { locals, ops } = self;
        Block::<T> {
            locals,
            ops: ops.into_iter().map(|v| v.map_globals(f)).collect(),
        }
    }
}

struct Locals<T: Display>(T);

impl<T: Display> Display for Locals<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "locals [{}]", self.0)
    }
}

impl<Global: Display> Display for Block<Global> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str("{")?;

        if !self.locals.is_empty() {
            write!(
                f,
                "\n{}\n",
                Indent(Locals(self.locals.iter().join(", ")), 1)
            )?;
        }

        for i in self.ops.iter().map(|l| Indent(l, 1)) {
            write!(f, "\n{i}")?;
        }

        f.write_str("\n}")
    }
}

struct Indent<T: Display>(T, usize);

impl<T: Display> Display for Indent<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let s = self.0.to_string();
        f.write_str(&*s.lines().map(|l| "    ".repeat(self.1) + l).join("\n"))
    }
}
