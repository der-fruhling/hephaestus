pub mod asm;
mod instruction;

use bitflags::bitflags;
use bytes::{Buf, BufMut, Bytes, BytesMut};
use itertools::Itertools;
use ordered_hash_map::OrderedHashMap;
use std::{collections::HashMap, fmt::{Debug, Display, Formatter}};
use thiserror::Error;
use instruction::{Instruction, Op};

#[derive(Default, Debug)]
pub struct Module {
    pub flags: ModuleFlags,
    pub custom_features: Vec<CustomFeatureDeclaration>,
    pub imported_modules: OrderedHashMap<String, ImportedModule>,
    pub globals: Vec<Item>,
    functions: HashMap<String, usize>,
    global_vars: HashMap<String, usize>
}

#[derive(Debug)]
pub struct CustomFeatureDeclaration {
    pub name: String,
    pub content: Bytes
}

impl Module {
    pub(crate) fn recalculate(&mut self) {
        self.functions.clear();
        self.global_vars.clear();

        for (i, item) in self.globals.iter().enumerate() {
            match item {
                Item::Function(f) => {
                    if f.name.is_some() { self.functions.insert(f.name.clone().unwrap(), i); }
                },
                Item::Global(f) => {
                    if f.name.is_some() { self.global_vars.insert(f.name.clone().unwrap(), i); }
                }
            };
        }
    }
    
    pub fn as_bytes(&self) -> Bytes {
        let mut bytes = BytesMut::new();
        self.encode(&mut bytes);
        bytes.freeze()
    }
}

#[derive(Debug)]
pub enum ImportedModule {
    /// Refers to a native library. Actual resolved path is system dependent.
    ///
    /// The found library is linked dynamicly.
    ///
    /// Only supported on native platforms.
    DynamicLibrary(String),

    /// Refers to a native library. Actual resolved path is system dependent.
    ///
    /// The found library is linked staticly.
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
    WitComponent(Bytes)
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
            ImportedModule::WitComponent(content) => {
                bytes.put_u8(Self::WIT_COMPONENT);
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
                let name = String::from_utf8_lossy(&bytes[..name_len]).into_owned();
                bytes.advance(name_len);

                Ok(ImportedModule::DynamicLibrary(name))
            }
            Self::STATIC_LIBRARY => {
                let name_len = bytes.get_u16() as usize;
                let name = String::from_utf8_lossy(&bytes[..name_len]).into_owned();
                bytes.advance(name_len);

                Ok(ImportedModule::StaticLibrary(name))
            }
            Self::WIT_COMPONENT => {
                let content_len = bytes.get_u32() as usize;
                let content = Bytes::copy_from_slice(&bytes[..content_len]);
                bytes.advance(content_len);

                Ok(ImportedModule::WitComponent(content))
            }
            _ => Err(DecodeError::InvalidImportedModuleType)
        }
    }
}

#[repr(u16)]
pub enum Target {
    /// Indicates that the target is not known.
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
    WebAssembly32PartialComponent = 0xD011
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
        const HAS_CUSTOM_FEATURES = 1 << 0;
        const SUPPORTS_WIT_COMPONENTS = 1 << 1;
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
    #[error("invalid item type: {0}")]
    InvalidItemType(u8),

    #[error("erroneous block end op: {0:?}")]
    BlockEnd(Op),
}

pub trait BinaryEncodable {
    fn encode(&self, bytes: &mut BytesMut);
    fn decode(bytes: &mut Bytes) -> Result<Self, DecodeError> where Self: Sized;
}

const MODULE_MAGIC: u32 = 0xCAFEEE77;
const CURRENT_VERSION: u16 = 0;

type ModuleDecoderFn = fn(&mut Bytes) -> Result<Module, DecodeError>;

static MODULE_DECODERS: &[ModuleDecoderFn] = &[
    |bytes| -> Result<Module, DecodeError> {
        let flags = ModuleFlags::from_bits_retain(bytes.get_u32());

        let custom_features = if flags.contains(ModuleFlags::HAS_CUSTOM_FEATURES) {
            let mut custom_features = vec![];
            let feature_len = bytes.get_u16() as usize;

            for _ in 0..feature_len {
                let name_len = bytes.get_u16() as usize;
                let name = String::from_utf8_lossy(&bytes.split_to(name_len)).into_owned();
                let content_len = bytes.get_u16() as usize;
                let content = bytes.split_to(content_len);
                custom_features.push(CustomFeatureDeclaration {
                    name,
                    content
                })
            }

            custom_features
        } else { vec![] };

        let imported_module_len = bytes.get_u16();
        let imported_modules = (0..imported_module_len).map(|_| {
            let name_len = bytes.get_u16() as usize;
            let name = String::from_utf8_lossy(&bytes.split_to(name_len)).into_owned();
            ImportedModule::decode(bytes).map(|n| (name, n))
        }).try_collect()?;

        let global_len = bytes.get_u32() as usize;
        let mut globals = vec![];

        for _ in 0..global_len {
            globals.push(Item::decode(bytes)?);
        }

        let mut module = Module {
            flags,
            custom_features,
            imported_modules,
            globals,
            functions: Default::default(),
            global_vars: Default::default(),
        };

        module.recalculate();

        Ok(module)
    }
];

impl BinaryEncodable for Module {
    fn encode(&self, bytes: &mut BytesMut) {
        bytes.put_u32(MODULE_MAGIC);
        bytes.put_u16(CURRENT_VERSION);

        // everything after this comment can be changed if a decoder is written
        // for the new version.
        let mut flags = self.flags;
        flags.set(ModuleFlags::HAS_CUSTOM_FEATURES, !self.custom_features.is_empty());
        bytes.put_u32(flags.bits());

        if !self.custom_features.is_empty() {
            bytes.put_u16(self.custom_features.len() as u16);

            for f in &self.custom_features {
                bytes.put_u16(f.name.len() as u16);
                bytes.put_slice(f.name.as_bytes());
                bytes.put_u16(f.content.len() as u16);
                bytes.put(f.content.clone());
            }
        }

        bytes.put_u16(self.imported_modules.len() as u16);

        for (name, module) in &self.imported_modules {
            bytes.put_u16(name.len() as u16);
            bytes.put_slice(name.as_bytes());

            module.encode(bytes);
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
        if magic != MODULE_MAGIC { return Err(DecodeError::MissingMagic); }

        let version = bytes.get_u16();
        if version > CURRENT_VERSION {
            return Err(DecodeError::UnsupportedVersion(version))
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
    IPtr = 12
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
        unsafe { std::mem::transmute(val) }
    }
}

#[derive(Debug)]
pub enum Item {
    Global(Global),
    Function(Function)
}

impl Item {
    const GLOBAL: u8 = 1;
    const FUNCTION: u8 = 2;
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
        }
    }

    fn decode(bytes: &mut Bytes) -> Result<Self, DecodeError>
    where
        Self: Sized,
    {
        match bytes.get_u8() {
            Self::GLOBAL => Ok(Item::Global(Global::decode(bytes)?)),
            Self::FUNCTION => Ok(Item::Function(Function::decode(bytes)?)),
            other => Err(DecodeError::InvalidItemType(other))
        }
    }
}

#[derive(Debug)]
pub struct Global {
    pub name: Option<String>,
    pub linkage: LinkageFlags,
    pub ty: Type,
    pub mutable: bool,
    pub initializer: Option<Block>
}

impl BinaryEncodable for Global {
    fn encode(&self, bytes: &mut BytesMut) {
        let mut linkage = self.linkage;
        linkage.set(LinkageFlags::IS_MUTABLE, self.mutable);
        linkage.set(LinkageFlags::HAS_BLOCK, self.initializer.is_some());

        bytes.put_u16(linkage.bits());
        bytes.put_u8(self.ty.into_u8());

        if linkage.contains(LinkageFlags::EXPORTED) || linkage.contains(LinkageFlags::IMPORTED) {
            let name = self.name.as_ref().unwrap();
            bytes.put_u16(name.len() as u16);
            bytes.put_slice(name.as_bytes());
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
        let ty = Type::from_u8(bytes.get_u8());

        let name = if linkage.contains(LinkageFlags::EXPORTED) || linkage.contains(LinkageFlags::IMPORTED) {
            let name_len = bytes.get_u16() as usize;
            Some(String::from_utf8_lossy(&bytes.split_to(name_len).to_vec()).into_owned())
        } else {
            None
        };

        let initializer = if linkage.contains(LinkageFlags::HAS_BLOCK) {
            let local_count = bytes.get_u16() as usize;
            let locals = bytes.split_to(local_count).into_iter().map(Type::from_u8).collect_vec();
            let mut instructions = Vec::<Instruction>::new();

            loop {
                match Instruction::decode(bytes) {
                    Ok(instruction) => instructions.push(instruction),
                    Err(DecodeError::BlockEnd(Op::End)) => break,
                    Err(e) => return Err(e)
                };
            }

            Some(Block {
                locals,
                ops: instructions
            })
        } else { None };

        Ok(Global {
            name,
            linkage,
            ty,
            mutable: linkage.contains(LinkageFlags::IS_MUTABLE),
            initializer,
        })
    }
}

#[derive(Debug)]
pub struct Function {
    pub linkage: LinkageFlags,
    pub name: Option<String>,
    pub return_type: Option<Type>,
    pub params: OrderedHashMap<String, Type>,
    pub block: Option<Block>
}

impl BinaryEncodable for Function {
    fn encode(&self, bytes: &mut BytesMut) {
        let mut linkage = self.linkage;
        linkage.set(LinkageFlags::HAS_BLOCK, self.block.is_some());
        bytes.put_u16(linkage.bits());

        if linkage.contains(LinkageFlags::EXPORTED) || linkage.contains(LinkageFlags::IMPORTED) {
            let name = self.name.as_ref().unwrap();
            bytes.put_u16(name.len() as u16);
            bytes.put_slice(name.as_bytes());
        }

        bytes.put_u8(self.return_type.map(Type::into_u8).unwrap_or(0xFF));

        bytes.put_u16(self.params.len() as u16);

        for (name, param) in &self.params {
            bytes.put_u16(name.len() as u16);
            bytes.put_slice(name.as_bytes());

            bytes.put_u8(param.into_u8());
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

        let name = if linkage.contains(LinkageFlags::EXPORTED) || linkage.contains(LinkageFlags::IMPORTED) {
            let name_len = bytes.get_u16() as usize;
            Some(String::from_utf8_lossy(&bytes.split_to(name_len).to_vec()).into_owned())
        } else {
            None
        };

        let return_type = bytes.get_u8();
        let return_type = if return_type != 0xFF { Some(Type::from_u8(return_type)) } else { None };

        let param_len = bytes.get_u16();
        let params = (0..param_len).map(|_| {
            let name_len = bytes.get_u16() as usize;
            let name = String::from_utf8_lossy(&bytes.split_to(name_len)).into_owned();

            let ty = Type::from_u8(bytes.get_u8());

            (name, ty)
        }).collect();

        let block = if linkage.contains(LinkageFlags::HAS_BLOCK) {
            let local_count = bytes.get_u16() as usize;
            let locals = bytes.split_to(local_count).into_iter().map(Type::from_u8).collect_vec();
            let mut instructions = Vec::<Instruction>::new();

            loop {
                match Instruction::decode(bytes) {
                    Ok(instruction) => instructions.push(instruction),
                    Err(DecodeError::BlockEnd(Op::End)) => break,
                    Err(e) => return Err(e)
                };
            }

            Some(Block {
                locals,
                ops: instructions
            })
        } else { None };

        Ok(Self {
            linkage,
            name,
            return_type,
            params,
            block
        })
    }
}

#[derive(Debug, Clone)]
pub struct Block {
    pub locals: Vec<Type>,
    pub ops: Vec<Instruction>
}

struct Indent<T: Display>(T, usize);

impl<T: Display> Display for Indent<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let s = self.0.to_string();
        f.write_str(&*s.lines().map(|l| "    ".repeat(self.1) + l).join("\n"))
    }
}
