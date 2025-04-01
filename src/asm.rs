use bytes::{BufMut, Bytes, BytesMut};
use itertools::Itertools;
use ordered_hash_map::OrderedHashMap;
use pest::{error::LineColLocation, iterators::Pair, Parser};
use std::fmt::Formatter;
use std::vec::IntoIter;
use std::{borrow::Cow, fmt::Display, str::FromStr};
use thiserror::Error;

use crate::instruction::Instruction;
use crate::{
    ComplexType, Const, ImportedModule, Item, LinkageFlags, MetadataContentKind,
    MetadataDeclaration, Module, ModuleFlags, Source, StructField, StructFieldLayout, StructLayout,
    Target, Type,
};

mod grammar {
    #[derive(pest_derive::Parser)]
    #[grammar = "heph.pest"]
    pub struct Grammar;
}

use grammar::{Grammar, Rule};

#[derive(Clone, Copy, Debug)]
pub struct LineCol {
    pub line: usize,
    pub col: usize,
}

impl Display for LineCol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.line, self.col)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Span(pub LineCol, pub Option<LineCol>);

impl From<LineColLocation> for Span {
    fn from(value: LineColLocation) -> Self {
        match value {
            LineColLocation::Pos((line, col)) => Self(LineCol { line, col }, None),
            LineColLocation::Span((fl, fc), (tl, tc)) => Self(
                LineCol { line: fl, col: fc },
                Some(LineCol { line: tl, col: tc }),
            ),
        }
    }
}

impl Display for Span {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(to) = self.1 {
            let from = self.0;
            write!(f, "{from}-{to}")
        } else {
            Display::fmt(&self.0, f)
        }
    }
}

#[derive(Error, Debug)]
#[error("{tgt:?} : {message}")]
pub struct AsmError {
    pub message: String,
    pub tgt: Option<Span>,
}

impl From<pest::error::Error<Rule>> for AsmError {
    fn from(value: pest::error::Error<Rule>) -> Self {
        Self {
            message: value.variant.to_string(),
            tgt: Some(value.line_col.into()),
        }
    }
}

impl AsmError {
    pub fn new_for(tgt: &Pair<'_, Rule>, msg: impl Into<String>) -> Self {
        let (line, col) = tgt.line_col();

        Self {
            tgt: Some(Span(LineCol { line, col }, None)),
            message: msg.into(),
        }
    }
}

#[derive(Default, Debug)]
pub struct ParsedHeph {
    target: Target,
    module_flags: ModuleFlags,
    globals: OrderedHashMap<String, Global>,
}

impl ParsedHeph {
    pub fn finish(self) -> Module {
        let mut module = Module::default();
        module.flags = self.module_flags;
        module.target = self.target;

        for (name, global) in self.globals {
            match global {
                Global::Var(var) => {
                    let (linkage, source) = var.linkage.split();
                    let global_item = crate::Global {
                        linkage,
                        source,
                        name: Some(name),
                        ty: var.ty,
                        mutable: var.mutable,
                        initializer: var.block.map(collect_block),
                    };

                    module.globals.push(Item::Global(global_item));
                }
                Global::Function(funct) => {
                    let (linkage, source) = funct.linkage.split();
                    let funct_item = crate::Function {
                        linkage,
                        source,
                        name: Some(name),
                        return_type: funct.return_type,
                        params: funct.params,
                        block: funct.block.map(collect_block),
                    };

                    module.globals.push(Item::Function(funct_item));
                }
                Global::ImportedModule(m) => {
                    module.globals.push(Item::ImportedModule(m));
                }
                Global::Struct(s) => {
                    let (linkage, source) = s.linkage.split();
                    module.globals.push(Item::Struct(crate::Struct {
                        layout: s.layout,
                        linkage,
                        source,
                        name: Some(name),
                        values: s.values,
                    }))
                }
                Global::Enum(e) => {
                    let (linkage, source) = e.linkage.split();
                    module.globals.push(Item::Enum(crate::Enum {
                        backing_type: e.backing_type,
                        linkage,
                        source,
                        name: Some(name),
                        values: e.values,
                    }))
                }
            }
        }

        module.recalculate();
        module
    }
}

fn collect_block(block: Block) -> crate::Block {
    crate::Block {
        locals: block
            .locals
            .values()
            .skip(block.inherited_locals)
            .map(|c| c.ty())
            .collect(),
        ops: block.instructions,
    }
}

#[derive(Debug)]
pub enum Global {
    Var(Var),
    Function(Function),
    ImportedModule(ImportedModule),
    Struct(Struct),
    Enum(Enum),
}

#[derive(Error, Debug, PartialEq, Eq)]
#[error("error parsing type \"{0}\"")]
pub struct TypeParseError(String);

impl FromStr for Type {
    type Err = TypeParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "u8" => Self::U8,
            "u16" => Self::U16,
            "u32" => Self::U32,
            "u64" => Self::U64,
            "i8" => Self::I8,
            "i16" => Self::I16,
            "i32" => Self::I32,
            "i64" => Self::I64,
            "f32" => Self::F32,
            "f64" => Self::F64,
            "u_ptr" => Self::UPtr,
            "i_ptr" => Self::IPtr,
            other => return Err(TypeParseError(other.into())),
        })
    }
}

impl TryFrom<SpannedStr<'_>> for Type {
    type Error = AsmError;

    fn try_from(value: SpannedStr<'_>) -> Result<Self, Self::Error> {
        Type::from_str(value.text).map_err(|e| AsmError {
            tgt: value.span,
            message: e.to_string(),
        })
    }
}

impl ComplexType {
    pub fn parse(parsed: &ParsedHeph, value: Pair<'_, Rule>) -> Result<Self, AsmError> {
        match value.as_rule() {
            Rule::ty => Type::try_from(SpannedStr::from(value)).map(ComplexType::Primitive),
            Rule::ref_type => {
                let (name, _) =
                    find_global_by_name(parsed, Name::from(value.into_inner().next().unwrap()))?;
                Ok(ComplexType::StructRef(name))
            }
            Rule::enum_type => {
                let (name, g) = find_global_by_name(
                    parsed,
                    Name::from(value.clone().into_inner().next().unwrap()),
                )?;

                let ty = if let Global::Enum(e) = g {
                    e.backing_type
                } else {
                    return Err(AsmError::new_for(&value, "not an enum"));
                };

                Ok(ComplexType::Enum(ty, name))
            }
            other => unreachable!("invalid complex type rule: {:?}", other),
        }
    }
}

#[derive(Debug)]
pub enum Linkage {
    Export(LinkageFlags),
    Import(Box<Linkage>, u32, String),
    Internal(InlinePref),
    Expected(Box<Linkage>),
}

impl Linkage {
    pub fn split(self) -> (LinkageFlags, Option<Source>) {
        match self {
            Linkage::Export(flags) => (flags, None),
            Linkage::Import(linkage, module, item) => {
                (linkage.split().0, Some(Source { module, item }))
            }
            Linkage::Internal(pref) => match pref {
                InlinePref::Force => (
                    LinkageFlags::INTERNAL | LinkageFlags::INLINE | LinkageFlags::FORCE_INLINE,
                    None,
                ),
                InlinePref::Prefer => (LinkageFlags::INTERNAL | LinkageFlags::INLINE, None),
                InlinePref::Never => (LinkageFlags::INTERNAL, None),
            },
            Linkage::Expected(linkage) => (LinkageFlags::EXPECTED | linkage.split().0, None),
        }
    }
}

#[derive(Debug)]
pub enum InlinePref {
    Force,
    Prefer,
    Never,
}

#[derive(Debug)]
pub struct Function {
    pub linkage: Linkage,
    pub params: OrderedHashMap<String, ComplexType>,
    pub return_type: Option<ComplexType>,
    pub block: Option<Block>,
}

#[derive(Debug)]
pub struct Var {
    pub ty: ComplexType,
    pub linkage: Linkage,
    pub mutable: bool,
    pub block: Option<Block>,
}

#[derive(Debug)]
pub struct Struct {
    pub layout: StructLayout,
    pub linkage: Linkage,
    pub values: Vec<StructField>,
}

#[derive(Debug)]
pub struct Enum {
    pub backing_type: Type,
    pub linkage: Linkage,
    pub values: OrderedHashMap<String, Const>,
}

#[derive(Debug, Default)]
pub struct Block {
    pub inherited_locals: usize,
    // locals should be predictable
    pub locals: OrderedHashMap<String, ComplexType>,
    pub return_type: Option<ComplexType>,
    pub instructions: Vec<Instruction>,
}

#[derive(Copy, Clone)]
struct SpannedStr<'a> {
    pub text: &'a str,
    pub span: Option<Span>,
}

impl<'a> From<Pair<'a, Rule>> for SpannedStr<'a> {
    fn from(value: Pair<'a, Rule>) -> Self {
        let (line, col) = value.line_col();

        Self {
            text: value.as_str(),
            span: Some(Span(LineCol { line, col }, None)),
        }
    }
}

enum ParsedInstruction<'a> {
    Nop,
    ConstInt(SpannedStr<'a>, Pair<'a, Rule>),
    ConstFp(SpannedStr<'a>, SpannedStr<'a>),
    GlobalSet(SpannedStr<'a>, Name<'a>),
    GlobalGet(SpannedStr<'a>, Name<'a>),
    GlobalTee(SpannedStr<'a>, Name<'a>),
    LocalSet(SpannedStr<'a>, Name<'a>),
    LocalGet(SpannedStr<'a>, Name<'a>),
    LocalTee(SpannedStr<'a>, Name<'a>),
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Bor,
    Band,
    Bxor,
    Inv,
    Cast(SpannedStr<'a>, SpannedStr<'a>),
    Return,
    Call(Name<'a>),
    CallDynamic,
    GetFnUPtr(Name<'a>),
    IfElse(Block, Block),
    If(Block),
    Loop(Block),
    Break(Option<SpannedStr<'a>>),
    Continue(Option<SpannedStr<'a>>),
    MemAlloc,
    MemRealloc,
    MemFree,
    MemLoad(SpannedStr<'a>, Option<SpannedStr<'a>>),
    MemStore(SpannedStr<'a>, Option<SpannedStr<'a>>),
    Discard(Option<Pair<'a, Rule>>),
    Duplicate(Option<Pair<'a, Rule>>),
    Cmp,
    TestGt,
    TestGeq,
    TestLt,
    TestLeq,
    TestEq,
    TestNeq,
}

enum MappedInstruction {
    Single(Instruction),
    Many(Box<[Instruction]>),
}

enum MappedInstructionIterator {
    Single(Option<Instruction>),
    Many(IntoIter<Instruction>),
}

impl Iterator for MappedInstructionIterator {
    type Item = Instruction;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            MappedInstructionIterator::Single(v) => v.take(),
            MappedInstructionIterator::Many(v) => v.next(),
        }
    }
}

impl IntoIterator for MappedInstruction {
    type Item = Instruction;
    type IntoIter = MappedInstructionIterator;

    fn into_iter(self) -> Self::IntoIter {
        match self {
            MappedInstruction::Single(v) => MappedInstructionIterator::Single(Some(v)),
            MappedInstruction::Many(v) => MappedInstructionIterator::Many(v.into_vec().into_iter()),
        }
    }
}

impl From<Instruction> for MappedInstruction {
    #[inline]
    fn from(value: Instruction) -> Self {
        Self::Single(value)
    }
}

impl<const N: usize> From<[Instruction; N]> for MappedInstruction {
    #[inline]
    fn from(value: [Instruction; N]) -> Self {
        Self::Many(value.into_iter().collect())
    }
}

impl From<Vec<Instruction>> for MappedInstruction {
    #[inline]
    fn from(value: Vec<Instruction>) -> Self {
        Self::Many(value.into_boxed_slice())
    }
}

fn parse_int(text: Pair<Rule>) -> i128 {
    let unparsed = text.as_str();

    match text.into_inner().next().map(|p| p.as_rule()) {
        Some(Rule::hex) => {
            let num = if unparsed.starts_with(['-', '+']) {
                Cow::Owned(String::from(&unparsed[0..=0]) + &unparsed[3..])
            } else {
                Cow::Borrowed(&unparsed[2..])
            };

            // this should always succeed if the rule is setup correctly
            i128::from_str_radix(&num, 16).unwrap()
        }
        None => i128::from_str_radix(unparsed, 10).unwrap(),
        Some(other) => unreachable!("illegal rule for int number specialization: {other:?}"),
    }
}

impl<'a> ParsedInstruction<'a> {
    pub fn map_to_instruction(
        self,
        ctx: &ParsedHeph,
        block: &Block,
    ) -> Result<MappedInstruction, AsmError> {
        Ok(match self {
            ParsedInstruction::Nop => Instruction::Nop.into(),
            ParsedInstruction::ConstInt(ty, int) => {
                let ty = Type::try_from(ty)?;
                let parsed_int = parse_int(int.clone());

                let c = match ty {
                    Type::U8 => Const::U8(parsed_int.try_into().map_err(|_| AsmError::new_for(&int, "invalid number for type"))?),
                    Type::U16 => Const::U16(parsed_int.try_into().map_err(|_| AsmError::new_for(&int, "invalid number for type"))?),
                    Type::U32 => Const::U32(parsed_int.try_into().map_err(|_| AsmError::new_for(&int, "invalid number for type"))?),
                    Type::U64 => Const::U64(parsed_int.try_into().map_err(|_| AsmError::new_for(&int, "invalid number for type"))?),
                    Type::I8 => Const::I8(parsed_int.try_into().map_err(|_| AsmError::new_for(&int, "invalid number for type"))?),
                    Type::I16 => Const::I16(parsed_int.try_into().map_err(|_| AsmError::new_for(&int, "invalid number for type"))?),
                    Type::I32 => Const::I32(parsed_int.try_into().map_err(|_| AsmError::new_for(&int, "invalid number for type"))?),
                    Type::I64 => Const::I64(parsed_int.try_into().map_err(|_| AsmError::new_for(&int, "invalid number for type"))?),
                    other => unreachable!("illegal type for integer const (this shouldve been caught earlier!): {other:?}")
                };

                Instruction::Const(c).into()
            }
            ParsedInstruction::ConstFp(ty, number) => {
                let ty = Type::try_from(ty)?;
                let parsed_f64 = f64::from_str(number.text).map_err(|e| AsmError {
                    tgt: number.span,
                    message: e.to_string(),
                })?;

                let c = match ty {
                    Type::F32 => Const::F32(parsed_f64 as f32),
                    Type::F64 => Const::F64(parsed_f64),
                    other => unreachable!(
                        "illegal type for fp const (this shouldve been caught earlier!): {other:?}"
                    ),
                };

                Instruction::Const(c).into()
            }
            ParsedInstruction::GlobalSet(ty, global) => {
                let ty = Type::try_from(ty)?;
                let (name, _) = find_global_by_name(ctx, global)?;

                Instruction::SetGlobal(ty, name).into()
            }
            ParsedInstruction::GlobalGet(ty, global) => {
                let ty = Type::try_from(ty)?;
                let (name, _) = find_global_by_name(ctx, global)?;

                Instruction::GetGlobal(ty, name).into()
            }
            ParsedInstruction::GlobalTee(ty, global) => {
                let ty = Type::try_from(ty)?;
                let (name, _) = find_global_by_name(ctx, global)?;

                [Instruction::Duplicate, Instruction::SetGlobal(ty, name)].into()
            }
            ParsedInstruction::LocalSet(ty, local) => {
                let ty = Type::try_from(ty)?;
                let index = find_local_by_name(block, local)?;
                Instruction::SetLocal(ty, index).into()
            }
            ParsedInstruction::LocalGet(ty, local) => {
                let ty = Type::try_from(ty)?;
                let index = find_local_by_name(block, local)?;
                Instruction::GetLocal(ty, index).into()
            }
            ParsedInstruction::LocalTee(ty, local) => {
                let ty = Type::try_from(ty)?;
                let index = find_local_by_name(block, local)?;
                [Instruction::Duplicate, Instruction::SetLocal(ty, index)].into()
            }
            ParsedInstruction::Add => Instruction::Add.into(),
            ParsedInstruction::Sub => Instruction::Sub.into(),
            ParsedInstruction::Mul => Instruction::Mul.into(),
            ParsedInstruction::Div => Instruction::Div.into(),
            ParsedInstruction::Rem => Instruction::Rem.into(),
            ParsedInstruction::Bor => Instruction::BitOr.into(),
            ParsedInstruction::Band => Instruction::BitAnd.into(),
            ParsedInstruction::Bxor => Instruction::BitXor.into(),
            ParsedInstruction::Inv => Instruction::Inv.into(),
            ParsedInstruction::Cast(from, to) => {
                let from = Type::try_from(from)?;
                let to = Type::try_from(to)?;

                Instruction::Cast(from, to).into()
            }
            ParsedInstruction::Return => Instruction::Return.into(),
            ParsedInstruction::Call(name) => {
                Instruction::Call(Self::get_global_by_name(ctx, name)?).into()
            }
            ParsedInstruction::CallDynamic => Instruction::CallDynamic.into(),
            ParsedInstruction::GetFnUPtr(name) => {
                Instruction::GetFnUPtr(Self::get_global_by_name(ctx, name)?).into()
            }
            ParsedInstruction::IfElse(if_true, if_false) => {
                Instruction::IfElse(collect_block(if_true), collect_block(if_false)).into()
            }
            ParsedInstruction::If(blk) => Instruction::If(collect_block(blk)).into(),
            ParsedInstruction::Loop(blk) => Instruction::Loop(collect_block(blk)).into(),
            ParsedInstruction::Break(depth) => Instruction::Break(Self::parse_maybe(depth)?).into(),
            ParsedInstruction::Continue(depth) => {
                Instruction::Continue(Self::parse_maybe(depth)?).into()
            }
            ParsedInstruction::MemAlloc => Instruction::Alloc.into(),
            ParsedInstruction::MemRealloc => Instruction::Realloc.into(),
            ParsedInstruction::MemFree => Instruction::Free.into(),
            ParsedInstruction::MemLoad(ty, off) => {
                Instruction::Load(Type::try_from(ty)?, Self::parse_maybe(off)?).into()
            }
            ParsedInstruction::MemStore(ty, off) => {
                Instruction::Store(Type::try_from(ty)?, Self::parse_maybe(off)?).into()
            }
            ParsedInstruction::Discard(None) => Instruction::Discard.into(),
            ParsedInstruction::Discard(Some(rep)) => {
                let parsed_int = parse_int(rep.clone());
                let rep: usize = parsed_int
                    .try_into()
                    .map_err(|_| AsmError::new_for(&rep, "too many repetitions!"))?;

                vec![Instruction::Discard; rep].into()
            }
            ParsedInstruction::Duplicate(None) => Instruction::Duplicate.into(),
            ParsedInstruction::Duplicate(Some(rep)) => {
                let parsed_int = parse_int(rep.clone());
                let rep: usize = parsed_int
                    .try_into()
                    .map_err(|_| AsmError::new_for(&rep, "too many repetitions!"))?;

                vec![Instruction::Duplicate; rep].into()
            }
            ParsedInstruction::Cmp => Instruction::CmpOrd.into(),
            ParsedInstruction::TestGt => Instruction::TestGt.into(),
            ParsedInstruction::TestGeq => Instruction::TestGtEq.into(),
            ParsedInstruction::TestLt => Instruction::TestLt.into(),
            ParsedInstruction::TestLeq => Instruction::TestLtEq.into(),
            ParsedInstruction::TestEq => Instruction::TestEq.into(),
            ParsedInstruction::TestNeq => Instruction::TestNeq.into(),
        })
    }

    fn parse_maybe<T: FromStr + Default>(depth: Option<SpannedStr>) -> Result<T, AsmError>
    where
        T::Err: Display,
    {
        depth
            .map(|depth| {
                T::from_str(depth.text).map_err(|e| AsmError {
                    tgt: depth.span,
                    message: e.to_string(),
                })
            })
            .unwrap_or_else(|| Ok(T::default()))
    }

    fn get_global_by_name(ctx: &ParsedHeph, name: Name) -> Result<u32, AsmError> {
        Ok(ctx
            .globals
            .keys()
            .enumerate()
            .find(|s| match name {
                Name::Named(n) => s.1 == n.text,
                Name::Numeric(_, idx, _) => s.0 == idx,
            })
            .map(|v| v.0)
            .ok_or_else(|| AsmError {
                tgt: match name {
                    Name::Named(n) => n.span,
                    Name::Numeric(_, _, span) => Some(span),
                },
                message: format!(
                    "unknown global {}; did you forget to define it above this usage?",
                    name
                ),
            })? as u32)
    }
}

#[derive(Copy, Clone)]
enum Name<'a> {
    Named(SpannedStr<'a>),
    Numeric(char, usize, Span),
}

impl<'a> From<Pair<'a, Rule>> for Name<'a> {
    fn from(value: Pair<'a, Rule>) -> Self {
        let text = SpannedStr::from(value.clone());
        let v = value.into_inner().next();

        match v.as_ref().map(|v| v.as_rule()) {
            Some(Rule::special_number) => {
                let v = v.unwrap().into_inner().next().unwrap();
                debug_assert_eq!(v.as_rule(), Rule::number);
                let (line, col) = v.line_col();
                Self::Numeric(
                    text.text.chars().next().unwrap(),
                    parse_int(v) as usize,
                    Span(LineCol { line, col }, None),
                )
            }
            Some(other) => unreachable!("illegal name rule: {:?}", other),
            None => Self::Named(text.into()),
        }
    }
}

impl<'a> From<&'a str> for Name<'a> {
    fn from(value: &'a str) -> Self {
        Self::Named(SpannedStr {
            text: value,
            span: None,
        })
    }
}

impl Display for Name<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Name::Named(n) => f.write_str(n.text),
            Name::Numeric(c, n, _) => write!(f, "{c}%{n}"),
        }
    }
}

fn find_global_by_name<'a>(
    ctx: &'a ParsedHeph,
    global: Name<'_>,
) -> Result<(u32, &'a Global), AsmError> {
    Ok(ctx
        .globals
        .iter()
        .enumerate()
        .find_map(|(i, (name, g))| match global {
            Name::Named(n) => {
                if name == n.text {
                    Some((i as u32, g))
                } else {
                    None
                }
            }
            Name::Numeric(_, idx, _) => {
                if i == idx {
                    Some((i as u32, g))
                } else {
                    None
                }
            }
        })
        .ok_or_else(|| AsmError {
            tgt: match global {
                Name::Named(n) => n.span,
                Name::Numeric(_, _, span) => Some(span),
            },
            message: format!("global {} not found", global),
        })?)
}

// local can refer to either params or locals
// in the final instruction, these are combined, but here they
// are still separated and need to be adjusted
fn find_local_by_name(block: &Block, local: Name<'_>) -> Result<u32, AsmError> {
    Ok(block
        .locals
        .keys()
        .enumerate()
        .find(|v| match local {
            Name::Named(n) => v.1 == n.text,
            Name::Numeric(_, idx, _) => v.0 == idx,
        })
        .ok_or_else(|| AsmError {
            tgt: match local {
                Name::Named(n) => n.span,
                Name::Numeric(_, _, span) => Some(span),
            },
            message: format!("local {} not found", local),
        })?
        .0 as u32)
}

impl<'a> ParsedInstruction<'a> {
    fn parse(
        parsed: &ParsedHeph,
        locals: &[(&String, &ComplexType)],
        value: Pair<'a, Rule>,
    ) -> Result<Self, AsmError> {
        debug_assert_eq!(value.as_rule(), Rule::instruction);
        let mut value = value.into_inner();
        let op = value.next().unwrap();

        match op.as_rule() {
            Rule::OP_NOP => Ok(Self::Nop),
            Rule::OP_CONST => {
                let ty = value.next().unwrap();

                match ty.as_rule() {
                    Rule::int_ty => Ok(Self::ConstInt(ty.into(), value.next().unwrap().into())),
                    Rule::fp_ty => Ok(Self::ConstFp(ty.into(), value.next().unwrap().into())),
                    other => unreachable!("illegal type rule for const: {other:?}"),
                }
            }
            Rule::OP_GLOBAL_SET => Ok(Self::GlobalSet(
                value.next().unwrap().into(),
                value.next().unwrap().into(),
            )),
            Rule::OP_GLOBAL_GET => Ok(Self::GlobalGet(
                value.next().unwrap().into(),
                value.next().unwrap().into(),
            )),
            Rule::OP_GLOBAL_TEE => Ok(Self::GlobalTee(
                value.next().unwrap().into(),
                value.next().unwrap().into(),
            )),
            Rule::OP_LOCAL_SET => Ok(Self::LocalSet(
                value.next().unwrap().into(),
                value.next().unwrap().into(),
            )),
            Rule::OP_LOCAL_GET => Ok(Self::LocalGet(
                value.next().unwrap().into(),
                value.next().unwrap().into(),
            )),
            Rule::OP_LOCAL_TEE => Ok(Self::LocalTee(
                value.next().unwrap().into(),
                value.next().unwrap().into(),
            )),
            Rule::OP_ADD => Ok(Self::Add),
            Rule::OP_SUB => Ok(Self::Sub),
            Rule::OP_MUL => Ok(Self::Mul),
            Rule::OP_DIV => Ok(Self::Div),
            Rule::OP_REM => Ok(Self::Rem),
            Rule::OP_BOR => Ok(Self::Bor),
            Rule::OP_BAND => Ok(Self::Band),
            Rule::OP_BXOR => Ok(Self::Bxor),
            Rule::OP_INV => Ok(Self::Inv),
            Rule::OP_CAST => Ok(Self::Cast(
                value.next().unwrap().into(),
                value.next().unwrap().into(),
            )),
            Rule::OP_RETURN => Ok(Self::Return),
            Rule::OP_CALL => Ok(Self::Call(value.next().unwrap().into())),
            Rule::OP_CALL_DYNAMIC => Ok(Self::CallDynamic),
            Rule::OP_GET_FN_UPTR => Ok(Self::GetFnUPtr(value.next().unwrap().into())),
            Rule::OP_IF => {
                let block = parse_block(parsed, None, value.next().unwrap(), locals)?;

                if value.next().is_some_and(|r| r.as_rule() == Rule::OP_ELSE) {
                    let otherwise = parse_block(parsed, None, value.next().unwrap(), locals)?;
                    Ok(Self::IfElse(block, otherwise))
                } else {
                    Ok(Self::If(block))
                }
            }
            Rule::OP_LOOP => Ok(Self::Loop(parse_block(
                parsed,
                None,
                value.next().unwrap(),
                locals,
            )?)),
            Rule::OP_BREAK => Ok(Self::Break(value.next().map(Into::into))),
            Rule::OP_CONTINUE => Ok(Self::Continue(value.next().map(Into::into))),
            Rule::OP_MEM_ALLOC => Ok(Self::MemAlloc),
            Rule::OP_MEM_REALLOC => Ok(Self::MemRealloc),
            Rule::OP_MEM_FREE => Ok(Self::MemFree),
            Rule::OP_MEM_LOAD => Ok(Self::MemLoad(
                value.next().unwrap().into(),
                value.next().map(Into::into),
            )),
            Rule::OP_MEM_STORE => Ok(Self::MemStore(
                value.next().unwrap().into(),
                value.next().map(Into::into),
            )),
            Rule::OP_DISCARD => Ok(Self::Discard(value.next())),
            Rule::OP_CMP => Ok(Self::Cmp),
            Rule::OP_TEST_GT => Ok(Self::TestGt),
            Rule::OP_TEST_GEQ => Ok(Self::TestGeq),
            Rule::OP_TEST_LT => Ok(Self::TestLt),
            Rule::OP_TEST_LEQ => Ok(Self::TestLeq),
            Rule::OP_TEST_EQ => Ok(Self::TestEq),
            Rule::OP_TEST_NEQ => Ok(Self::TestNeq),
            Rule::OP_DUP => Ok(Self::Duplicate(value.next())),
            other => unreachable!("illegal instruction rule: {other:?}"),
        }
    }
}

fn parse_linkage(parsed: &ParsedHeph, pair: Pair<'_, Rule>) -> Result<Linkage, AsmError> {
    debug_assert!(pair.as_rule() == Rule::linkage || pair.as_rule() == Rule::type_linkage);
    let mut pairs = pair.into_inner();
    let mut still_taking_export_policy = true;
    let mut linkage = Linkage::Internal(InlinePref::Never);

    while let Some(pair) = pairs.next() {
        match pair.as_rule() {
            Rule::KW_EXPORT if still_taking_export_policy => {
                still_taking_export_policy = false;

                let default_flags =
                    LinkageFlags::DOCUMENTED | LinkageFlags::ACCESSIBLE | LinkageFlags::EXPORTED;

                if pairs
                    .peek()
                    .is_some_and(|v| v.as_rule() == Rule::export_mods)
                {
                    let flags = pairs
                        .next()
                        .unwrap()
                        .into_inner()
                        .fold(default_flags, |f, v| match v.as_rule() {
                            Rule::KW_ACCESSIBLE => f | LinkageFlags::ACCESSIBLE,
                            Rule::KW_NO_ACCESSIBLE => f & !LinkageFlags::ACCESSIBLE,
                            Rule::KW_DOCUMENT => f | LinkageFlags::DOCUMENTED,
                            Rule::KW_NO_DOCUMENT => f & !LinkageFlags::DOCUMENTED,
                            other => unreachable!("illegal linkage rule: {other:?}"),
                        });
                    linkage = Linkage::Export(flags)
                }
            }

            Rule::KW_INTERNAL if still_taking_export_policy => {
                still_taking_export_policy = false;

                if pairs.peek().is_some_and(|v| v.as_rule() != Rule::name) {
                    match pairs.next().unwrap().as_rule() {
                        Rule::KW_INLINE_ALWAYS => linkage = Linkage::Internal(InlinePref::Force),
                        Rule::KW_INLINE => linkage = Linkage::Internal(InlinePref::Prefer),
                        other => unreachable!("illegal inline preference rule: {other:?}"),
                    }
                }
            }

            Rule::name => {
                let (source_module, _) = find_global_by_name(parsed, Name::from(pair))?;
                let source_name = pairs.next().unwrap().as_str();
                linkage = Linkage::Import(Box::new(linkage), source_module, source_name.into());
            }

            Rule::KW_EXPECT => linkage = Linkage::Expected(Box::new(linkage)),

            _ => {}
        }
    }

    Ok(linkage)
}

pub trait ModuleResolver {
    fn open_module(&self, name: &str);
}

pub fn parse_text_asm(text: &str) -> Result<ParsedHeph, AsmError> {
    let mut rules = Grammar::parse(Rule::document, text)?
        .next()
        .unwrap()
        .into_inner();
    let header = rules.next().unwrap();

    debug_assert_eq!(
        header.as_rule(),
        Rule::header,
        "document must start with a header"
    );

    let version_rule = header.into_inner().next().unwrap();
    debug_assert_eq!(version_rule.as_rule(), Rule::version);

    // this will always succeed because the rule declares that version is always
    // exactly 3 numbers separated by dots, which is a valid semver.
    let version = semver::Version::from_str(version_rule.as_str()).unwrap();
    let this_version = semver::Version::from_str(env!("CARGO_PKG_VERSION")).unwrap();

    if version > this_version {
        return Err(AsmError::new_for(
            &version_rule,
            "this version of hephaestus is too old to parse this file",
        ));
    }

    let mut parsed = ParsedHeph::default();

    for pair in rules {
        match pair.as_rule() {
            Rule::global => {
                let mut pairs = pair.into_inner();
                let linkage = parse_linkage(&parsed, pairs.next().unwrap())?;

                let mutable = if pairs.peek().is_some_and(|r| r.as_rule() == Rule::MUTABLE) {
                    pairs.next();
                    true
                } else {
                    false
                };

                let name = pairs.next().unwrap();
                let ty = ComplexType::parse(&parsed, pairs.next().unwrap())?;
                let block = pairs.next();

                parsed.globals.insert(
                    name.as_str().into(),
                    Global::Var(Var {
                        ty: ty.clone(),
                        linkage,
                        mutable,
                        block: if let Some(block) = block {
                            Some(parse_block(&parsed, Some(ty), block, &[])?)
                        } else {
                            None
                        },
                    }),
                );
            }
            Rule::function => {
                let mut pairs = pair.into_inner();
                let linkage = parse_linkage(&parsed, pairs.next().unwrap())?;
                let name = pairs.next().unwrap();
                let (params, ty): (OrderedHashMap<String, ComplexType>, Option<ComplexType>) = {
                    let params_or_ty = pairs.next().unwrap();

                    let (params, ty) = if params_or_ty.as_rule() == Rule::parameters {
                        let params = params_or_ty.into_inner();
                        let ty = pairs.next().unwrap();

                        assert!(ty.as_rule() == Rule::ty || ty.as_rule() == Rule::VOID);

                        // parse params
                        (
                            Some(params.map(|v| parse_local_var(&parsed, v)).try_collect()?),
                            ty,
                        )
                    } else {
                        (None, params_or_ty)
                    };

                    (
                        params.unwrap_or_default(),
                        if ty.as_rule() == Rule::VOID {
                            None
                        } else {
                            Some(ComplexType::parse(&parsed, ty)?)
                        },
                    )
                };

                let has_import_linkage =
                    matches!(linkage, Linkage::Import(_, _, _) | Linkage::Expected(_));

                parsed.globals.insert(
                    name.as_str().into(),
                    Global::Function(Function {
                        linkage,
                        return_type: ty.clone(),
                        params: params.clone(),
                        block: None,
                    }),
                );

                let params_vec = params.iter().collect_vec();

                // this uses a roundabout way because the function body may expect the
                // function to be a part of the global table inside the block, e.g.
                // in recursive functions or functions that need to pass a refernce to
                // themselves to another place.
                //
                // the above adds a global with an empty block as a placeholder that
                // the block can refer to, and the below actually builds the block with
                // said context intact.
                if let Some(block_pair) = pairs.next() {
                    if has_import_linkage {
                        return Err(AsmError::new_for(
                            &block_pair,
                            "import declarations cannot have blocks",
                        ));
                    }

                    let block = parse_block(&parsed, ty, block_pair, &params_vec[..])?;
                    let Global::Function(f) = parsed
                        .globals
                        .get_mut(&String::from(name.as_str()))
                        .unwrap()
                    else {
                        unreachable!()
                    };
                    f.block = Some(block);
                } else if !has_import_linkage {
                    return Err(AsmError::new_for(
                        &name,
                        "this declaration is missing a block",
                    ));
                }
            }
            Rule::global_attribute => {
                let mut pairs = pair.into_inner();
                let path = pairs.next().unwrap();
                debug_assert_eq!(path.as_rule(), Rule::attribute_path);

                match path.as_str() {
                    "target" | "heph::target" => {
                        let target_name = pairs.next().ok_or_else(|| {
                            AsmError::new_for(
                                &path,
                                "this attribute requires a target to be specified",
                            )
                        })?;

                        let target = Target::from_str(target_name.as_str()).map_err(|e| {
                            AsmError::new_for(&target_name, &format!("invalid target name: {e}"))
                        })?;

                        parsed.target = target;
                    }

                    "metadata" | "heph::metadata" => {
                        let name = pairs
                            .next()
                            .ok_or_else(|| AsmError::new_for(&path, "name not specified"))?;
                        let ty = pairs
                            .next()
                            .ok_or_else(|| AsmError::new_for(&path, "type not specified"))?;
                        let content = pairs
                            .next()
                            .ok_or_else(|| AsmError::new_for(&path, "content not specified"))?;

                        let kind = match ty.as_str() {
                            "string" => MetadataContentKind::String,
                            "target" => MetadataContentKind::Target,
                            "bytes" => MetadataContentKind::Bytes,
                            #[cfg(feature = "cbor-features")]
                            "cbor" => MetadataContentKind::CBOR,
                            "dwarf" => continue,
                            other => todo!(),
                        };

                        let meta = MetadataDeclaration {
                            name: name.as_str().into(),
                            kind,
                            content: match kind {
                                MetadataContentKind::String => {
                                    Bytes::copy_from_slice(content.as_str().as_bytes())
                                }
                                MetadataContentKind::Bytes => {
                                    Bytes::from(hex::decode(content.as_str()).unwrap())
                                }
                                #[cfg(feature = "cbor-features")]
                                MetadataContentKind::CBOR => {
                                    let mut contents = vec![content.as_str()];

                                    for pair in pairs {
                                        contents.push(pair.as_str());
                                    }

                                    let content = contents.into_iter().join(",");

                                    use rustc_serialize::Encodable;

                                    let mut encoder = cbor::Encoder::from_memory();
                                    match &content[..] {
                                        "<nothing>" => {
                                            cbor::Cbor::Null.encode(&mut encoder).unwrap();
                                        }

                                        other => {
                                            rustc_serialize::json::Json::from_str(&other)
                                                .unwrap()
                                                .encode(&mut encoder)
                                                .unwrap();
                                        }
                                    }

                                    encoder.into_bytes().into()
                                }
                                MetadataContentKind::Target => {
                                    let mut bytes = BytesMut::new();
                                    bytes.put_u16(
                                        Target::from_str(content.as_str()).unwrap().into_u16(),
                                    );
                                    bytes.freeze()
                                }
                                MetadataContentKind::DWARF => unreachable!(),
                            },
                        };
                    }

                    "use_wit_components" | "heph::use_wit_components" => {
                        parsed.module_flags |= ModuleFlags::SUPPORTS_WIT_COMPONENTS;
                    }

                    "partial" | "heph::partial" => {
                        parsed.module_flags |= ModuleFlags::PARTIAL;
                    }

                    _ => {}
                }
            }
            Rule::import_module => {
                let pair = pair.into_inner().next().unwrap();

                match pair.as_rule() {
                    Rule::import_static => {
                        let mut pairs = pair.into_inner();
                        let mut module_name = pairs.next().unwrap().as_str();
                        module_name = &module_name[1..module_name.len() - 1]; // remove quotes
                        let name = pairs.next().unwrap();
                        parsed.globals.insert(
                            name.as_str().into(),
                            Global::ImportedModule(ImportedModule::StaticLibrary(
                                module_name.into(),
                            )),
                        );
                    }

                    Rule::import_dynamic => {
                        let mut pairs = pair.into_inner();
                        let mut module_name = pairs.next().unwrap().as_str();
                        module_name = &module_name[1..module_name.len() - 1]; // remove quotes
                        let name = pairs.next().unwrap();
                        parsed.globals.insert(
                            name.as_str().into(),
                            Global::ImportedModule(ImportedModule::DynamicLibrary(
                                module_name.into(),
                            )),
                        );
                    }

                    Rule::import_component => {
                        #[cfg(not(feature = "component-model"))]
                        {
                            return Err(AsmError::new_for(&pair, "the component model is not supported by this build of hephaestus, you must build hephaestus with the -Fcomponent-model option for this feature"));
                        }

                        #[cfg(feature = "component-model")]
                        {
                            // TODO ensure that WIT components are enabled
                            let orig = pair.clone();
                            let mut pairs = pair.into_inner();
                            let mut world_name = pairs.next().unwrap().as_str();
                            world_name = &world_name[1..world_name.len() - 1]; // remove quotes
                            let mut component_name = pairs.next().unwrap().as_str();
                            component_name = &component_name[1..component_name.len() - 1]; // remove quotes
                            let name = pairs.next().unwrap();

                            let wit = wit_component::decode_reader(
                                std::fs::File::open(component_name).map_err(|e| {
                                    AsmError::new_for(
                                        &orig,
                                        format!("failed to open WIT component file: {e}"),
                                    )
                                })?,
                            )
                            .map_err(|e| {
                                AsmError::new_for(
                                    &orig,
                                    format!("failed to decode WIT component: {e}"),
                                )
                            })?;

                            let _world = wit
                                .resolve()
                                .select_world(wit.package(), Some(world_name))
                                .map_err(|e| {
                                    AsmError::new_for(
                                        &orig,
                                        format!("could not find WIT world: {e}"),
                                    )
                                })?;

                            let encoded = wit_component::encode(wit.resolve(), wit.package())
                                .map_err(|e| {
                                    AsmError::new_for(
                                        &orig,
                                        format!("failed to re-encode WIT component: {e}"),
                                    )
                                })?;

                            parsed.globals.insert(
                                name.as_str().into(),
                                Global::ImportedModule(ImportedModule::WitComponent(
                                    world_name.into(),
                                    bytes::Bytes::from(encoded),
                                )),
                            );
                        }
                    }

                    other => unreachable!("illegal import module rule: {other:?}"),
                }
            }
            Rule::struct_definition => {
                let mut pairs = pair.into_inner();
                let linkage = parse_linkage(&parsed, pairs.next().unwrap())?;
                let name = pairs.next().unwrap();
                let fields = pairs
                    .map(|i| -> Result<StructField, AsmError> {
                        let (name, ty) = parse_local_var(&parsed, i)?;

                        // TODO empty space
                        Ok(StructField::Data {
                            layout: StructFieldLayout::Automatic, // TODO
                            name,
                            ty,
                        })
                    })
                    .try_collect()?;

                let struct_global = Struct {
                    layout: StructLayout::Hephaestus, // TODO
                    linkage,
                    values: fields,
                };

                parsed
                    .globals
                    .insert(name.as_str().into(), Global::Struct(struct_global));
            }
            Rule::enum_definition => {
                let mut pairs = pair.into_inner();
                let linkage = parse_linkage(&parsed, pairs.next().unwrap())?;
                let name = pairs.next().unwrap();
                let backing_type = Type::try_from(SpannedStr::from(pairs.next().unwrap()))?;
                let fields = pairs
                    .map(|i| -> Result<(String, Const), AsmError> {
                        let (name, value) = parse_enum_variant(backing_type, i)?;

                        // TODO empty space
                        Ok((name, value))
                    })
                    .try_collect()?;

                let enum_global = Enum {
                    backing_type,
                    linkage,
                    values: fields,
                };

                parsed
                    .globals
                    .insert(name.as_str().into(), Global::Enum(enum_global));
            }
            Rule::EOI => break,
            other => unreachable!("illegal declaration rule: {other:?}"),
        }
    }

    Ok(parsed)
}

fn parse_block(
    parsed: &ParsedHeph,
    return_type: Option<ComplexType>,
    block: Pair<Rule>,
    parent_locals: &[(&String, &ComplexType)],
) -> Result<Block, AsmError> {
    debug_assert_eq!(block.as_rule(), Rule::block);

    let mut pairs = block.into_inner().peekable();

    let block_locals = match pairs.peek().map(|v| v.as_rule()) {
        Some(Rule::local_var) => pairs
            .peeking_take_while(|v| v.as_rule() == Rule::local_var)
            .map(|v| parse_local_var(parsed, v))
            .try_collect::<_, Vec<_>, _>()?,
        Some(Rule::local_list) => pairs
            .next()
            .unwrap()
            .into_inner()
            .enumerate()
            .map(|(i, ty)| {
                ComplexType::parse(parsed, ty)
                    .map(|ty| (format!("@%{}", i + parent_locals.len()), ty))
            })
            .try_collect::<_, Vec<_>, _>()?,
        Some(_) | None => Vec::new(),
    };

    let locals: OrderedHashMap<String, ComplexType> = [
        parent_locals
            .iter()
            .map(|(a, b)| ((*a).clone(), (*b).clone()))
            .collect_vec(),
        block_locals,
    ]
    .into_iter()
    .flatten()
    .collect();

    let mut block = Block {
        inherited_locals: parent_locals.len(),
        locals,
        return_type,
        instructions: Vec::new(),
    };

    let local_vec = block.locals.iter().collect_vec();
    let instructions: Vec<MappedInstruction> = pairs
        .map(|r| ParsedInstruction::parse(parsed, &local_vec[..], r))
        .map(|i| i.and_then(|i| i.map_to_instruction(&parsed, &block)))
        .try_collect()?;

    for i in instructions {
        match i {
            MappedInstruction::Single(s) => block.instructions.push(s),
            MappedInstruction::Many(m) => block.instructions.extend(m),
        }
    }

    Ok(block)
}

fn parse_local_var(
    parsed: &ParsedHeph,
    v: Pair<'_, Rule>,
) -> Result<(String, ComplexType), AsmError> {
    debug_assert_eq!(v.as_rule(), Rule::local_var);

    let mut pairs = v.into_inner();
    let name = pairs.next().unwrap().as_str().to_string();
    let ty = ComplexType::parse(parsed, pairs.next().unwrap())?;
    Ok((name, ty))
}

fn parse_enum_variant(backing_type: Type, v: Pair<'_, Rule>) -> Result<(String, Const), AsmError> {
    debug_assert_eq!(v.as_rule(), Rule::enum_variant);

    let mut pairs = v.into_inner();
    let name = pairs.next().unwrap().as_str().to_string();
    let value = parse_int(pairs.next().unwrap());
    Ok((
        name,
        match backing_type {
            Type::U8 => Const::U8(value as u8),
            Type::U16 => Const::U16(value as u16),
            Type::U32 => Const::U32(value as u32),
            Type::U64 => Const::U64(value as u64),
            Type::I8 => Const::I8(value as i8),
            Type::I16 => Const::I16(value as i16),
            Type::I32 => Const::I32(value as i32),
            Type::I64 => Const::I64(value as i64),
            Type::F32 => Const::F32(value as f32),
            Type::F64 => Const::F64(value as f64),
            _ => unreachable!(),
        },
    ))
}

#[cfg(test)]
mod tests {
    use std::error::Error;
    use std::str::FromStr;

    use crate::asm::*;
    use crate::Type;

    use super::parse_text_asm;

    #[derive(Error, Debug)]
    #[error("assumption not met: {0}")]
    #[repr(transparent)]
    struct FailedAssumption(&'static str);

    #[test]
    fn old_version_permitted() {
        parse_text_asm("%hephaestus 0.0.9").unwrap();
    }

    #[test]
    fn this_version_permitted() {
        parse_text_asm(&format!("%hephaestus {}", env!("CARGO_PKG_VERSION"))).unwrap();
    }

    #[test]
    #[should_panic]
    fn new_version_fails() {
        parse_text_asm("%hephaestus 99999.0.0").unwrap();
    }

    #[test]
    fn type_parsing() {
        assert_eq!(Type::from_str("u8"), Ok(Type::U8));
        assert_eq!(Type::from_str("u16"), Ok(Type::U16));
        assert_eq!(Type::from_str("u32"), Ok(Type::U32));
        assert_eq!(Type::from_str("u64"), Ok(Type::U64));
        assert_eq!(Type::from_str("i8"), Ok(Type::I8));
        assert_eq!(Type::from_str("i16"), Ok(Type::I16));
        assert_eq!(Type::from_str("i32"), Ok(Type::I32));
        assert_eq!(Type::from_str("i64"), Ok(Type::I64));
        assert_eq!(Type::from_str("f32"), Ok(Type::F32));
        assert_eq!(Type::from_str("f64"), Ok(Type::F64));
        assert_eq!(Type::from_str("u_ptr"), Ok(Type::UPtr));
        assert_eq!(Type::from_str("i_ptr"), Ok(Type::IPtr));
        assert!(Type::from_str("void").is_err());
        assert!(Type::from_str("invalid").is_err());
    }

    #[test]
    fn function_parse() {
        const TEXT: &str = r#"
        %hephaestus 0.1.0

        funct @main($argc: u32, $argv: u_ptr) -> i32
        {
            local $test: i32;
            local $return_value: i32;

            const i32 1
            const u64 7
            cast u64 -> i32
            sub
            local.set i32 $test

            const i32 0
            local.set i32 $return_value

            local.get i32 $test
            local.get i32 $return_value
            add

            return
        }
        "#;

        let parsed = parse_text_asm(TEXT.trim()).unwrap();

        assert!(
            parsed.globals.len() == 1,
            "global length does not match expected"
        );
        assert!(
            matches!(&parsed.globals["@main"], Global::Function(_)),
            "global not a function"
        );

        let Global::Function(f) = &parsed.globals["@main"] else {
            unreachable!("main is not function?")
        };

        assert!(f.block.is_some());

        if let Some(block) = &f.block {
            assert_eq!(block.locals.len() - block.inherited_locals, 2);
            assert_eq!(block.locals["$test"], ComplexType::Primitive(Type::I32));
            assert_eq!(
                block.locals["$return_value"],
                ComplexType::Primitive(Type::I32)
            );
            let locals: [&str; 2] = block
                .locals
                .keys()
                .skip(block.inherited_locals)
                .map(|v| v.as_str())
                .collect_array()
                .unwrap();
            assert!(
                matches!(locals, ["$test", "$return_value"]),
                "order of locals was not preserved"
            );

            assert_eq!(block.return_type, f.return_type);
        } else {
            unreachable!()
        }

        assert_eq!(f.params.len(), 2);
        assert_eq!(f.params["$argc"], ComplexType::Primitive(Type::U32));
        assert_eq!(f.params["$argv"], ComplexType::Primitive(Type::UPtr));
        let locals: [&str; 2] = f.params.keys().map(|v| v.as_str()).collect_array().unwrap();
        assert!(
            matches!(locals, ["$argc", "$argv"]),
            "order of params was not preserved"
        );
    }

    #[test]
    fn complex_types_cannot_be_inline() {
        assert!(parse_text_asm(
            r#"
        %hephaestus 0.1.0

        internal inline struct @test {
            $test: i32
        }
        "#
            .trim()
        )
        .is_err());

        assert!(parse_text_asm(
            r#"
        %hephaestus 0.1.0

        internal inline(always) struct @test {
            $test: i32
        }
        "#
            .trim()
        )
        .is_err());

        assert!(parse_text_asm(
            r#"
        %hephaestus 0.1.0

        internal inline enum @test : u32 {
            $Test = 69,
            $Test2 = 420
        }
        "#
            .trim()
        )
        .is_err());

        assert!(parse_text_asm(
            r#"
        %hephaestus 0.1.0

        internal inline(always) enum @test : u32 {
            $Test = 69,
            $Test2 = 420
        }
        "#
            .trim()
        )
        .is_err());
    }

    #[test]
    fn enum_type_is_equal_to_backing_type() -> Result<(), Box<dyn Error>> {
        let parsed = parse_text_asm(
            r#"
        %hephaestus 0.1.0

        internal enum @test : u32 {
            $Test = 69,
            $Test2 = 420
        }

        internal funct @main($test_param: enum @test) -> void {
        }
        "#
            .trim(),
        )?;

        let Global::Enum(e) = find_global_by_name(&parsed, "@test".into())?.1 else {
            return Err(FailedAssumption("@test was not compiled to an enum").into());
        };
        let Global::Function(f) = find_global_by_name(&parsed, "@main".into())?.1 else {
            return Err(FailedAssumption("@main was not compiled to a function").into());
        };

        let param = f
            .params
            .get("$test_param")
            .ok_or(FailedAssumption("no $test_param"))?;

        assert_eq!(param.ty(), e.backing_type);

        Ok(())
    }
}
