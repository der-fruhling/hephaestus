use std::{borrow::Cow, fmt::Display, str::FromStr};
use std::vec::IntoIter;
use itertools::Itertools;
use ordered_hash_map::OrderedHashMap;
use pest::{error::LineColLocation, iterators::Pair, Parser};
use thiserror::Error;

use crate::{Const, Item, LinkageFlags, Module, Type};
use crate::instruction::Instruction;

#[derive(pest_derive::Parser)]
#[grammar = "heph.pest"]
struct Grammar;

#[derive(Clone, Copy, Debug)]
pub struct LineCol { pub line: usize, pub col: usize }

impl Display for LineCol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.line, self.col)
    }
}

#[derive(Clone, Debug)]
pub struct Span(pub LineCol, pub Option<LineCol>);

impl From<LineColLocation> for Span {
    fn from(value: LineColLocation) -> Self {
        match value {
            LineColLocation::Pos((line, col)) => Self(LineCol { line, col }, None),
            LineColLocation::Span((fl, fc), (tl, tc)) => Self(
                LineCol { line: fl, col: fc },
                Some(LineCol { line: tl, col: tc })
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
#[error("{tgt} : {message}")]
pub struct AsmError {
    pub message: String,
    pub tgt: Span
}

impl From<pest::error::Error<Rule>> for AsmError {
    fn from(value: pest::error::Error<Rule>) -> Self {
        Self {
            message: value.variant.to_string(),
            tgt: value.line_col.into()
        }
    }
}

impl AsmError {
    pub fn new_for(tgt: &Pair<'_, Rule>, msg: impl Into<String>) -> Self {
        let (line, col) = tgt.line_col();

        Self {
            tgt: Span(LineCol { line, col }, None),
            message: msg.into()
        }
    }
}

#[derive(Default, Debug)]
pub struct ParsedHeph {
    globals: OrderedHashMap<String, Global>
}

impl ParsedHeph {
    pub fn finish(self) -> Module {
        let mut module = Module::default();
        // TODO Module flags
        // TODO imported modules

        for (name, global) in self.globals {
            match global {
                Global::Var(var) => {
                    let global_item = crate::Global {
                        linkage: LinkageFlags::EXPORTED | LinkageFlags::ACCESSIBLE | LinkageFlags::DOCUMENTED,
                        name: Some(name),
                        ty: var.ty,
                        mutable: var.mutable,
                        initializer: var.block.map(collect_block)
                    };

                    module.globals.push(Item::Global(global_item));
                }

                Global::Function(funct) => {
                    let funct_item = crate::Function {
                        // TODO linkage
                        linkage: LinkageFlags::EXPORTED | LinkageFlags::ACCESSIBLE | LinkageFlags::DOCUMENTED,
                        name: Some(name),
                        return_type: funct.block.return_type,
                        params: funct.params,
                        block: Some(collect_block(funct.block))
                    };

                    module.globals.push(Item::Function(funct_item));
                }
            }
        }

        module.recalculate();
        module
    }
}

fn collect_block(block: Block) -> crate::Block {
    crate::Block {
        locals: block.locals.values().skip(block.inherited_locals).copied().collect(),
        ops: block.instructions
    }
}

#[derive(Debug)]
pub enum Global {
    Var(Var),
    Function(Function)
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
            other => return Err(TypeParseError(other.into()))
        })
    }
}

impl TryFrom<SpannedStr<'_>> for Type {
    type Error = AsmError;

    fn try_from(value: SpannedStr<'_>) -> Result<Self, Self::Error> {
        Type::from_str(value.text).map_err(|e| AsmError {
            tgt: value.span,
            message: e.to_string()
        })
    }
}

#[derive(Debug)]
pub struct Function {
    pub params: OrderedHashMap<String, Type>,
    pub block: Block,
}

#[derive(Debug)]
pub struct Var {
    pub ty: Type,
    pub mutable: bool,
    pub block: Option<Block>,
}

#[derive(Debug, Default)]
pub struct Block {
    pub inherited_locals: usize,
    // locals should be predictable
    pub locals: OrderedHashMap<String, Type>,
    pub return_type: Option<Type>,
    pub instructions: Vec<Instruction>
}

struct SpannedStr<'a> {
    pub text: &'a str,
    pub span: Span
}

impl<'a> From<Pair<'a, Rule>> for SpannedStr<'a> {
    fn from(value: Pair<'a, Rule>) -> Self {
        let (line, col) = value.line_col();

        Self {
            text: value.as_str(),
            span: Span(LineCol { line, col }, None)
        }
    }
}

enum ParsedInstruction<'a> {
    Nop,
    ConstInt(SpannedStr<'a>, Pair<'a, Rule>),
    ConstFp(SpannedStr<'a>, SpannedStr<'a>),
    GlobalSet(SpannedStr<'a>, SpannedStr<'a>),
    GlobalGet(SpannedStr<'a>, SpannedStr<'a>),
    GlobalTee(SpannedStr<'a>, SpannedStr<'a>),
    LocalSet(SpannedStr<'a>, SpannedStr<'a>),
    LocalGet(SpannedStr<'a>, SpannedStr<'a>),
    LocalTee(SpannedStr<'a>, SpannedStr<'a>),
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
    Call(SpannedStr<'a>),
    CallDynamic,
    GetFnUPtr(SpannedStr<'a>),
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
    Many(Box<[Instruction]>)
}

enum MappedInstructionIterator {
    Single(Option<Instruction>),
    Many(IntoIter<Instruction>)
}

impl Iterator for MappedInstructionIterator {
    type Item = Instruction;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            MappedInstructionIterator::Single(v) => v.take(),
            MappedInstructionIterator::Many(v) => v.next()
        }
    }
}

impl IntoIterator for MappedInstruction {
    type Item = Instruction;
    type IntoIter = MappedInstructionIterator;

    fn into_iter(self) -> Self::IntoIter {
        match self {
            MappedInstruction::Single(v) => MappedInstructionIterator::Single(Some(v)),
            MappedInstruction::Many(v) => MappedInstructionIterator::Many(v.into_vec().into_iter())
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
        },
        None => {
            i128::from_str_radix(unparsed, 10).unwrap()
        },
        Some(other) => unreachable!("illegal rule for int number specialization: {other:?}")
    }
}

impl<'a> ParsedInstruction<'a> {
    pub fn map_to_instruction(self, ctx: &ParsedHeph, block: &Block) -> Result<MappedInstruction, AsmError> {
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
            },
            ParsedInstruction::ConstFp(ty, number) => {
                let ty = Type::try_from(ty)?;
                let parsed_f64 = f64::from_str(number.text).map_err(|e| AsmError {
                    tgt: number.span,
                    message: e.to_string()
                })?;

                let c = match ty {
                    Type::F32 => Const::F32(parsed_f64 as f32),
                    Type::F64 => Const::F64(parsed_f64),
                    other => unreachable!("illegal type for fp const (this shouldve been caught earlier!): {other:?}")
                };

                Instruction::Const(c).into()
            },
            ParsedInstruction::GlobalSet(ty, global) => {
                let ty = Type::try_from(ty)?;
                let index = find_global_by_name(ctx, global)?;

                Instruction::SetGlobal(ty, index).into()
            },
            ParsedInstruction::GlobalGet(ty, global) => {
                let ty = Type::try_from(ty)?;
                let index = find_global_by_name(ctx, global)?;

                Instruction::GetGlobal(ty, index).into()
            },
            ParsedInstruction::GlobalTee(ty, global) => {
                let ty = Type::try_from(ty)?;
                let index = find_global_by_name(ctx, global)?;

                [Instruction::Duplicate, Instruction::SetGlobal(ty, index)].into()
            },
            ParsedInstruction::LocalSet(ty, local) => {
                let ty = Type::try_from(ty)?;
                let index = find_local_by_name(block, local)?;
                Instruction::SetLocal(ty, index).into()
            },
            ParsedInstruction::LocalGet(ty, local) => {
                let ty = Type::try_from(ty)?;
                let index = find_local_by_name(block, local)?;
                Instruction::GetLocal(ty, index).into()
            },
            ParsedInstruction::LocalTee(ty, local) => {
                let ty = Type::try_from(ty)?;
                let index = find_local_by_name(block, local)?;
                [Instruction::Duplicate, Instruction::SetLocal(ty, index)].into()
            },
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
            },
            ParsedInstruction::Return => Instruction::Return.into(),
            ParsedInstruction::Call(name) => Instruction::Call(Self::get_global_by_name(ctx, name)?).into(),
            ParsedInstruction::CallDynamic => Instruction::CallDynamic.into(),
            ParsedInstruction::GetFnUPtr(name) => Instruction::GetFnUPtr(Self::get_global_by_name(ctx, name)?).into(),
            ParsedInstruction::IfElse(if_true, if_false) => Instruction::IfElse(collect_block(if_true), collect_block(if_false)).into(),
            ParsedInstruction::If(blk) => Instruction::If(collect_block(blk)).into(),
            ParsedInstruction::Loop(blk) => Instruction::Loop(collect_block(blk)).into(),
            ParsedInstruction::Break(depth) => Instruction::Break(Self::parse_maybe(depth)?).into(),
            ParsedInstruction::Continue(depth) => Instruction::Continue(Self::parse_maybe(depth)?).into(),
            ParsedInstruction::MemAlloc => Instruction::Alloc.into(),
            ParsedInstruction::MemRealloc => Instruction::Realloc.into(),
            ParsedInstruction::MemFree => Instruction::Free.into(),
            ParsedInstruction::MemLoad(ty, off) => Instruction::Load(
                Type::try_from(ty)?,
                Self::parse_maybe(off)?
            ).into(),
            ParsedInstruction::MemStore(ty, off) => Instruction::Store(
                Type::try_from(ty)?,
                Self::parse_maybe(off)?
            ).into(),
            ParsedInstruction::Discard(None) => Instruction::Discard.into(),
            ParsedInstruction::Discard(Some(rep)) => {
                let parsed_int = parse_int(rep.clone());
                let rep: usize = parsed_int.try_into().map_err(|_| AsmError::new_for(&rep, "too many repetitions!"))?;

                vec![Instruction::Discard; rep].into()
            }
            ParsedInstruction::Duplicate(None) => Instruction::Duplicate.into(),
            ParsedInstruction::Duplicate(Some(rep)) => {
                let parsed_int = parse_int(rep.clone());
                let rep: usize = parsed_int.try_into().map_err(|_| AsmError::new_for(&rep, "too many repetitions!"))?;
                
                vec![Instruction::Duplicate; rep].into()
            }
            ParsedInstruction::Cmp => Instruction::CmpOrd.into(),
            ParsedInstruction::TestGt => Instruction::TestGt.into(),
            ParsedInstruction::TestGeq => Instruction::TestGtEq.into(),
            ParsedInstruction::TestLt => Instruction::TestLt.into(),
            ParsedInstruction::TestLeq => Instruction::TestLtEq.into(),
            ParsedInstruction::TestEq => Instruction::TestEq.into(),
            ParsedInstruction::TestNeq => Instruction::TestNeq.into()
        })
    }

    fn parse_maybe<T: FromStr + Default>(depth: Option<SpannedStr>) -> Result<T, AsmError>
    where
        T::Err: Display
    {
        depth.map(|depth| {
            T::from_str(depth.text).map_err(|e| AsmError {
                tgt: depth.span,
                message: e.to_string()
            })
        }).unwrap_or_else(|| Ok(T::default()))
    }

    fn get_global_by_name(ctx: &ParsedHeph, name: SpannedStr) -> Result<u32, AsmError> {
        Ok(ctx.globals.keys().enumerate()
            .find(|s| s.1 == name.text)
            .map(|v| v.0)
            .ok_or_else(|| AsmError {
                tgt: name.span,
                message: format!("unknown global {}; did you forget to define it above this usage?", name.text)
            })? as u32)
    }
}

fn find_global_by_name(ctx: &ParsedHeph, global: SpannedStr<'_>) -> Result<u32, AsmError> {
    Ok(ctx.globals.keys().enumerate()
        .find(|v| v.1 == global.text)
        .ok_or_else(|| AsmError {
            tgt: global.span,
            message: format!("global {} not found", global.text)
        })?.0 as u32)
}

// local can refer to either params or locals
// in the final instruction, these are combined, but here they
// are still separated and need to be adjusted
fn find_local_by_name(block: &Block, local: SpannedStr<'_>) -> Result<u32, AsmError> {
    Ok(block.locals.keys().enumerate()
        .find(|v| v.1 == local.text)
        .ok_or_else(|| AsmError {
            tgt: local.span,
            message: format!("local {} not found", local.text)
        })?.0 as u32)
}

impl<'a> ParsedInstruction<'a> {
    fn parse(parsed: &ParsedHeph, locals: &[(&String, &Type)], value: Pair<'a, Rule>) -> Result<Self, AsmError> {
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
                    other => unreachable!("illegal type rule for const: {other:?}")
                }
            },
            Rule::OP_GLOBAL_SET => Ok(Self::GlobalSet(value.next().unwrap().into(), value.next().unwrap().into())),
            Rule::OP_GLOBAL_GET => Ok(Self::GlobalGet(value.next().unwrap().into(), value.next().unwrap().into())),
            Rule::OP_GLOBAL_TEE => Ok(Self::GlobalTee(value.next().unwrap().into(), value.next().unwrap().into())),
            Rule::OP_LOCAL_SET => Ok(Self::LocalSet(value.next().unwrap().into(), value.next().unwrap().into())),
            Rule::OP_LOCAL_GET => Ok(Self::LocalGet(value.next().unwrap().into(), value.next().unwrap().into())),
            Rule::OP_LOCAL_TEE => Ok(Self::LocalTee(value.next().unwrap().into(), value.next().unwrap().into())),
            Rule::OP_ADD => Ok(Self::Add),
            Rule::OP_SUB => Ok(Self::Sub),
            Rule::OP_MUL => Ok(Self::Mul),
            Rule::OP_DIV => Ok(Self::Div),
            Rule::OP_REM => Ok(Self::Rem),
            Rule::OP_BOR => Ok(Self::Bor),
            Rule::OP_BAND => Ok(Self::Band),
            Rule::OP_BXOR => Ok(Self::Bxor),
            Rule::OP_INV => Ok(Self::Inv),
            Rule::OP_CAST => Ok(Self::Cast(value.next().unwrap().into(), value.next().unwrap().into())),
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
            Rule::OP_LOOP => Ok(Self::Loop(parse_block(parsed, None, value.next().unwrap(), locals)?)),
            Rule::OP_BREAK => Ok(Self::Break(value.next().map(Into::into))),
            Rule::OP_CONTINUE => Ok(Self::Continue(value.next().map(Into::into))),
            Rule::OP_MEM_ALLOC => Ok(Self::MemAlloc),
            Rule::OP_MEM_REALLOC => Ok(Self::MemRealloc),
            Rule::OP_MEM_FREE => Ok(Self::MemFree),
            Rule::OP_MEM_LOAD => Ok(Self::MemLoad(value.next().unwrap().into(), value.next().map(Into::into))),
            Rule::OP_MEM_STORE => Ok(Self::MemStore(value.next().unwrap().into(), value.next().map(Into::into))),
            Rule::OP_DISCARD => Ok(Self::Discard(value.next().map(Into::into))),
            Rule::OP_CMP => Ok(Self::Cmp),
            Rule::OP_TEST_GT => Ok(Self::TestGt),
            Rule::OP_TEST_GEQ => Ok(Self::TestGeq),
            Rule::OP_TEST_LT => Ok(Self::TestLt),
            Rule::OP_TEST_LEQ => Ok(Self::TestLeq),
            Rule::OP_TEST_EQ => Ok(Self::TestEq),
            Rule::OP_TEST_NEQ => Ok(Self::TestNeq),
            Rule::OP_DUP => Ok(Self::Duplicate(value.next().map(Into::into))),
            other => unreachable!("illegal instruction rule: {other:?}")
        }
    }
}

pub fn parse_text_asm(text: &str) -> Result<ParsedHeph, AsmError> {
    let mut rules = Grammar::parse(Rule::document, text)?.next().unwrap().into_inner();
    let header = rules.next().unwrap();

    debug_assert_eq!(header.as_rule(), Rule::header, "document must start with a header");

    let version_rule = header.into_inner().next().unwrap();
    debug_assert_eq!(version_rule.as_rule(), Rule::version);

    // this will always succeed because the rule declares that version is always
    // exactly 3 numbers separated by dots, which is a valid semver.
    let version = semver::Version::from_str(version_rule.as_str()).unwrap();
    let this_version = semver::Version::from_str(env!("CARGO_PKG_VERSION")).unwrap();

    if version > this_version {
        return Err(AsmError::new_for(&version_rule, "this version of hephaestus is too old to parse this file"));
    }

    let mut parsed = ParsedHeph::default();

    for pair in rules {
        match pair.as_rule() {
            Rule::global => {
                let mut pairs = pair.into_inner();
                
                let mutable = if pairs.peek().is_some_and(|r| r.as_rule() == Rule::MUTABLE) {
                    pairs.next();
                    true
                } else { false };
                
                let name = pairs.next().unwrap();
                let ty = Type::try_from(SpannedStr::from(pairs.next().unwrap()))?;
                let block = pairs.next();
                
                parsed.globals.insert(name.as_str().into(), Global::Var(Var {
                    ty,
                    mutable,
                    block: if let Some(block) = block {
                        Some(parse_block(&parsed, Some(ty), block, &[])?)
                    } else {
                        None
                    }
                }));
            }
            Rule::function => {
                let mut pairs = pair.into_inner();
                let name = pairs.next().unwrap();
                let (params, ty): (OrderedHashMap<String, Type>, Option<Type>) = {
                    let params_or_ty = pairs.next().unwrap();

                    let (params, ty) = if params_or_ty.as_rule() == Rule::parameters {
                        let params = params_or_ty.into_inner();
                        let ty = pairs.next().unwrap();

                        assert!(ty.as_rule() == Rule::ty || ty.as_rule() == Rule::VOID);

                        // parse params
                        (Some(params.map(parse_local_var).try_collect()?), ty)
                    } else {
                        (None, params_or_ty)
                    };

                    (params.unwrap_or_default(), if ty.as_rule() == Rule::VOID {
                        None
                    } else {
                        Some(Type::try_from(SpannedStr::from(ty))?)
                    })
                };

                parsed.globals.insert(name.as_str().into(), Global::Function(Function {
                    params: params.clone(),
                    block: Block::default()
                }));
                
                let params_vec = params.iter().collect_vec();
                
                // this uses a roundabout way because the function body may expect the
                // function to be a part of the global table inside the block, e.g.
                // in recursive functions or functions that need to pass a refernce to
                // themselves to another place.
                //
                // the above adds a global with an empty block as a placeholder that
                // the block can refer to, and the below actually builds the block with
                // said context intact.
                let block = parse_block(&parsed, ty, pairs.next().unwrap(), &params_vec[..])?;
                let Global::Function(f) = parsed.globals.get_mut(&String::from(name.as_str())).unwrap() else { unreachable!() };
                f.block = block;
            },
            Rule::EOI => break,
            other => unreachable!("illegal declaration rule: {other:?}")
        }
    }

    Ok(parsed)
}

fn parse_block(parsed: &ParsedHeph, return_type: Option<Type>, block: Pair<Rule>, parent_locals: &[(&String, &Type)]) -> Result<Block, AsmError> {
    debug_assert_eq!(block.as_rule(), Rule::block);
    
    let mut pairs = block.into_inner().peekable();

    let locals: OrderedHashMap<String, Type> = [
        parent_locals.iter().map(|(a, b)| ((*a).clone(), **b)).collect_vec(),
        pairs
            .peeking_take_while(|v| v.as_rule() == Rule::local_var)
            .map(parse_local_var)
            .try_collect::<_, Vec<_>, _>()?
    ].into_iter().flatten().collect();

    let mut block = Block {
        inherited_locals: parent_locals.len(),
        locals,
        return_type,
        instructions: Vec::new()
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

fn parse_local_var(v: Pair<'_, Rule>) -> Result<(String, Type), AsmError> {
    debug_assert_eq!(v.as_rule(), Rule::local_var);

    let mut pairs = v.into_inner();
    let name = pairs.next().unwrap().as_str().to_string();
    let ty = Type::try_from(SpannedStr::from(pairs.next().unwrap()));
    ty.map(|ty| (name, ty))
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use crate::asm::*;
    use crate::Type;

    use super::parse_text_asm;

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
        parse_text_asm(&format!("%hephaestus 99999.0.0")).unwrap();
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

        assert!(parsed.globals.len() == 1, "global length does not match expected");
        assert!(matches!(&parsed.globals["@main"], Global::Function(_)), "global not a function");

        let Global::Function(f) = &parsed.globals["@main"] else { unreachable!("main is not function?") };

        assert_eq!(f.block.locals.len() - f.block.inherited_locals, 2);
        assert_eq!(f.block.locals["$test"], Type::I32);
        assert_eq!(f.block.locals["$return_value"], Type::I32);
        let locals: [&str; 2] = f.block.locals.keys().skip(f.block.inherited_locals).map(|v| v.as_str()).collect_array().unwrap();
        assert!(matches!(locals, ["$test", "$return_value"]), "order of locals was not preserved");

        assert_eq!(f.params.len(), 2);
        assert_eq!(f.params["$argc"], Type::U32);
        assert_eq!(f.params["$argv"], Type::UPtr);
        let locals: [&str; 2] = f.params.keys().map(|v| v.as_str()).collect_array().unwrap();
        assert!(matches!(locals, ["$argc", "$argv"]), "order of params was not preserved");
    }
}
