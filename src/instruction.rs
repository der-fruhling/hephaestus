use bytes::{Buf, BufMut, Bytes, BytesMut};
use std::fmt::{Debug, Display, Formatter, Write};
use itertools::Itertools;
use crate::{BinaryEncodable, Block, Const, DecodeError, Indent, Type};

#[derive(Clone)]
pub enum Instruction {
    Nop,
    Const(Const),
    SetGlobal(Type, u32),
    GetGlobal(Type, u32),
    SetLocal(Type, u32),
    GetLocal(Type, u32),
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    BitOr,
    BitAnd,
    BitXor,
    Inv,
    CmpOrd,
    TestGt,
    TestGtEq,
    TestLt,
    TestLtEq,
    TestEq,
    TestNeq,
    Cast(Type, Type),
    CastChangeSign,
    Return,
    Call(u32),
    CallDynamic,
    GetFnUPtr(u32),
    If(Block),
    IfElse(Block, Block),
    Loop(Block),
    Break(u8),
    Continue(u8),
    Alloc,
    Realloc,
    Free,
    Load(Type, i16),
    Store(Type, i16),
    Discard,
    Duplicate
}

#[allow(non_camel_case_types)]
#[derive(Copy, Clone, Debug)]
#[repr(u8)]
#[non_exhaustive]
pub enum Op {
    Nop,

    ConstU8,
    ConstU16,
    ConstU32,
    ConstU64,
    ConstI8,
    ConstI16,
    ConstI32,
    ConstI64,
    ConstF32,
    ConstF64,

    SetGlobalU8,
    SetGlobalU16,
    SetGlobalU32,
    SetGlobalU64,
    SetGlobalI8,
    SetGlobalI16,
    SetGlobalI32,
    SetGlobalI64,
    SetGlobalF32,
    SetGlobalF64,
    SetGlobalUptr,
    SetGlobalIptr,

    GetGlobalU8,
    GetGlobalU16,
    GetGlobalU32,
    GetGlobalU64,
    GetGlobalI8,
    GetGlobalI16,
    GetGlobalI32,
    GetGlobalI64,
    GetGlobalF32,
    GetGlobalF64,
    GetGlobalUptr,
    GetGlobalIptr,

    SetLocalU8,
    SetLocalU16,
    SetLocalU32,
    SetLocalU64,
    SetLocalI8,
    SetLocalI16,
    SetLocalI32,
    SetLocalI64,
    SetLocalF32,
    SetLocalF64,
    SetLocalUptr,
    SetLocalIptr,

    GetLocalU8,
    GetLocalU16,
    GetLocalU32,
    GetLocalU64,
    GetLocalI8,
    GetLocalI16,
    GetLocalI32,
    GetLocalI64,
    GetLocalF32,
    GetLocalF64,
    GetLocalUptr,
    GetLocalIptr,

    Add,
    Sub,
    Mul,
    Div,
    Rem,
    BitOr,
    BitAnd,
    BitXor,
    Inv,

    CastChangeSign,
    CastF32_F64,
    CastF64_F32,
    CastI64_Iptr,
    CastU64_Uptr,
    CastI32_Iptr,
    CastU32_Uptr,
    CastArbitrary,

    Alloc,
    Realloc,
    Free,

    LoadU8,
    LoadU16,
    LoadU32,
    LoadU64,
    LoadI8,
    LoadI16,
    LoadI32,
    LoadI64,
    LoadF32,
    LoadF64,
    LoadUptr,
    LoadIptr,

    StoreU8,
    StoreU16,
    StoreU32,
    StoreU64,
    StoreI8,
    StoreI16,
    StoreI32,
    StoreI64,
    StoreF32,
    StoreF64,
    StoreUptr,
    StoreIptr,

    Discard,

    Return,
    Call,
    CallDynamic,
    GetFnUPtr,
    If,
    Else,
    Loop,
    Break,
    BreakArbitrary,
    Continue,
    ContinueArbitrary,

    Cmp,
    TestGt,
    TestLt,
    TestGtEq,
    TestLtEq,
    TestEq,
    TestNeq,

    Duplicate,

    /// ends any block
    End = 0xff
}

impl Op {
    pub fn into_u8(self) -> u8 {
        self as u8
    }

    pub fn from_u8(val: u8) -> Self {
        unsafe { std::mem::transmute(val) }
    }
}

impl BinaryEncodable for Instruction {
    fn encode(&self, bytes: &mut BytesMut) {
        match self {
            Instruction::Nop => bytes.put_u8(Op::Nop.into_u8()),
            Instruction::Const(Const::U8(v)) => { bytes.put_u8(Op::ConstU8.into_u8()); bytes.put_u8(*v); },
            Instruction::Const(Const::U16(v)) => { bytes.put_u8(Op::ConstU16.into_u8()); bytes.put_u16(*v); },
            Instruction::Const(Const::U32(v)) => { bytes.put_u8(Op::ConstU32.into_u8()); bytes.put_u32(*v); },
            Instruction::Const(Const::U64(v)) => { bytes.put_u8(Op::ConstU64.into_u8()); bytes.put_u64(*v); },
            Instruction::Const(Const::I8(v)) => { bytes.put_u8(Op::ConstI8.into_u8()); bytes.put_i8(*v); },
            Instruction::Const(Const::I16(v)) => { bytes.put_u8(Op::ConstI16.into_u8()); bytes.put_i16(*v); },
            Instruction::Const(Const::I32(v)) => { bytes.put_u8(Op::ConstI32.into_u8()); bytes.put_i32(*v); },
            Instruction::Const(Const::I64(v)) => { bytes.put_u8(Op::ConstI64.into_u8()); bytes.put_i64(*v); },
            Instruction::Const(Const::F32(v)) => { bytes.put_u8(Op::ConstF32.into_u8()); bytes.put_f32(*v); },
            Instruction::Const(Const::F64(v)) => { bytes.put_u8(Op::ConstF64.into_u8()); bytes.put_f64(*v); },

            Instruction::SetGlobal(Type::U8, v) => { bytes.put_u8(Op::SetGlobalU8.into_u8()); bytes.put_u32(*v); },
            Instruction::SetGlobal(Type::U16, v) => { bytes.put_u8(Op::SetGlobalU16.into_u8()); bytes.put_u32(*v); },
            Instruction::SetGlobal(Type::U32, v) => { bytes.put_u8(Op::SetGlobalU32.into_u8()); bytes.put_u32(*v); },
            Instruction::SetGlobal(Type::U64, v) => { bytes.put_u8(Op::SetGlobalU64.into_u8()); bytes.put_u32(*v); },
            Instruction::SetGlobal(Type::I8, v) => { bytes.put_u8(Op::SetGlobalI8.into_u8()); bytes.put_u32(*v); },
            Instruction::SetGlobal(Type::I16, v) => { bytes.put_u8(Op::SetGlobalI16.into_u8()); bytes.put_u32(*v); },
            Instruction::SetGlobal(Type::I32, v) => { bytes.put_u8(Op::SetGlobalI32.into_u8()); bytes.put_u32(*v); },
            Instruction::SetGlobal(Type::I64, v) => { bytes.put_u8(Op::SetGlobalI64.into_u8()); bytes.put_u32(*v); },
            Instruction::SetGlobal(Type::F32, v) => { bytes.put_u8(Op::SetGlobalF32.into_u8()); bytes.put_u32(*v); },
            Instruction::SetGlobal(Type::F64, v) => { bytes.put_u8(Op::SetGlobalF64.into_u8()); bytes.put_u32(*v); },
            Instruction::SetGlobal(Type::UPtr, v) => { bytes.put_u8(Op::SetGlobalUptr.into_u8()); bytes.put_u32(*v); },
            Instruction::SetGlobal(Type::IPtr, v) => { bytes.put_u8(Op::SetGlobalIptr.into_u8()); bytes.put_u32(*v); },

            Instruction::GetGlobal(Type::U8, v) => { bytes.put_u8(Op::GetGlobalU8.into_u8()); bytes.put_u32(*v); },
            Instruction::GetGlobal(Type::U16, v) => { bytes.put_u8(Op::GetGlobalU16.into_u8()); bytes.put_u32(*v); },
            Instruction::GetGlobal(Type::U32, v) => { bytes.put_u8(Op::GetGlobalU32.into_u8()); bytes.put_u32(*v); },
            Instruction::GetGlobal(Type::U64, v) => { bytes.put_u8(Op::GetGlobalU64.into_u8()); bytes.put_u32(*v); },
            Instruction::GetGlobal(Type::I8, v) => { bytes.put_u8(Op::GetGlobalI8.into_u8()); bytes.put_u32(*v); },
            Instruction::GetGlobal(Type::I16, v) => { bytes.put_u8(Op::GetGlobalI16.into_u8()); bytes.put_u32(*v); },
            Instruction::GetGlobal(Type::I32, v) => { bytes.put_u8(Op::GetGlobalI32.into_u8()); bytes.put_u32(*v); },
            Instruction::GetGlobal(Type::I64, v) => { bytes.put_u8(Op::GetGlobalI64.into_u8()); bytes.put_u32(*v); },
            Instruction::GetGlobal(Type::F32, v) => { bytes.put_u8(Op::GetGlobalF32.into_u8()); bytes.put_u32(*v); },
            Instruction::GetGlobal(Type::F64, v) => { bytes.put_u8(Op::GetGlobalF64.into_u8()); bytes.put_u32(*v); },
            Instruction::GetGlobal(Type::UPtr, v) => { bytes.put_u8(Op::GetGlobalUptr.into_u8()); bytes.put_u32(*v); },
            Instruction::GetGlobal(Type::IPtr, v) => { bytes.put_u8(Op::GetGlobalIptr.into_u8()); bytes.put_u32(*v); },

            Instruction::SetLocal(Type::U8, v) => { bytes.put_u8(Op::SetLocalU8.into_u8()); bytes.put_u32(*v); },
            Instruction::SetLocal(Type::U16, v) => { bytes.put_u8(Op::SetLocalU16.into_u8()); bytes.put_u32(*v); },
            Instruction::SetLocal(Type::U32, v) => { bytes.put_u8(Op::SetLocalU32.into_u8()); bytes.put_u32(*v); },
            Instruction::SetLocal(Type::U64, v) => { bytes.put_u8(Op::SetLocalU64.into_u8()); bytes.put_u32(*v); },
            Instruction::SetLocal(Type::I8, v) => { bytes.put_u8(Op::SetLocalI8.into_u8()); bytes.put_u32(*v); },
            Instruction::SetLocal(Type::I16, v) => { bytes.put_u8(Op::SetLocalI16.into_u8()); bytes.put_u32(*v); },
            Instruction::SetLocal(Type::I32, v) => { bytes.put_u8(Op::SetLocalI32.into_u8()); bytes.put_u32(*v); },
            Instruction::SetLocal(Type::I64, v) => { bytes.put_u8(Op::SetLocalI64.into_u8()); bytes.put_u32(*v); },
            Instruction::SetLocal(Type::F32, v) => { bytes.put_u8(Op::SetLocalF32.into_u8()); bytes.put_u32(*v); },
            Instruction::SetLocal(Type::F64, v) => { bytes.put_u8(Op::SetLocalF64.into_u8()); bytes.put_u32(*v); },
            Instruction::SetLocal(Type::UPtr, v) => { bytes.put_u8(Op::SetLocalUptr.into_u8()); bytes.put_u32(*v); },
            Instruction::SetLocal(Type::IPtr, v) => { bytes.put_u8(Op::SetLocalIptr.into_u8()); bytes.put_u32(*v); },

            Instruction::GetLocal(Type::U8, v) => { bytes.put_u8(Op::GetLocalU8.into_u8()); bytes.put_u32(*v); },
            Instruction::GetLocal(Type::U16, v) => { bytes.put_u8(Op::GetLocalU16.into_u8()); bytes.put_u32(*v); },
            Instruction::GetLocal(Type::U32, v) => { bytes.put_u8(Op::GetLocalU32.into_u8()); bytes.put_u32(*v); },
            Instruction::GetLocal(Type::U64, v) => { bytes.put_u8(Op::GetLocalU64.into_u8()); bytes.put_u32(*v); },
            Instruction::GetLocal(Type::I8, v) => { bytes.put_u8(Op::GetLocalI8.into_u8()); bytes.put_u32(*v); },
            Instruction::GetLocal(Type::I16, v) => { bytes.put_u8(Op::GetLocalI16.into_u8()); bytes.put_u32(*v); },
            Instruction::GetLocal(Type::I32, v) => { bytes.put_u8(Op::GetLocalI32.into_u8()); bytes.put_u32(*v); },
            Instruction::GetLocal(Type::I64, v) => { bytes.put_u8(Op::GetLocalI64.into_u8()); bytes.put_u32(*v); },
            Instruction::GetLocal(Type::F32, v) => { bytes.put_u8(Op::GetLocalF32.into_u8()); bytes.put_u32(*v); },
            Instruction::GetLocal(Type::F64, v) => { bytes.put_u8(Op::GetLocalF64.into_u8()); bytes.put_u32(*v); },
            Instruction::GetLocal(Type::UPtr, v) => { bytes.put_u8(Op::GetLocalUptr.into_u8()); bytes.put_u32(*v); },
            Instruction::GetLocal(Type::IPtr, v) => { bytes.put_u8(Op::GetLocalIptr.into_u8()); bytes.put_u32(*v); },

            Instruction::Add => bytes.put_u8(Op::Add.into_u8()),
            Instruction::Sub => bytes.put_u8(Op::Sub.into_u8()),
            Instruction::Mul => bytes.put_u8(Op::Mul.into_u8()),
            Instruction::Div => bytes.put_u8(Op::Div.into_u8()),
            Instruction::Rem => bytes.put_u8(Op::Rem.into_u8()),
            Instruction::BitOr => bytes.put_u8(Op::BitOr.into_u8()),
            Instruction::BitAnd => bytes.put_u8(Op::BitAnd.into_u8()),
            Instruction::BitXor => bytes.put_u8(Op::BitXor.into_u8()),
            Instruction::Inv => bytes.put_u8(Op::Inv.into_u8()),

            Instruction::Cast(Type::U8, Type::I8) => bytes.put_u8(Op::CastChangeSign.into_u8()),
            Instruction::Cast(Type::U16, Type::I16) => bytes.put_u8(Op::CastChangeSign.into_u8()),
            Instruction::Cast(Type::U32, Type::I32) => bytes.put_u8(Op::CastChangeSign.into_u8()),
            Instruction::Cast(Type::U32, Type::UPtr) => bytes.put_u8(Op::CastU32_Uptr.into_u8()),
            Instruction::Cast(Type::U64, Type::I64) => bytes.put_u8(Op::CastChangeSign.into_u8()),
            Instruction::Cast(Type::U64, Type::UPtr) => bytes.put_u8(Op::CastU64_Uptr.into_u8()),
            Instruction::Cast(Type::I8, Type::U8) => bytes.put_u8(Op::CastChangeSign.into_u8()),
            Instruction::Cast(Type::I16, Type::U16) => bytes.put_u8(Op::CastChangeSign.into_u8()),
            Instruction::Cast(Type::I32, Type::U32) => bytes.put_u8(Op::CastChangeSign.into_u8()),
            Instruction::Cast(Type::I32, Type::IPtr) => bytes.put_u8(Op::CastI32_Iptr.into_u8()),
            Instruction::Cast(Type::I64, Type::U64) => bytes.put_u8(Op::CastChangeSign.into_u8()),
            Instruction::Cast(Type::I64, Type::IPtr) => bytes.put_u8(Op::CastI64_Iptr.into_u8()),
            Instruction::Cast(Type::F32, Type::F64) => bytes.put_u8(Op::CastF32_F64.into_u8()),
            Instruction::Cast(Type::F64, Type::F32) => bytes.put_u8(Op::CastF64_F32.into_u8()),
            Instruction::Cast(Type::IPtr, Type::UPtr) => bytes.put_u8(Op::CastChangeSign.into_u8()),
            Instruction::Cast(Type::UPtr, Type::IPtr) => bytes.put_u8(Op::CastChangeSign.into_u8()),
            Instruction::Cast(a, b) if a == b => { /* just don't encode anything */ },
            Instruction::Cast(a, b) => { bytes.put_u8(Op::CastArbitrary.into_u8()); bytes.put_u8((a.into_u8() << 4) | b.into_u8()); }
            Instruction::CastChangeSign => bytes.put_u8(Op::CastChangeSign.into_u8()),

            Instruction::Return => bytes.put_u8(Op::Return.into_u8()),
            Instruction::Call(f) => { bytes.put_u8(Op::Call.into_u8()); bytes.put_u32(*f); }
            Instruction::CallDynamic => bytes.put_u8(Op::CallDynamic.into_u8()),
            Instruction::GetFnUPtr(f) => { bytes.put_u8(Op::GetFnUPtr.into_u8()); bytes.put_u32(*f); }
            Instruction::If(block) => Self::encode_if(bytes, block),
            Instruction::IfElse(if_true, if_false) => Self::encode_if_else(bytes, if_true, if_false),
            Instruction::Loop(block) => Self::encode_loop(bytes, block),
            Instruction::Break(0) => bytes.put_u8(Op::Break.into_u8()),
            Instruction::Break(depth) => { bytes.put_u8(Op::BreakArbitrary.into_u8()); bytes.put_u8(*depth); }
            Instruction::Continue(0) => bytes.put_u8(Op::Continue.into_u8()),
            Instruction::Continue(depth) => { bytes.put_u8(Op::ContinueArbitrary.into_u8()); bytes.put_u8(*depth); }

            Instruction::Alloc => bytes.put_u8(Op::Alloc.into_u8()),
            Instruction::Realloc => bytes.put_u8(Op::Realloc.into_u8()),
            Instruction::Free => bytes.put_u8(Op::Free.into_u8()),

            Instruction::Load(Type::U8, v) => { bytes.put_u8(Op::LoadU8.into_u8()); bytes.put_i16(*v); },
            Instruction::Load(Type::U16, v) => { bytes.put_u8(Op::LoadU16.into_u8()); bytes.put_i16(*v); },
            Instruction::Load(Type::U32, v) => { bytes.put_u8(Op::LoadU32.into_u8()); bytes.put_i16(*v); },
            Instruction::Load(Type::U64, v) => { bytes.put_u8(Op::LoadU64.into_u8()); bytes.put_i16(*v); },
            Instruction::Load(Type::I8, v) => { bytes.put_u8(Op::LoadI8.into_u8()); bytes.put_i16(*v); },
            Instruction::Load(Type::I16, v) => { bytes.put_u8(Op::LoadI16.into_u8()); bytes.put_i16(*v); },
            Instruction::Load(Type::I32, v) => { bytes.put_u8(Op::LoadI32.into_u8()); bytes.put_i16(*v); },
            Instruction::Load(Type::I64, v) => { bytes.put_u8(Op::LoadI64.into_u8()); bytes.put_i16(*v); },
            Instruction::Load(Type::F32, v) => { bytes.put_u8(Op::LoadF32.into_u8()); bytes.put_i16(*v); },
            Instruction::Load(Type::F64, v) => { bytes.put_u8(Op::LoadF64.into_u8()); bytes.put_i16(*v); },
            Instruction::Load(Type::UPtr, v) => { bytes.put_u8(Op::LoadUptr.into_u8()); bytes.put_i16(*v); },
            Instruction::Load(Type::IPtr, v) => { bytes.put_u8(Op::LoadIptr.into_u8()); bytes.put_i16(*v); },

            Instruction::Store(Type::U8, v) => { bytes.put_u8(Op::StoreU8.into_u8()); bytes.put_i16(*v); },
            Instruction::Store(Type::U16, v) => { bytes.put_u8(Op::StoreU16.into_u8()); bytes.put_i16(*v); },
            Instruction::Store(Type::U32, v) => { bytes.put_u8(Op::StoreU32.into_u8()); bytes.put_i16(*v); },
            Instruction::Store(Type::U64, v) => { bytes.put_u8(Op::StoreU64.into_u8()); bytes.put_i16(*v); },
            Instruction::Store(Type::I8, v) => { bytes.put_u8(Op::StoreI8.into_u8()); bytes.put_i16(*v); },
            Instruction::Store(Type::I16, v) => { bytes.put_u8(Op::StoreI16.into_u8()); bytes.put_i16(*v); },
            Instruction::Store(Type::I32, v) => { bytes.put_u8(Op::StoreI32.into_u8()); bytes.put_i16(*v); },
            Instruction::Store(Type::I64, v) => { bytes.put_u8(Op::StoreI64.into_u8()); bytes.put_i16(*v); },
            Instruction::Store(Type::F32, v) => { bytes.put_u8(Op::StoreF32.into_u8()); bytes.put_i16(*v); },
            Instruction::Store(Type::F64, v) => { bytes.put_u8(Op::StoreF64.into_u8()); bytes.put_i16(*v); },
            Instruction::Store(Type::UPtr, v) => { bytes.put_u8(Op::StoreUptr.into_u8()); bytes.put_i16(*v); },
            Instruction::Store(Type::IPtr, v) => { bytes.put_u8(Op::StoreIptr.into_u8()); bytes.put_i16(*v); },

            Instruction::Discard => bytes.put_u8(Op::Discard.into_u8()),
            Instruction::Duplicate => bytes.put_u8(Op::Duplicate.into_u8()),

            Instruction::CmpOrd => bytes.put_u8(Op::Cmp.into_u8()),

            Instruction::TestGt => {}
            Instruction::TestGtEq => {}
            Instruction::TestLt => {}
            Instruction::TestLtEq => {}
            Instruction::TestEq => {}
            Instruction::TestNeq => {}
        }
    }

    fn decode(bytes: &mut Bytes) -> Result<Self, DecodeError> where Self: Sized {
        let op = Op::from_u8(bytes.get_u8());
        Ok(match op {
            Op::Nop => Self::Nop,

            Op::ConstU8 => Self::Const(Const::U8(bytes.get_u8())),
            Op::ConstU16 => Self::Const(Const::U16(bytes.get_u16())),
            Op::ConstU32 => Self::Const(Const::U32(bytes.get_u32())),
            Op::ConstU64 => Self::Const(Const::U64(bytes.get_u64())),
            Op::ConstI8 => Self::Const(Const::I8(bytes.get_i8())),
            Op::ConstI16 => Self::Const(Const::I16(bytes.get_i16())),
            Op::ConstI32 => Self::Const(Const::I32(bytes.get_i32())),
            Op::ConstI64 => Self::Const(Const::I64(bytes.get_i64())),
            Op::ConstF32 => Self::Const(Const::F32(bytes.get_f32())),
            Op::ConstF64 => Self::Const(Const::F64(bytes.get_f64())),

            Op::SetGlobalU8 => Self::SetGlobal(Type::U8, bytes.get_u32()),
            Op::SetGlobalU16 => Self::SetGlobal(Type::U16, bytes.get_u32()),
            Op::SetGlobalU32 => Self::SetGlobal(Type::U32, bytes.get_u32()),
            Op::SetGlobalU64 => Self::SetGlobal(Type::U64, bytes.get_u32()),
            Op::SetGlobalI8 => Self::SetGlobal(Type::I8, bytes.get_u32()),
            Op::SetGlobalI16 => Self::SetGlobal(Type::I16, bytes.get_u32()),
            Op::SetGlobalI32 => Self::SetGlobal(Type::I32, bytes.get_u32()),
            Op::SetGlobalI64 => Self::SetGlobal(Type::I64, bytes.get_u32()),
            Op::SetGlobalF32 => Self::SetGlobal(Type::F32, bytes.get_u32()),
            Op::SetGlobalF64 => Self::SetGlobal(Type::F64, bytes.get_u32()),
            Op::SetGlobalUptr => Self::SetGlobal(Type::UPtr, bytes.get_u32()),
            Op::SetGlobalIptr => Self::SetGlobal(Type::IPtr, bytes.get_u32()),

            Op::GetGlobalU8 => Self::GetGlobal(Type::U8, bytes.get_u32()),
            Op::GetGlobalU16 => Self::GetGlobal(Type::U16, bytes.get_u32()),
            Op::GetGlobalU32 => Self::GetGlobal(Type::U32, bytes.get_u32()),
            Op::GetGlobalU64 => Self::GetGlobal(Type::U64, bytes.get_u32()),
            Op::GetGlobalI8 => Self::GetGlobal(Type::I8, bytes.get_u32()),
            Op::GetGlobalI16 => Self::GetGlobal(Type::I16, bytes.get_u32()),
            Op::GetGlobalI32 => Self::GetGlobal(Type::I32, bytes.get_u32()),
            Op::GetGlobalI64 => Self::GetGlobal(Type::I64, bytes.get_u32()),
            Op::GetGlobalF32 => Self::GetGlobal(Type::F32, bytes.get_u32()),
            Op::GetGlobalF64 => Self::GetGlobal(Type::F64, bytes.get_u32()),
            Op::GetGlobalUptr => Self::GetGlobal(Type::UPtr, bytes.get_u32()),
            Op::GetGlobalIptr => Self::GetGlobal(Type::IPtr, bytes.get_u32()),

            Op::SetLocalU8 => Self::SetLocal(Type::U8, bytes.get_u32()),
            Op::SetLocalU16 => Self::SetLocal(Type::U16, bytes.get_u32()),
            Op::SetLocalU32 => Self::SetLocal(Type::U32, bytes.get_u32()),
            Op::SetLocalU64 => Self::SetLocal(Type::U64, bytes.get_u32()),
            Op::SetLocalI8 => Self::SetLocal(Type::I8, bytes.get_u32()),
            Op::SetLocalI16 => Self::SetLocal(Type::I16, bytes.get_u32()),
            Op::SetLocalI32 => Self::SetLocal(Type::I32, bytes.get_u32()),
            Op::SetLocalI64 => Self::SetLocal(Type::I64, bytes.get_u32()),
            Op::SetLocalF32 => Self::SetLocal(Type::F32, bytes.get_u32()),
            Op::SetLocalF64 => Self::SetLocal(Type::F64, bytes.get_u32()),
            Op::SetLocalUptr => Self::SetLocal(Type::UPtr, bytes.get_u32()),
            Op::SetLocalIptr => Self::SetLocal(Type::IPtr, bytes.get_u32()),

            Op::GetLocalU8 => Self::GetLocal(Type::U8, bytes.get_u32()),
            Op::GetLocalU16 => Self::GetLocal(Type::U16, bytes.get_u32()),
            Op::GetLocalU32 => Self::GetLocal(Type::U32, bytes.get_u32()),
            Op::GetLocalU64 => Self::GetLocal(Type::U64, bytes.get_u32()),
            Op::GetLocalI8 => Self::GetLocal(Type::I8, bytes.get_u32()),
            Op::GetLocalI16 => Self::GetLocal(Type::I16, bytes.get_u32()),
            Op::GetLocalI32 => Self::GetLocal(Type::I32, bytes.get_u32()),
            Op::GetLocalI64 => Self::GetLocal(Type::I64, bytes.get_u32()),
            Op::GetLocalF32 => Self::GetLocal(Type::F32, bytes.get_u32()),
            Op::GetLocalF64 => Self::GetLocal(Type::F64, bytes.get_u32()),
            Op::GetLocalUptr => Self::GetLocal(Type::UPtr, bytes.get_u32()),
            Op::GetLocalIptr => Self::GetLocal(Type::IPtr, bytes.get_u32()),

            Op::Add => Self::Add,
            Op::Sub => Self::Sub,
            Op::Mul => Self::Mul,
            Op::Div => Self::Div,
            Op::Rem => Self::Rem,
            Op::BitOr => Self::BitOr,
            Op::BitAnd => Self::BitAnd,
            Op::BitXor => Self::BitXor,
            Op::Inv => Self::Inv,

            Op::CastU32_Uptr => Self::Cast(Type::U32, Type::UPtr),
            Op::CastU64_Uptr => Self::Cast(Type::U64, Type::UPtr),
            Op::CastI32_Iptr => Self::Cast(Type::I32, Type::IPtr),
            Op::CastI64_Iptr => Self::Cast(Type::I64, Type::IPtr),
            Op::CastF32_F64 => Self::Cast(Type::F32, Type::F64),
            Op::CastF64_F32 => Self::Cast(Type::F64, Type::F32),
            Op::CastArbitrary => {
                let b = bytes.get_u8();
                Self::Cast(Type::from_u8((b >> 4) & 0xF), Type::from_u8(b & 0xF))
            }
            Op::CastChangeSign => Self::CastChangeSign,

            Op::Return => Self::Return,
            Op::Call => Self::Call(bytes.get_u32()),
            Op::CallDynamic => Self::CallDynamic,
            Op::GetFnUPtr => Self::GetFnUPtr(bytes.get_u32()),
            Op::If => Self::read_instruction_if_block(bytes)?,
            Op::Loop => Self::Loop(Self::read_instruction_block(bytes)?),
            Op::Break => Self::Break(0),
            Op::BreakArbitrary => Self::Break(bytes.get_u8()),
            Op::Continue => Self::Continue(0),
            Op::ContinueArbitrary => Self::Continue(bytes.get_u8()),

            Op::Alloc => Self::Alloc,
            Op::Realloc => Self::Realloc,
            Op::Free => Self::Free,

            Op::LoadU8 => Self::Load(Type::U8, bytes.get_i16()),
            Op::LoadU16 => Self::Load(Type::U16, bytes.get_i16()),
            Op::LoadU32 => Self::Load(Type::U32, bytes.get_i16()),
            Op::LoadU64 => Self::Load(Type::U64, bytes.get_i16()),
            Op::LoadI8 => Self::Load(Type::I8, bytes.get_i16()),
            Op::LoadI16 => Self::Load(Type::I16, bytes.get_i16()),
            Op::LoadI32 => Self::Load(Type::I32, bytes.get_i16()),
            Op::LoadI64 => Self::Load(Type::I64, bytes.get_i16()),
            Op::LoadF32 => Self::Load(Type::F32, bytes.get_i16()),
            Op::LoadF64 => Self::Load(Type::F64, bytes.get_i16()),
            Op::LoadUptr => Self::Load(Type::UPtr, bytes.get_i16()),
            Op::LoadIptr => Self::Load(Type::IPtr, bytes.get_i16()),

            Op::StoreU8 => Self::Store(Type::U8, bytes.get_i16()),
            Op::StoreU16 => Self::Store(Type::U16, bytes.get_i16()),
            Op::StoreU32 => Self::Store(Type::U32, bytes.get_i16()),
            Op::StoreU64 => Self::Store(Type::U64, bytes.get_i16()),
            Op::StoreI8 => Self::Store(Type::I8, bytes.get_i16()),
            Op::StoreI16 => Self::Store(Type::I16, bytes.get_i16()),
            Op::StoreI32 => Self::Store(Type::I32, bytes.get_i16()),
            Op::StoreI64 => Self::Store(Type::I64, bytes.get_i16()),
            Op::StoreF32 => Self::Store(Type::F32, bytes.get_i16()),
            Op::StoreF64 => Self::Store(Type::F64, bytes.get_i16()),
            Op::StoreUptr => Self::Store(Type::UPtr, bytes.get_i16()),
            Op::StoreIptr => Self::Store(Type::IPtr, bytes.get_i16()),

            Op::Discard => Self::Discard,
            Op::Duplicate => Self::Duplicate,

            Op::Cmp => Self::CmpOrd,
            Op::TestGt => Self::TestGt,
            Op::TestGtEq => Self::TestGtEq,
            Op::TestLt => Self::TestLt,
            Op::TestLtEq => Self::TestLtEq,
            Op::TestEq => Self::TestEq,
            Op::TestNeq => Self::TestNeq,

            Op::End => return Err(DecodeError::BlockEnd(Op::End)),
            Op::Else => return Err(DecodeError::BlockEnd(Op::Else)),
        })
    }
}

impl Instruction {
    fn read_instruction_block(bytes: &mut Bytes) -> Result<Block, DecodeError> {
        let mut instructions = vec![];
        let locals = (0..bytes.get_u16()).map(|_| Type::from_u8(bytes.get_u8())).collect_vec();

        loop {
            match Instruction::decode(bytes) {
                Ok(instruction) => instructions.push(instruction),
                Err(DecodeError::BlockEnd(Op::End)) => break,
                Err(e) => return Err(e)
            }
        }

        Ok(Block { locals, ops: instructions })
    }

    fn read_instruction_if_block(bytes: &mut Bytes) -> Result<Instruction, DecodeError> {
        let mut instructions = vec![];
        let locals = (0..bytes.get_u16()).map(|_| Type::from_u8(bytes.get_u8())).collect_vec();

        loop {
            match Instruction::decode(bytes) {
                Ok(instruction) => instructions.push(instruction),
                Err(DecodeError::BlockEnd(Op::End)) => break,
                Err(DecodeError::BlockEnd(Op::Else)) => return Ok(Self::IfElse(Block { locals, ops: instructions }, Self::read_instruction_block(bytes)?)),
                Err(e) => return Err(e)
            }
        }

        Ok(Self::If(Block { locals, ops: instructions }))
    }

    fn encode_if(bytes: &mut BytesMut, block: &Block) {
        bytes.put_u8(Op::If.into_u8());

        Self::put_locals(bytes, &block);

        for inst in &block.ops {
            inst.encode(bytes);
        }

        bytes.put_u8(Op::End.into_u8());
    }

    fn put_locals(bytes: &mut BytesMut, block: &&Block) {
        bytes.put_u16(block.locals.len() as u16);
        for local in &block.locals {
            bytes.put_u8(local.into_u8());
        }
    }

    fn encode_if_else(bytes: &mut BytesMut, if_true: &Block, if_false: &Block) {
        bytes.put_u8(Op::If.into_u8());

        Self::put_locals(bytes, &if_true);

        for inst in &if_true.ops {
            inst.encode(bytes);
        }

        bytes.put_u8(Op::Else.into_u8());

        Self::put_locals(bytes, &if_false);

        for inst in &if_false.ops {
            inst.encode(bytes);
        }

        bytes.put_u8(Op::End.into_u8());
    }

    fn encode_loop(bytes: &mut BytesMut, block: &Block) {
        bytes.put_u8(Op::Loop.into_u8());

        Self::put_locals(bytes, &block);

        for inst in &block.ops {
            inst.encode(bytes);
        }

        bytes.put_u8(Op::End.into_u8());
    }
}

impl Display for Instruction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Instruction::Nop => f.write_str("nop"),
            Instruction::Const(c) => write!(f, "const {} {}", c.ty(), c),
            Instruction::SetGlobal(ty, g) => write!(f, "global.set {} @%{}", ty, g),
            Instruction::GetGlobal(ty, g) => write!(f, "global.get {} @%{}", ty, g),
            Instruction::SetLocal(ty, l) => write!(f, "local.set {} $%{}", ty, l),
            Instruction::GetLocal(ty, l) => write!(f, "local.get {} $%{}", ty, l),
            Instruction::Add => f.write_str("add"),
            Instruction::Sub => f.write_str("sub"),
            Instruction::Mul => f.write_str("mul"),
            Instruction::Div => f.write_str("div"),
            Instruction::Rem => f.write_str("rem"),
            Instruction::BitOr => f.write_str("bor"),
            Instruction::BitAnd => f.write_str("band"),
            Instruction::BitXor => f.write_str("bxor"),
            Instruction::Inv => f.write_str("inv"),
            Instruction::Cast(a, b) => write!(f, "cast {} -> {}", a, b),
            Instruction::CastChangeSign => f.write_str("cast.chgsign # erased type cast"),
            Instruction::Return => f.write_str("return"),
            Instruction::Call(g) => write!(f, "call @%{}", g),
            Instruction::CallDynamic => f.write_str("call.dynamic"),
            Instruction::GetFnUPtr(g) => write!(f, "fnptr @%{}", g),
            Instruction::If(instructions) => write!(f, "if {instructions}"),
            Instruction::IfElse(a, b) => write!(f, "if {a} else {b}"),
            Instruction::Loop(instructions) => write!(f, "loop {instructions}"),
            Instruction::Break(0) => f.write_str("break"),
            Instruction::Break(depth) => write!(f, "break %{}", depth),
            Instruction::Continue(0) => f.write_str("continue"),
            Instruction::Continue(depth) => write!(f, "continue %{}", depth),
            Instruction::Alloc => f.write_str("mem.alloc"),
            Instruction::Realloc => f.write_str("mem.realloc"),
            Instruction::Free => f.write_str("mem.free"),
            Instruction::Load(ty, off) => write!(f, "mem.load {} {:+}", ty, off),
            Instruction::Store(ty, off) => write!(f, "mem.store {} {:+}", ty, off),
            Instruction::Discard => f.write_str("discard"),
            Instruction::Duplicate => f.write_str("dup"),
            Instruction::CmpOrd => f.write_str("cmp"),
            Instruction::TestGt => f.write_str("test.gt"),
            Instruction::TestGtEq => f.write_str("test.geq"),
            Instruction::TestLt => f.write_str("test.lt"),
            Instruction::TestLtEq => f.write_str("test.leq"),
            Instruction::TestEq => f.write_str("test.eq"),
            Instruction::TestNeq => f.write_str("test.neq"),
        }
    }
}

impl Debug for Instruction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Instruction::Nop => f.write_str("Nop"),
            Instruction::Const(c) => write!(f, "Const({:?} {:?})", c.ty(), c),
            Instruction::SetGlobal(ty, g) => write!(f, "SetGlobal({:?} {:?})", ty, g),
            Instruction::GetGlobal(ty, g) => write!(f, "GetGlobal({:?} {:?})", ty, g),
            Instruction::SetLocal(ty, l) => write!(f, "SetLocal({:?} {:?})", ty, l),
            Instruction::GetLocal(ty, l) => write!(f, "GetLocal({:?} {:?})", ty, l),
            Instruction::Add => f.write_str("Add"),
            Instruction::Sub => f.write_str("Sub"),
            Instruction::Mul => f.write_str("Mul"),
            Instruction::Div => f.write_str("Div"),
            Instruction::Rem => f.write_str("Rem"),
            Instruction::BitOr => f.write_str("BitOr"),
            Instruction::BitAnd => f.write_str("BitAnd"),
            Instruction::BitXor => f.write_str("BitXor"),
            Instruction::Inv => f.write_str("Inv"),
            Instruction::Cast(a, b) => write!(f, "Cast({:?} -> {:?})", a, b),
            Instruction::CastChangeSign => f.write_str("CastChangeSign"),
            Instruction::Return => f.write_str("Return"),
            Instruction::Call(g) => write!(f, "Call({:?})", g),
            Instruction::CallDynamic => f.write_str("CallDynamic"),
            Instruction::GetFnUPtr(g) => write!(f, "GetFnUptr({:?})", g),
            Instruction::If(instructions) => f.debug_struct("If")
                .field("instructions", instructions)
                .finish(),
            Instruction::IfElse(if_true, if_false) => f.debug_struct("IfElse")
                .field("if_true", if_true)
                .field("if_false", if_false)
                .finish(),
            Instruction::Loop(instructions) => f.debug_struct("Loop")
                .field("instructions", instructions)
                .finish(),
            Instruction::Break(depth) => write!(f, "Break(%{:?})", depth),
            Instruction::Continue(depth) => write!(f, "Continue(%{:?})", depth),
            Instruction::Alloc => f.write_str("Alloc"),
            Instruction::Realloc => f.write_str("Realloc"),
            Instruction::Free => f.write_str("Free"),
            Instruction::Load(ty, off) => write!(f, "Load({:?} offset: {:+?})", ty, off),
            Instruction::Store(ty, off) => write!(f, "Store({:?} offset: {:+?})", ty, off),
            Instruction::Discard => f.write_str("Discard"),
            Instruction::Duplicate => f.write_str("Duplicate"),
            Instruction::CmpOrd => f.write_str("Cmp"),
            Instruction::TestGt => f.write_str("TestGt"),
            Instruction::TestGtEq => f.write_str("TestGtEq"),
            Instruction::TestLt => f.write_str("TestLt"),
            Instruction::TestLtEq => f.write_str("TestLtEq"),
            Instruction::TestEq => f.write_str("TestEq"),
            Instruction::TestNeq => f.write_str("TestNeq"),
        }
    }
}
