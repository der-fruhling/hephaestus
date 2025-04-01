use std::{
    collections::{HashMap, HashSet},
    fmt::Display,
    ops::Index,
};

use thiserror::Error;

use crate::{
    ComplexType, Enum, Function, Global, Item, LinkageFlags, Module, ModuleFlags, Source, Struct,
    Target, CURRENT_VERSION,
};

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default /* required for map_globals */)]
pub struct Ref(u32, u32);

impl Display for Ref {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.0, self.1)
    }
}

impl Ref {
    pub fn new(module: u32, item: u32) -> Self {
        Self(module, item)
    }

    pub fn module(&self) -> usize {
        self.0 as usize
    }

    pub fn item(&self) -> usize {
        self.1 as usize
    }

    pub fn unwrap(self) -> (u32, u32) {
        (self.0, self.1)
    }
}

pub type RefModule = Module<Ref>;
pub type RefItem = Item<Ref>;
pub type RefSource = Source<Ref>;
pub type RefComplexType = ComplexType<Ref>;
pub type RefGlobal = Global<Ref>;
pub type RefFunction = Function<Ref>;
pub type RefStruct = Struct<Ref>;
pub type RefEnum = Enum<Ref>;

pub struct Linker {
    target: Target,
    modules: Vec<RefModule>,
    items: HashMap<String, Ref>,
}

pub trait ResolvableModule {
    fn resolve(self, index: u32) -> RefModule;
}

impl ResolvableModule for Module {
    fn resolve(self, index: u32) -> RefModule {
        self.map_globals(|g| Ref::new(index, g))
    }
}

impl ResolvableModule for RefModule {
    fn resolve(self, _index: u32) -> RefModule {
        self
    }
}

#[derive(Error, Debug)]
pub enum LinkError {
    #[error("mismatched target: module was for '{1}' but this linker demands '{0}'")]
    MismatchedTarget(Target, Target),
    #[error("malformed module: {0}")]
    MalformedModule(&'static str),
    #[error("duplicate symbol: {0}")]
    DuplicateSymbol(String),
    #[error("module with id {0} not found")]
    ModuleNotFound(u32),
    #[error("item with id {0} not found")]
    ItemNotFound(Ref),
}

#[derive(Default)]
struct ModuleCollector {
    refs: HashSet<Ref>,
    items: Vec<Item>,
    indices: HashMap<Ref, u32>,
}

impl ModuleCollector {
    fn collect(&mut self, linker: &Linker, r: Ref, item: &RefItem) -> Result<(), LinkError> {
        if self.refs.contains(&r) {
            // already collected
            return Ok(());
        }

        self.refs.insert(r);

        for global in item.needed_globals() {
            let item = linker.item(*global)?;
            self.collect(linker, *global, item)?;
        }

        let index = self.items.len() as u32;
        self.indices.insert(r, index);
        self.items
            .push(item.clone().map_globals(|g| self.indices[&g]));
        Ok(())
    }
}

impl Linker {
    pub fn new(target: Target) -> Self {
        Self {
            target,
            modules: vec![],
            items: Default::default(),
        }
    }

    pub fn put_module(&mut self, module: impl ResolvableModule) -> Result<(), LinkError> {
        let m = module.resolve(self.modules.len() as u32);

        if self.target != m.target {
            return Err(LinkError::MismatchedTarget(self.target, m.target));
        }

        let i = self.modules.len() as u32;

        for (j, g) in m.globals.iter().enumerate() {
            let Some(linkage) = g.linkage() else {
                continue;
            };

            if linkage.contains(LinkageFlags::EXPECTED) {
                continue;
            }

            if let Some(name) = g.name() {
                if self
                    .items
                    .insert(name.to_string(), Ref::new(i, j as u32))
                    .is_some()
                {
                    return Err(LinkError::DuplicateSymbol(name.to_string()));
                }
            }
        }

        self.modules.push(m);
        Ok(())
    }

    fn resolve(&self, v: Ref) -> Result<Ref, LinkError> {
        let t = self
            .modules
            .get(v.module())
            .ok_or_else(|| LinkError::ModuleNotFound(v.0))?
            .globals
            .get(v.item())
            .ok_or_else(|| LinkError::ItemNotFound(v))?;

        let Some(linkage) = t.linkage() else {
            return Ok(v);
        };

        if linkage.contains(LinkageFlags::EXPECTED) {
            let Some(name) = t.name() else {
                return Err(LinkError::MalformedModule("expected item is not named"));
            };

            if let Some(target) = self.items.get(name) {
                return Ok(*target);
            }
        }

        Ok(v)
    }

    fn resolve_expectations(&mut self) -> Result<(), LinkError> {
        let mut map = HashMap::<Ref, Ref>::new();

        for global in self.modules.iter().flat_map(|v| v.globals.iter()) {
            for v in global.needed_globals() {
                if let Some(t) = map.get(v) {
                    map.insert(*v, *t);
                } else {
                    map.insert(*v, self.resolve(*v)?);
                }
            }
        }

        for global in self.modules.iter_mut().flat_map(|v| v.globals.iter_mut()) {
            *global = global
                .clone()
                .map_globals(|v| map.get(&v).cloned().unwrap_or(v));
        }

        Ok(())
    }

    fn collect_module(self) -> Result<Module, LinkError> {
        let mut collector = ModuleCollector::default();

        for (i, global) in self.modules.iter().enumerate().flat_map(|(i, v)| {
            v.globals
                .iter()
                .enumerate()
                .map(move |(j, g)| (Ref::new(i as u32, j as u32), g))
        }) {
            if !collector.refs.contains(&i) {
                collector.collect(&self, i, global)?;
            }
        }

        let mut module = Module {
            version: CURRENT_VERSION,
            flags: ModuleFlags::empty(),
            target: self.target,
            metadata: vec![],
            globals: collector.items,
            imported_modules: Default::default(),
            functions: Default::default(),
            global_vars: Default::default(),
            structs: Default::default(),
            enums: Default::default(),
        };

        module.recalculate();
        Ok(module)
    }

    pub fn item(&self, r: Ref) -> Result<&RefItem, LinkError> {
        self.modules
            .get(r.module())
            .ok_or_else(|| LinkError::ModuleNotFound(r.0))?
            .globals
            .get(r.item())
            .ok_or_else(|| LinkError::ItemNotFound(r))
    }

    pub fn link(mut self) -> Result<Module, LinkError> {
        log::debug!("Linking {} modules", self.modules.len());

        self.resolve_expectations()?;
        let module = self.collect_module()?;

        Ok(module)
    }
}

impl Index<Ref> for Linker {
    type Output = RefItem;

    fn index(&self, index: Ref) -> &Self::Output {
        &self.modules[index.module() as usize].globals[index.item() as usize]
    }
}
