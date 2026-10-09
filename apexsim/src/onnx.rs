//! A minimal ONNX reader and evaluator for the small graphs this project
//! exports.
//!
//! The plan asks for a trained policy to be exported to ONNX and for the Rust
//! side to produce the same results on identical observations. That needs an
//! ONNX runtime in Rust. The obvious choice is the `ort` crate, which ships a
//! full ONNX Runtime binary — roughly a hundred megabytes, a network fetch at
//! build time, and a second numerical stack whose behaviour on this machine
//! nobody here can audit.
//!
//! The graphs exported here are a three-layer perceptron: `Gemm` and `Tanh`,
//! plus whatever the exporter emits around them (`Constant` for folded
//! weights, `MatMul`/`Add` in newer export modes, `Relu`, `Identity`). That
//! is a small, fixed set, and an evaluator for it is a few hundred lines of
//! protobuf walking. It fails loudly on any op it does not implement rather
//! than guessing — a silently mis-evaluated graph is worse than no graph.
//!
//! Numerics are `f32` throughout, because that is what ONNX initializers are.
//! Parity against a reference runtime is therefore a *declared tolerance*
//! question, not an equality question, and `tests/phase12_gate.rs` measures
//! the actual worst-case difference rather than assuming one.
//!
//! Only the subset of `onnx.proto` actually needed is decoded. Field numbers
//! are from the ONNX specification and are cited beside each use.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Protobuf wire format — just enough to walk a ModelProto
// ---------------------------------------------------------------------------

const WIRE_VARINT: u8 = 0;
const WIRE_I64: u8 = 1;
const WIRE_LEN: u8 = 2;
const WIRE_I32: u8 = 5;

#[derive(Clone, Debug)]
enum Field<'a> {
    Varint(u64),
    /// Carried so the wire format is walked correctly; no field in the subset
    /// this reader decodes uses a 64-bit fixed field.
    #[allow(dead_code)]
    Fixed64(u64),
    Bytes(&'a [u8]),
    Fixed32(u32),
}

/// Walk a message's fields, calling `f(number, field)` for each. An error
/// from `f` stops the walk and is returned: protobuf has no other error
/// channel, and continuing past one would mean decoding half a message.
fn walk(mut b: &[u8], mut f: impl FnMut(u32, Field<'_>) -> Result<(), String>) -> Result<(), String> {
    while !b.is_empty() {
        let (key, rest) = read_varint(b)?;
        b = rest;
        let num = (key >> 3) as u32;
        let wt = (key & 7) as u8;
        let (field, rest) = match wt {
            WIRE_VARINT => {
                let (v, r) = read_varint(b)?;
                (Field::Varint(v), r)
            }
            WIRE_I64 => {
                if b.len() < 8 {
                    return Err("truncated 64-bit field".into());
                }
                (Field::Fixed64(u64::from_le_bytes(b[..8].try_into().unwrap())), &b[8..])
            }
            WIRE_LEN => {
                let (n, r) = read_varint(b)?;
                let n = n as usize;
                if r.len() < n {
                    return Err("truncated length-delimited field".into());
                }
                (Field::Bytes(&r[..n]), &r[n..])
            }
            WIRE_I32 => {
                if b.len() < 4 {
                    return Err("truncated 32-bit field".into());
                }
                (Field::Fixed32(u32::from_le_bytes(b[..4].try_into().unwrap())), &b[4..])
            }
            other => return Err(format!("unsupported protobuf wire type {other}")),
        };
        b = rest;
        f(num, field)?;
    }
    Ok(())
}

fn read_varint(b: &[u8]) -> Result<(u64, &[u8]), String> {
    let mut v: u64 = 0;
    let mut shift = 0u32;
    for (i, &byte) in b.iter().enumerate() {
        if shift >= 64 {
            return Err("varint overflows 64 bits".into());
        }
        v |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return Ok((v, &b[i + 1..]));
        }
        shift += 7;
    }
    Err("unterminated varint".into())
}

// ---------------------------------------------------------------------------
// Tensors
// ---------------------------------------------------------------------------

/// ONNX `TensorProto.DataType` values this reader accepts.
const DT_FLOAT: i64 = 1;
const DT_DOUBLE: i64 = 11;

/// A dense row-major tensor.
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    pub dims: Vec<usize>,
    pub data: Vec<f32>,
}

impl Tensor {
    pub fn rows(&self) -> Result<usize, String> {
        match self.dims.len() {
            1 => Ok(1),
            2 => Ok(self.dims[0]),
            n => Err(format!("expected a 1-D or 2-D tensor, got {n} dimensions")),
        }
    }

    pub fn cols(&self) -> Result<usize, String> {
        match self.dims.len() {
            1 => Ok(self.dims[0]),
            2 => Ok(self.dims[1]),
            n => Err(format!("expected a 1-D or 2-D tensor, got {n} dimensions")),
        }
    }

    /// Element-wise unary op.
    fn map(&self, f: impl Fn(f32) -> f32) -> Tensor {
        Tensor { dims: self.dims.clone(), data: self.data.iter().copied().map(f).collect() }
    }

    /// Element-wise op with ONNX numpy-style broadcasting of a scalar or a
    /// same-shaped tensor. Anything else is refused rather than truncated.
    fn zip(&self, other: &Tensor, name: &str, f: impl Fn(f32, f32) -> f32) -> Result<Tensor, String> {
        if self.dims == other.dims {
            let data = self.data.iter().zip(&other.data).map(|(a, b)| f(*a, *b)).collect();
            return Ok(Tensor { dims: self.dims.clone(), data });
        }
        if other.data.len() == 1 {
            let s = other.data[0];
            let data = self.data.iter().copied().map(|a| f(a, s)).collect();
            return Ok(Tensor { dims: self.dims.clone(), data });
        }
        // A 1-D operand broadcasts along the trailing axis -- the bias case an
        // exported Gemm produces ([n] against [1, n]).
        if other.dims.len() == 1 && self.dims.last() == Some(&other.dims[0]) {
            let w = other.dims[0];
            let data = self
                .data
                .iter()
                .enumerate()
                .map(|(i, a)| f(*a, other.data[i % w]))
                .collect();
            return Ok(Tensor { dims: self.dims.clone(), data });
        }
        // ...and the other way round, which is how a bias meets a column.
        if self.dims.len() == 1 && other.dims.last() == Some(&self.dims[0]) {
            let w = self.dims[0];
            let data = other
                .data
                .iter()
                .enumerate()
                .map(|(i, b)| f(self.data[i % w], *b))
                .collect();
            return Ok(Tensor { dims: other.dims.clone(), data });
        }
        Err(format!(
            "{name}: cannot broadcast {:?} against {:?}",
            self.dims, other.dims
        ))
    }

    /// Matrix multiply, requiring an exact shape match on the inner dimension.
    fn matmul(&self, b: &Tensor) -> Result<Tensor, String> {
        if self.dims.len() != 2 || b.dims.len() != 2 {
            return Err(format!(
                "MatMul needs 2-D operands, got {:?} and {:?}",
                self.dims, b.dims
            ));
        }
        let (m, k) = (self.dims[0], self.dims[1]);
        let (k2, n) = (b.dims[0], b.dims[1]);
        if k != k2 {
            return Err(format!("MatMul inner dimensions disagree: {k} vs {k2}"));
        }
        let mut data = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for t in 0..k {
                    acc += self.data[i * k + t] * b.data[t * n + j];
                }
                data[i * n + j] = acc;
            }
        }
        Ok(Tensor { dims: vec![m, n], data })
    }
}

// ---------------------------------------------------------------------------
// Graph
// ---------------------------------------------------------------------------

/// One graph node. Private because the fields are ONNX plumbing: callers
/// load a model and evaluate it, they do not rebuild one.
#[derive(Clone, Debug)]
pub struct Node {
    pub op_type: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    /// Attribute values this evaluator understands. Attributes it does not
    /// understand are an error at load time, not a silent default.
    alpha: f32,
    beta: f32,
    trans_a: bool,
    trans_b: bool,
    axis: i64,
    value: Option<Tensor>,
}

impl Node {
    fn parse(b: &[u8]) -> Result<Node, String> {
        let mut n = Node {
            op_type: String::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            alpha: 1.0,
            beta: 1.0,
            trans_a: false,
            trans_b: false,
            axis: -1,
            value: None,
        };
        walk(b, |num, f| {
            match (num, f) {
                (1, Field::Bytes(b)) => n.inputs.push(String::from_utf8_lossy(b).into_owned()),
                (2, Field::Bytes(b)) => n.outputs.push(String::from_utf8_lossy(b).into_owned()),
                (4, Field::Bytes(b)) => n.op_type = String::from_utf8_lossy(b).into_owned(),
                // AttributeProto.name = 1, .f = 2, .i = 3, .t = 5, .ints = 8
                (5, Field::Bytes(attr)) => n.attribute(attr)?,
                _ => {}
            }
            Ok(())
        })?;
        Ok(n)
    }

    fn attribute(&mut self, b: &[u8]) -> Result<(), String> {
        let mut name = String::new();
        let mut f_val: Option<f32> = None;
        let mut i_val: Option<i64> = None;
        let mut t_val: Option<Tensor> = None;
        walk(b, |num, f| {
            match (num, f) {
                (1, Field::Bytes(s)) => name = String::from_utf8_lossy(s).into_owned(),
                (2, Field::Fixed32(v)) => f_val = Some(f32::from_bits(v)),
                (3, Field::Varint(v)) => i_val = Some(v as i64),
                (5, Field::Bytes(t)) => t_val = Some(parse_tensor(t)?),
                _ => {}
            }
            Ok(())
        })?;
        match name.as_str() {
            "alpha" => self.alpha = f_val.ok_or("Gemm alpha is not a float")?,
            "beta" => self.beta = f_val.ok_or("Gemm beta is not a float")?,
            "transA" => self.trans_a = i_val.unwrap_or(0) != 0,
            "transB" => self.trans_b = i_val.unwrap_or(0) != 0,
            "axis" => self.axis = i_val.ok_or("axis is not an integer")?,
            "value" => self.value = t_val,
            // An attribute nobody here reads is a decision this evaluator is
            // not qualified to make silently.
            other => return Err(format!("attribute {other:?} is not implemented")),
        }
        Ok(())
    }
}

fn parse_tensor(b: &[u8]) -> Result<Tensor, String> {
    // TensorProto: dims = 1, data_type = 2, float_data = 4, raw_data = 9.
    let mut dims: Vec<usize> = Vec::new();
    let mut data_type = DT_FLOAT;
    let mut float_data: Vec<f32> = Vec::new();
    let mut raw: Vec<u8> = Vec::new();
    walk(b, |num, f| {
        match (num, f) {
            (1, Field::Varint(v)) => dims.push(v as usize),
            (2, Field::Varint(v)) => data_type = v as i64,
            (4, Field::Fixed32(v)) => float_data.push(f32::from_bits(v)),
            (9, Field::Bytes(r)) => raw = r.to_vec(),
            _ => {}
        }
        Ok(())
    })?;
    if data_type != DT_FLOAT && data_type != DT_DOUBLE {
        return Err(format!("tensor data type {data_type} is not float or double"));
    }
    let data = if !raw.is_empty() {
        if data_type == DT_DOUBLE {
            if !raw.len().is_multiple_of(8) {
                return Err("double raw_data is not a whole number of f64".into());
            }
            raw.as_chunks::<8>()
                .0
                .iter()
                .map(|c| f64::from_le_bytes(*c) as f32)
                .collect()
        } else {
            if !raw.len().is_multiple_of(4) {
                return Err("float raw_data is not a whole number of f32".into());
            }
            raw.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect()
        }
    } else {
        float_data
    };
    let expected: usize = dims.iter().product::<usize>().max(1);
    if data.len() != expected {
        return Err(format!(
            "tensor declares dims {dims:?} ({expected} values) but carries {}",
            data.len()
        ));
    }
    Ok(Tensor { dims, data })
}

/// A parsed ONNX graph restricted to the ops this project emits.
#[derive(Clone, Debug)]
pub struct OnnxModel {
    pub producer: String,
    pub opset: i64,
    pub nodes: Vec<Node>,
    pub initializers: HashMap<String, Tensor>,
    pub input: String,
    pub input_dim: usize,
    pub output: String,
    pub output_dim: usize,
}

/// Every op type the evaluator implements. Anything else is refused at load.
const SUPPORTED_OPS: &[&str] =
    &["Gemm", "MatMul", "Add", "Sub", "Mul", "Div", "Relu", "Tanh", "Sigmoid", "Identity", "Constant"];

impl OnnxModel {
    pub fn parse(bytes: &[u8]) -> Result<OnnxModel, String> {
        let mut graph: Vec<u8> = Vec::new();
        let mut producer = String::new();
        let mut opset = 0i64;
        walk(bytes, |num, f| {
            match (num, f) {
                (2, Field::Bytes(b)) => producer = String::from_utf8_lossy(b).into_owned(),
                // ModelProto.opset_import = 8; OperatorSetIdProto.version = 2
                (8, Field::Bytes(b)) => {
                    let mut v = 0i64;
                    walk(b, |n, ff| {
                        if let (2, Field::Varint(x)) = (n, ff) {
                            v = x as i64;
                        }
                        Ok(())
                    })?;
                    opset = opset.max(v);
                }
                (7, Field::Bytes(b)) => graph = b.to_vec(),
                _ => {}
            }
            Ok(())
        })?;
        if graph.is_empty() {
            return Err("not an ONNX model: no graph".into());
        }
        let graph: &[u8] = &graph;

        let mut nodes = Vec::new();
        let mut init_tensors: Vec<Tensor> = Vec::new();
        let mut inputs: Vec<String> = Vec::new();
        let mut outputs: Vec<String> = Vec::new();
        let mut input_dims: Vec<usize> = Vec::new();
        let mut output_dims: Vec<usize> = Vec::new();
        // GraphProto: node = 1, initializer = 5, input = 11, output = 12
        walk(graph, |num, f| {
            match (num, f) {
                (1, Field::Bytes(b)) => nodes.push(Node::parse(b)?),
                (5, Field::Bytes(b)) => init_tensors.push(parse_tensor(b)?),
                (11, Field::Bytes(b)) => {
                    let v = parse_value_info(b)?;
                    input_dims.push(v.width);
                    inputs.push(v.name);
                }
                (12, Field::Bytes(b)) => {
                    let v = parse_value_info(b)?;
                    output_dims.push(v.width);
                    outputs.push(v.name);
                }
                _ => {}
            }
            Ok(())
        })?;

        // A TensorProto carries no name, so the names have to come from
        // `graph.input`, where ONNX IR v4+ also requires every initializer to
        // be listed. PyTorch emits the free input(s) first and the
        // initializers as the trailing run, in the same order, which is what
        // ONNX Runtime accepts and what this pairs up. Some exporters drop the
        // initializers from `graph.input` entirely (PyTorch's unless
        // `keep_initializers_as_inputs` is set), which leaves the weights
        // genuinely nameless in the file; there is no fallback that could
        // recover them, so that case is reported rather than guessed at.
        if inputs.len() <= init_tensors.len() {
            return Err(format!(
                "graph lists {} inputs for {} initializers, so the weights cannot be named: \
                 a TensorProto does not carry its own name, and ONNX IR v4+ requires every \
                 initializer to also be listed in graph.input after the free inputs. Re-export \
                 with keep_initializers_as_inputs=true.",
                inputs.len(),
                init_tensors.len()
            ));
        }
        let free = inputs.len() - init_tensors.len();
        let mut initializers = HashMap::new();
        for (name, t) in inputs[free..].iter().zip(init_tensors.iter()) {
            initializers.insert(name.clone(), t.clone());
        }
        let inputs = inputs;

        for n in &nodes {
            if !SUPPORTED_OPS.contains(&n.op_type.as_str()) {
                return Err(format!(
                    "node op {:?} is not implemented; this evaluator supports {}",
                    n.op_type,
                    SUPPORTED_OPS.join(", ")
                ));
            }
            if n.inputs.is_empty() && n.op_type != "Constant" {
                return Err(format!("node {} has no inputs", n.op_type));
            }
            if n.op_type == "Constant" && n.value.is_none() {
                return Err("Constant node without a value attribute".into());
            }
        }

        // The real input is the first free input; the initializers were paired
        // from the trailing entries above.
        let input = inputs[0].clone();
        let output = outputs
            .into_iter()
            .next()
            .ok_or("graph declares no output")?;
        // Widths come from the declared shapes, which is the only place a
        // dynamic batch-free width appears. A symbolic dimension (dim_param,
        // e.g. "n") is refused: this evaluator runs one observation at a time,
        // so a model whose feature width is symbolic cannot be checked against
        // the environment's declared OBS_DIM anyway.
        let input_dim = input_dims
            .first()
            .ok_or_else(|| format!("input {input:?} declares no shape"))?;
        let output_dim = output_dims
            .first()
            .ok_or_else(|| format!("output {output:?} declares no shape"))?;

        Ok(OnnxModel {
            producer,
            opset,
            nodes,
            initializers,
            input,
            input_dim: *input_dim,
            output,
            output_dim: *output_dim,
        })
    }

    pub fn load(path: &std::path::Path) -> Result<OnnxModel, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        OnnxModel::parse(&bytes).map_err(|e| format!("{}: {e}", path.display()))
    }

    /// Evaluate the graph on one observation vector.
    pub fn run(&self, input: &[f32]) -> Result<Vec<f32>, String> {
        if input.len() != self.input_dim {
            return Err(format!(
                "observation has {} values but the graph wants {}",
                input.len(),
                self.input_dim
            ));
        }
        let mut vals: HashMap<&str, Tensor> = HashMap::new();
        for (name, t) in &self.initializers {
            vals.insert(name.as_str(), t.clone());
        }
        vals.insert(self.input.as_str(), Tensor { dims: vec![1, self.input_dim], data: input.to_vec() });

        for node in &self.nodes {
            let out = self.eval_node(node, &vals)?;
            if node.outputs.len() != 1 {
                return Err(format!("node {} must have exactly one output", node.op_type));
            }
            vals.insert(node.outputs[0].as_str(), out);
        }
        let out = vals
            .get(self.output.as_str())
            .ok_or_else(|| format!("graph never produced {:?}", self.output))?;
        if out.data.len() != self.output_dim {
            return Err(format!(
                "output {:?} has {} values, expected {}",
                self.output,
                out.data.len(),
                self.output_dim
            ));
        }
        Ok(out.data.clone())
    }

    fn get<'v>(&self, vals: &'v HashMap<&str, Tensor>, name: &str, node: &str) -> Result<&'v Tensor, String> {
        vals.get(name).ok_or_else(|| format!("{node}: input {name:?} is undefined"))
    }

    fn eval_node(&self, node: &Node, vals: &HashMap<&str, Tensor>) -> Result<Tensor, String> {
        let op = node.op_type.as_str();
        if op == "Constant" {
            return node
                .value
                .clone()
                .ok_or_else(|| "Constant without a value".to_string());
        }
        let a = self.get(vals, &node.inputs[0], op)?;
        Ok(match op {
            "Identity" => a.clone(),
            "Relu" => a.map(|x| x.max(0.0)),
            "Tanh" => a.map(|x| x.tanh()),
            "Sigmoid" => a.map(|x| 1.0 / (1.0 + (-x).exp())),
            "MatMul" => a.matmul(self.get(vals, &node.inputs[1], op)?)?,
            "Add" => a.zip(self.get(vals, &node.inputs[1], op)?, op, |x, y| x + y)?,
            "Sub" => a.zip(self.get(vals, &node.inputs[1], op)?, op, |x, y| x - y)?,
            "Mul" => a.zip(self.get(vals, &node.inputs[1], op)?, op, |x, y| x * y)?,
            "Div" => a.zip(self.get(vals, &node.inputs[1], op)?, op, |x, y| x / y)?,
            "Gemm" => {
                let b = self.get(vals, &node.inputs[1], op)?;
                if node.inputs.len() < 3 || node.beta == 0.0 {
                    gemm(a, b, node.alpha, node.trans_a, node.trans_b)?
                } else {
                    let c = self.get(vals, &node.inputs[2], op)?;
                    let sum = gemm(a, b, node.alpha, node.trans_a, node.trans_b)?;
                    sum.zip(c, op, |x, y| x + node.beta * y)?
                }
            }
            other => return Err(format!("{other} is not implemented")),
        })
    }
}

/// A `ValueInfoProto`: its name, and the feature width of the last dimension.
///
/// ValueInfoProto.name = 1, .type = 2; TypeProto.tensor_type = 1;
/// TypeProto.Tensor.elem_type = 1, .shape = 2; TensorShapeProto.dim = 1;
/// Dimension.dim_value = 1, .dim_param = 2.
fn parse_value_info(b: &[u8]) -> Result<ValueInfo, String> {
    let mut name = String::new();
    let mut type_bytes: Vec<u8> = Vec::new();
    walk(b, |n, f| {
        match (n, f) {
            (1, Field::Bytes(s)) => name = String::from_utf8_lossy(s).into_owned(),
            (2, Field::Bytes(t)) => type_bytes = t.to_vec(),
            _ => {}
        }
        Ok(())
    })?;

    // TypeProto.tensor_type = 1 -> TypeProto.Tensor
    let mut tensor_type: Vec<u8> = Vec::new();
    walk(&type_bytes, |n, f| {
        if let (1, Field::Bytes(t)) = (n, f) {
            tensor_type = t.to_vec();
        }
        Ok(())
    })?;

    // TypeProto.Tensor.shape = 2 -> TensorShapeProto; dim = 1 -> Dimension
    let mut shape: Vec<u8> = Vec::new();
    let mut elem_type = 0i64;
    walk(&tensor_type, |n, f| {
        match (n, f) {
            (1, Field::Varint(v)) => elem_type = v as i64,
            (2, Field::Bytes(s)) => shape = s.to_vec(),
            _ => {}
        }
        Ok(())
    })?;
    if elem_type != DT_FLOAT && elem_type != DT_DOUBLE {
        return Err(format!("value {name:?} has data type {elem_type}, not float or double"));
    }

    let mut dims: Vec<Option<usize>> = Vec::new();
    walk(&shape, |n, f| {
        if let (1, Field::Bytes(d)) = (n, f) {
            let mut value = 0i64;
            let mut symbolic = false;
            walk(d, |dn, df| {
                match (dn, df) {
                    (1, Field::Varint(v)) => value = v as i64,
                    (2, Field::Bytes(_)) => symbolic = true,
                    _ => {}
                }
                Ok(())
            })?;
            dims.push(if symbolic || value <= 0 { None } else { Some(value as usize) });
        }
        Ok(())
    })?;

    // The last dimension is the feature width; a leading batch dimension is
    // expected to be dynamic (this evaluator runs one row at a time).
    let width = dims
        .last()
        .copied()
        .flatten()
        .ok_or_else(|| format!("value {name:?} has a symbolic feature width {dims:?}"))?;
    Ok(ValueInfo { name, width })
}

struct ValueInfo {
    name: String,
    width: usize,
}

/// `Y = alpha * A' * B' + beta * C`, which is ONNX `Gemm`.
fn gemm(a: &Tensor, b: &Tensor, alpha: f32, trans_a: bool, trans_b: bool) -> Result<Tensor, String> {
    let at = if trans_a { transpose(a)? } else { a.clone() };
    let bt = if trans_b { transpose(b)? } else { b.clone() };
    let m = at.rows()?;
    let n = bt.cols()?;
    let k = at.cols()?;
    let mut data = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for t in 0..k {
                acc += at.data[i * k + t] * bt.data[t * n + j];
            }
            data[i * n + j] = alpha * acc;
        }
    }
    Ok(Tensor { dims: vec![m, n], data })
}

fn transpose(t: &Tensor) -> Result<Tensor, String> {
    if t.dims.len() != 2 {
        return Err(format!("transpose needs a 2-D tensor, got {:?}", t.dims));
    }
    let (r, c) = (t.dims[0], t.dims[1]);
    let mut data = vec![0.0f32; r * c];
    for i in 0..r {
        for j in 0..c {
            data[j * r + i] = t.data[i * c + j];
        }
    }
    Ok(Tensor { dims: vec![c, r], data })
}

/// A loaded model plus the counters a latency claim needs.
pub struct OnnxSession {
    pub model: OnnxModel,
    pub path: String,
    evaluations: u64,
}

impl OnnxSession {
    pub fn load(path: &std::path::Path) -> Result<OnnxSession, String> {
        Ok(OnnxSession {
            model: OnnxModel::load(path)?,
            path: path.display().to_string(),
            evaluations: 0,
        })
    }

    pub fn run(&mut self, input: &[f32]) -> Result<Vec<f32>, String> {
        let out = self.model.run(input)?;
        self.evaluations += 1;
        Ok(out)
    }

    pub fn evaluations(&self) -> u64 {
        self.evaluations
    }
}

/// The C ABI for the evaluator, so a Python trainer can run the *Rust*
/// implementation over the same observations it gives a reference runtime.
/// Parity between two implementations cannot be checked from inside one of
/// them.
pub mod ffi {
    use super::OnnxSession;
    use std::ffi::{c_char, c_float, c_int, CString};
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        LazyLock, Mutex,
    };

    thread_local! {
        static LAST_ERROR: std::cell::RefCell<Option<CString>> =
            const { std::cell::RefCell::new(None) };
    }

    fn set_error(msg: impl Into<String>) {
        let s = msg.into().replace('\0', " ");
        LAST_ERROR.with(|c| *c.borrow_mut() = CString::new(s).ok());
    }

    fn store(s: &str) -> *const c_char {
        let owned = CString::new(s.replace('\0', " ")).unwrap_or_default();
        LAST_ERROR.with(|c| {
            *c.borrow_mut() = Some(owned);
            c.borrow().as_ref().map(|s| s.as_ptr()).unwrap_or(std::ptr::null())
        })
    }

    // Sessions are owned by the registry rather than handed out as raw
    // pointers, so a bad handle id is an error message instead of undefined
    // behaviour. The mutex is what makes the mutable borrow inside `run` safe.
    static SESSIONS: LazyLock<Mutex<Vec<(usize, OnnxSession)>>> =
        LazyLock::new(|| Mutex::new(Vec::new()));
    static NEXT_ID: AtomicUsize = AtomicUsize::new(1);

    /// Load a model. Returns a handle id (never 0), or 0 on failure.
    ///
    /// # Safety
    ///
    /// path is read as a NUL-terminated UTF-8 string and must be valid
    /// for reads of that string.
    #[no_mangle]
    pub unsafe extern "C" fn f1_onnx_open(path: *const c_char) -> usize {
        if path.is_null() {
            set_error("null model path");
            return 0;
        }
        let p = unsafe {
            let mut len = 0usize;
            while *path.add(len) != 0 {
                len += 1;
            }
            match std::str::from_utf8(std::slice::from_raw_parts(path as *const u8, len)) {
                Ok(s) => s.to_string(),
                Err(e) => {
                    set_error(format!("model path is not UTF-8: {e}"));
                    return 0;
                }
            }
        };
        match OnnxSession::load(std::path::Path::new(&p)) {
            Ok(sess) => {
                let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
                match SESSIONS.lock() {
                    Ok(mut v) => {
                        v.push((id, sess));
                        id
                    }
                    Err(_) => {
                        set_error("session registry is poisoned");
                        0
                    }
                }
            }
            Err(e) => {
                set_error(e);
                0
            }
        }
    }

    /// Run one observation. Returns the number of values written, or -1.
    ///
    /// # Safety
    ///
    /// id must be a handle from [`f1_onnx_open`] that has not been
    /// closed, input must be valid for reads of the model's declared input
    /// width, and out must be valid for writes of the declared output width.
    #[no_mangle]
    pub unsafe extern "C" fn f1_onnx_run(
        id: usize,
        input: *const c_float,
        n: usize,
        out: *mut c_float,
        cap: usize,
    ) -> c_int {
        if input.is_null() || out.is_null() {
            set_error("null observation or output buffer");
            return -1;
        }
        let obs = unsafe { std::slice::from_raw_parts(input, n) };
        let mut guard = match SESSIONS.lock() {
            Ok(g) => g,
            Err(_) => {
                set_error("session registry is poisoned");
                return -1;
            }
        };
        let Some((_, sess)) = guard.iter_mut().find(|(sid, _)| *sid == id) else {
            set_error(format!("no ONNX session with handle {id}"));
            return -1;
        };
        match sess.run(obs) {
            Ok(v) => {
                if v.len() > cap {
                    set_error(format!("output needs {} values, caller gave {cap}", v.len()));
                    return -1;
                }
                unsafe { std::ptr::copy_nonoverlapping(v.as_ptr(), out, v.len()) };
                v.len() as c_int
            }
            Err(e) => {
                set_error(e);
                -1
            }
        }
    }

    /// The model's declared widths, opset and node ops.
    ///
    /// # Safety
    ///
    /// id must be a live handle. The returned pointer borrows the
    /// model and is valid only until the next call on this id.
    #[no_mangle]
    pub unsafe extern "C" fn f1_onnx_io_json(id: usize) -> *const c_char {
        let guard = match SESSIONS.lock() {
            Ok(g) => g,
            Err(_) => return std::ptr::null(),
        };
        let Some((_, sess)) = guard.iter().find(|(sid, _)| *sid == id) else {
            set_error(format!("no ONNX session with handle {id}"));
            return std::ptr::null();
        };
        let json = serde_json::json!({
            "input_dim": sess.model.input_dim,
            "output_dim": sess.model.output_dim,
            "opset": sess.model.opset,
            "producer": sess.model.producer,
            "nodes": sess.model.nodes.iter().map(|n| n.op_type.clone()).collect::<Vec<_>>(),
            "path": sess.path,
        });
        store(&serde_json::to_string(&json).unwrap_or_default())
    }

    #[no_mangle]
    pub extern "C" fn f1_onnx_close(id: usize) {
        if let Ok(mut guard) = SESSIONS.lock() {
            guard.retain(|(sid, _)| *sid != id);
        }
    }

    #[no_mangle]
    pub extern "C" fn f1_onnx_last_error() -> *const c_char {
        LAST_ERROR.with(|c| match &*c.borrow() {
            Some(s) => s.as_ptr(),
            None => std::ptr::null(),
        })
    }
}
