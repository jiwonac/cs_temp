use egg::*;

define_language! {
    pub enum TnsrLang {
        "input"  = Input([Id; 1]), // Input T name@T/F@dim1_dim2_...
        "Add"    = Add([Id; 2]), // T x T --> T
        "Mul"    = Mul([Id; 2]), // T x T --> T
        "smul"   = Smul([Id; 2]), // S x T --> T
        "MatMul" = MatMul([Id; 2]), // T x T --> T
        "noop"   = Noop([Id;2]), // No-op used to combine multiple outputs
        "Relu"   = Relu([Id;1]), // Relu activation T --> T
        "Exp"    = Exp([Id; 2]), // Elementwise power T x S --> T
        "Transpose"  = Transpose([Id;1]), // T --> T
        Num(i32),
        Var(Symbol),
    }
}

// Enum defines the different types allowed in TnsLang
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum DataKind {
    Name, // Input tensors
    Scalar, // Input scalars
    Tnsr, // Tensor
}

impl Default for DataKind {
    fn default() -> Self {
        DataKind::Name
    }
}

// Metadata struct for TensorAnalysis (additional info for an e-class)
#[derive(Debug, Clone)]
pub struct Metadata {
    pub dtype: DataKind, // The type of the e-class
    pub name: String, // Name of input if it is an input type
    pub dims: Vec<i32>,
    pub constant_foldable: bool,
}

pub struct TnsrAnalysis {

}

impl Default for TnsrAnalysis {
    fn default() -> Self { TnsrAnalysis {

    } }
}

// Metadata analysis
impl Analysis<TnsrLang> for TnsrAnalysis {
    type Data = Metadata;

    // Merges two metadata when two eclasses are merged
    fn merge(&mut self, to: &mut Self::Data, from: Self::Data) -> egg::DidMerge {
        if from.constant_foldable && (!to.constant_foldable) {
            to.constant_foldable = false;
            DidMerge(true, false)
        } else {
            DidMerge(false, false)
        }
    }

    fn make(egraph: &EGraph<TnsrLang, Self>, enode: &TnsrLang) -> Self::Data {
        let x = |i: &Id| &egraph[*i].data;
        // Parse name@T/F@dim1_dim2_... and return a tuple
        // (String, bool, Vec<i32>)
        let parse_input = |name: &Id| {
            let name_vec: Vec<&str> = x(name).name.split("@").collect();
            assert!(name_vec.len() == 3);
            let name: String = String::from(name_vec[0]);
            let is_constant: bool = name_vec[1] == "T";
            let dims: Vec<i32> = name_vec[2]
                .split("_")
                .map(|x| x.parse::<i32>().unwrap())
                .collect();
            (name, is_constant, dims)
        };

        match enode {
            TnsrLang::Input([name]) => {
                assert!(x(name).dtype == DataKind::Name);
                let parsed_input = parse_input(name);
                Self::Data {
                    dtype: DataKind::Tnsr,
                    name: String::new(),
                    dims: parsed_input.2,
                    constant_foldable: parsed_input.1,
                }
            },

            TnsrLang::Add([a, b]) => {
                assert!(x(a).dtype == DataKind::Tnsr);
                assert!(x(b).dtype == DataKind::Tnsr);
                assert!(&x(a).dims == &x(b).dims);
                let foldable = x(a).constant_foldable && x(b).constant_foldable;
                Self::Data {
                    dtype: DataKind::Tnsr,
                    name: String::new(),
                    dims: x(a).dims.clone(),
                    constant_foldable: foldable,
                }
            },

            TnsrLang::Mul([a, b]) => {
                assert!(x(a).dtype == DataKind::Tnsr);
                assert!(x(b).dtype == DataKind::Tnsr);
                assert!(&x(a).dims == &x(b).dims);
                let foldable = x(a).constant_foldable && x(b).constant_foldable;
                Self::Data {
                    dtype: DataKind::Tnsr,
                    name: String::new(),
                    dims: x(a).dims.clone(),
                    constant_foldable: foldable,
                }
            },

            TnsrLang::Smul([s, t]) => {
                assert!(x(s).dtype == DataKind::Scalar);
                assert!(x(t).dtype == DataKind::Tnsr);
                let foldable = x(s).constant_foldable && x(t).constant_foldable;
                Self::Data {
                    dtype: DataKind::Tnsr,
                    name: String::new(),
                    dims: x(t).dims.clone(),
                    constant_foldable: foldable,
                }
            },

            TnsrLang::MatMul([a, b]) => {
                assert!(x(a).dtype == DataKind::Tnsr);
                assert!(x(b).dtype == DataKind::Tnsr);
                assert_eq!(x(a).dims.len(), vec![0, 0].len());
                assert_eq!(x(b).dims.len(), vec![0, 0].len());
                assert_eq!(x(a).dims[1], x(b).dims[0]);
                //println!("{} {} {} {}", x(a).dims[0], x(a).dims[1], x(b).dims[0], x(b).dims[1]);
                let foldable = x(a).constant_foldable && x(b).constant_foldable;
                Self::Data {
                    dtype: DataKind::Tnsr,
                    name: String::new(),
                    dims: vec![x(a).dims[0], x(b).dims[1]],
                    constant_foldable: foldable,
                }
            },

            TnsrLang::Noop([a, b]) => {
                assert!(x(a).dtype == DataKind::Tnsr);
                assert!(x(b).dtype == DataKind::Tnsr);
                let foldable = x(a).constant_foldable && x(b).constant_foldable;
                Self::Data {
                    dtype: DataKind::Tnsr,
                    name: String::new(),
                    dims: vec![],
                    constant_foldable: foldable,
                }
            },

            TnsrLang::Relu([t]) => {
                assert!(x(t).dtype == DataKind::Tnsr);
                let foldable = x(t).constant_foldable;
                Self::Data {
                    dtype: DataKind::Tnsr,
                    name: String::new(),
                    dims: vec![],
                    constant_foldable: foldable,
                }
            },

            TnsrLang::Exp([t, s]) => {
                assert!(x(t).dtype == DataKind::Tnsr);
                assert!(x(s).dtype == DataKind::Tnsr);
                let foldable = x(t).constant_foldable && x(s).constant_foldable;
                Self::Data {
                    dtype: DataKind::Tnsr,
                    name: String::new(),
                    dims: vec![],
                    constant_foldable: foldable,
                }
            },

            TnsrLang::Transpose([t]) => {
                assert!(x(t).dtype == DataKind::Tnsr);
                let foldable = x(t).constant_foldable;
                Self::Data {
                    dtype: DataKind::Tnsr,
                    name: String::new(),
                    dims: vec![],
                    constant_foldable: foldable,
                }
            },

            TnsrLang::Num(_n) => Self::Data {
                dtype: DataKind::Scalar,
                name: String::new(),
                dims: vec![],
                constant_foldable: true,
            },

            TnsrLang::Var(_s) => Self::Data {
                dtype: DataKind::Name,
                name: _s.as_str().to_string(),
                dims: vec![],
                constant_foldable: true,
            },
        }
    }
}

pub fn rules<A: Analysis<TnsrLang>>() -> Vec<Rewrite<TnsrLang, A>> { vec![
    rewrite!("ewadd-associative"; "(Add ?x (Add ?y ?z))" => "(Add (Add ?x ?y) ?z)"),
    rewrite!("ewadd-commutative"; "(Add ?x ?y)" => "(Add ?y ?x)"),
    rewrite!("Mul-associative"; "(Mul ?x (Mul ?y ?z))" => "(Mul (Mul ?x ?y) ?z)"),
    rewrite!("Mul-commutative"; "(Mul ?x ?y)" => "(Mul ?y ?x)"),
    rewrite!("distributivity-0"; "(Mul (Add ?x ?y) ?z)" => "(Add (Mul ?x ?z) (Mul ?y ?z))"),
    rewrite!("smul-associative"; "(smul (smul ?x ?y) ?w)" => "(smul ?x  (smul ?y ?w))"),
    rewrite!("distributivity-1"; "(smul (Add ?x ?y) ?w)" => "(Add (smul ?x ?w)  (smul ?y ?w))"),
    rewrite!("operator-commutativity-0"; "(smul (Mul ?x ?y) ?w)" => "(Mul ?x  (smul ?y ?w))"),
    rewrite!("MatMul-is-associative"; "(MatMul ?x (MatMul ?y ?z))" => "(MatMul (MatMul ?x ?y) ?z)"),
    rewrite!("MatMul-is-linear-0"; "(smul (MatMul ?x ?y) ?w)" => "(MatMul ?x  (smul ?y ?w))"),
    rewrite!("MatMul-is-linear-1"; "(MatMul ?x (Add ?y ?z))" => "(Add (MatMul ?x ?y) (MatMul ?x ?z))"),
    rewrite!("-ewadd-associative"; "(Add (Add ?x ?y) ?z)" =>  "(Add ?x (Add ?y ?z))"),
    rewrite!("-Mul-associative"; "(Mul (Mul ?x ?y) ?z)" => "(Mul ?x (Mul ?y ?z))"),
    rewrite!("-distributivity-0"; "(Add (Mul ?x ?z) (Mul ?y ?z))" => "(Mul (Add ?x ?y) ?z)"),
    rewrite!("-smul-associative"; "(smul ?x  (smul ?y ?w))" => "(smul (smul ?x ?y) ?w)"),
    rewrite!("-distributivity-1"; "(Add (smul ?x ?w)  (smul ?y ?w))" => "(smul (Add ?x ?y) ?w)"),
    rewrite!("-operator-commutativity-0"; "(Mul ?x  (smul ?y ?w))" => "(smul (Mul ?x ?y) ?w)"),
    rewrite!("-MatMul-is-associative"; "(MatMul (MatMul ?x ?y) ?z)" => "(MatMul ?x (MatMul ?y ?z))"),
    rewrite!("-MatMul-is-linear-0"; "(MatMul ?x  (smul ?y ?w))" => "(smul (MatMul ?x ?y) ?w)"),
    rewrite!("-MatMul-is-linear-1"; "(Add (MatMul ?x ?y) (MatMul ?x ?z))" => "(MatMul ?x (Add ?y ?z))"),
    rewrite!("relu-transpose"; "(relu (transpose ?x))" => "(transpose (relu ?x))"),
    rewrite!("-relu-transpose"; "(transpose (relu ?x))" => "(relu (transpose ?x))"),
    rewrite!("transpose-is-its-own-inverse"; "(transpose (transpose ?x))" => "?x"),
    rewrite!("transpose-commutativity-ewadd"; "(transpose (Add ?x ?y))" => "(Add (transpose ?x)  (transpose ?y))"),
    rewrite!("-transpose-commutativity-ewadd"; "(Add (transpose ?x)  (transpose ?y))" => "(transpose (Add ?x ?y))"),
    rewrite!("transpose-commutativity-ewmul"; "(transpose (ewmul ?x ?y))" => "(ewmul (transpose ?x)  (transpose ?y))"),
    rewrite!("transpose-commutativity-ewmul"; "(ewmul (transpose ?x)  (transpose ?y))" => "(transpose (ewmul ?x ?y))"),
    rewrite!("transpose-commutativity-smul"; "(smul (transpose ?x) ?w)" => "(transpose (smul ?x ?w))"),
    rewrite!("-transpose-commutativity-smul"; "(transpose (smul ?x ?w))" => "(smul (transpose ?x) ?w)"),
    rewrite!("matmul-transpose"; "(transpose (matmul ?x ?y))" => "(matmul (transpose ?y)  (transpose ?x))"),
    rewrite!("exp-ewmul"; "(Exp (Mul ?x ?y) ?z)" => "(Mul (Exp ?x ?z) (Exp ?y ?z))"),
    rewrite!("-exp-ewmul"; "(Mul (Exp ?x ?z) (Exp ?y ?z))" => "(Exp (Mul ?x ?y) ?z)"),
    rewrite!("exp-transpose"; "(Exp (Transpose ?x) ?y)" => "(Transpose (Exp ?x ?y))"),
    rewrite!("-exp-transpose"; "(Transpose (Exp ?x ?y))" => "(Exp (Transpose ?x) ?y)"),
]}

pub struct TnsrCost<'a> {
    pub egraph: &'a EGraph<TnsrLang, TnsrAnalysis>,
}

impl CostFunction<TnsrLang> for TnsrCost<'_> {
    type Cost = f32;
    /// Getting total cost for the subtree rooted at enode. See egg::CostFunction
    /// trait for more information on interface.
    fn cost<C: FnMut(Id) -> Self::Cost>(&mut self, enode: &TnsrLang, mut costs: C) -> Self::Cost {
        let self_cost = get_op_cost(&*self.egraph, enode);
        enode.fold(self_cost, |sum, id| sum + costs(id))
    }
}

const COPY_COST: f32 = 0.1;
const ADDITION_COST: f32 = 1.0;
const MULTIPLY_COST: f32 = 6.0;
const EXP_COST: f32 = 10.0;
const RELU_COST: f32 = 1.0;

fn get_op_cost(egraph: &EGraph<TnsrLang, TnsrAnalysis>, enode: &TnsrLang) -> f32 {
    let x = |i: &Id| &egraph[*i].data;
    match enode {
        TnsrLang::Num(_)
        | TnsrLang::Var(_)
        | TnsrLang::Input(_)
        | TnsrLang::Noop(_) => 0.0,

        TnsrLang::Add([a, b]) => {
            assert!(x(a).dtype == DataKind::Tnsr);
            assert!(x(b).dtype == DataKind::Tnsr);
            if x(a).constant_foldable && x(b).constant_foldable {
                0.0
            } else {
                let product: i32 = x(a).dims.iter().product::<i32>();
                product as f32 * ADDITION_COST
            }
        }

        TnsrLang::Mul([a, b]) => {
            assert!(x(a).dtype == DataKind::Tnsr);
            assert!(x(b).dtype == DataKind::Tnsr);
            if x(a).constant_foldable && x(b).constant_foldable {
                0.0
            } else {
                let product: i32 = x(a).dims.iter().product::<i32>();
                product as f32 * MULTIPLY_COST
            }
        }

        TnsrLang::Smul([s, t]) => {
            assert!(x(s).dtype == DataKind::Scalar);
            assert!(x(t).dtype == DataKind::Tnsr);
            if x(s).constant_foldable && x(t).constant_foldable {
                0.0
            } else {
                let product: i32 = x(t).dims.iter().product::<i32>();
                product as f32 * MULTIPLY_COST
            }        
        }

        TnsrLang::MatMul([a, b]) => {
            assert!(x(a).dtype == DataKind::Tnsr);
            assert!(x(b).dtype == DataKind::Tnsr);
            if x(a).constant_foldable && x(b).constant_foldable {
                0.0
            } else {
                (x(a).dims[0] * x(a).dims[1] * x(b).dims[1]) as f32 * MULTIPLY_COST
            }
        }

        TnsrLang::Transpose([t]) => {
            assert!(x(t).dtype == DataKind::Tnsr);
            if x(t).constant_foldable {
                0.0
            } else {
                let product: i32 = x(t).dims.iter().product::<i32>();
                product as f32 * COPY_COST
            }
        }

        TnsrLang::Relu([t]) => {
            assert!(x(t).dtype == DataKind::Tnsr);
            if x(t).constant_foldable {
                0.0
            } else {
                let product: i32 = x(t).dims.iter().product::<i32>();
                product as f32 * RELU_COST
            }
        }

        TnsrLang::Exp([t, s]) => {
            assert!(x(t).dtype == DataKind::Tnsr);
            assert!(x(s).dtype == DataKind::Scalar);
            if x(t).constant_foldable && x(s).constant_foldable {
                0.0
            } else {
                let product: i32 = x(t).dims.iter().product::<i32>();
                product as f32 * EXP_COST
            }
        }
    }
}