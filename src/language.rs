use egg::*;

define_language! {
    pub enum TnsrLang {
        "input"  = Input([Id; 1]), // Input T name@T/F@dim1_dim2_...
        "ewadd"  = Ewadd([Id; 2]), // T x T --> T
        "ewmul"  = Ewmul([Id; 2]), // T x T --> T
        "smul"   = Smul([Id; 2]), // S x T --> T 
        "matmul" = Matmul([Id; 2]), // T x T --> T
        "noop"   = Noop([Id;2]), // No-op used to combine multiple outputs
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

            TnsrLang::Ewadd([a, b]) => {
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

            TnsrLang::Ewmul([a, b]) => {
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

            TnsrLang::Matmul([a, b]) => {
                assert!(x(a).dtype == DataKind::Tnsr);
                assert!(x(b).dtype == DataKind::Tnsr);
                assert_eq!(x(a).dims.len(), vec![0, 0].len());
                assert_eq!(x(b).dims.len(), vec![0, 0].len());
                assert_eq!(x(a).dims[1], x(a).dims[0]);
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
    rewrite!("ewadd-associative"; "(ewadd ?x (ewadd ?y ?z))" => "(ewadd (ewadd ?x ?y) ?z)"),
    rewrite!("ewadd-commutative"; "(ewadd ?x ?y)" => "(ewadd ?y ?x)"),
    rewrite!("ewmul-associative"; "(ewmul ?x (ewmul ?y ?z))" => "(ewmul (ewmul ?x ?y) ?z)"),
    rewrite!("ewmul-commutative"; "(ewmul ?x ?y)" => "(ewmul ?y ?x)"),
    rewrite!("distributivity-0"; "(ewmul (ewadd ?x ?y) ?z)" => "(ewadd (ewmul ?x ?z) (ewmul ?y ?z))"),
    rewrite!("smul-associative"; "(smul (smul ?x ?y) ?w)" => "(smul ?x  (smul ?y ?w))"),
    rewrite!("distributivity-1"; "(smul (ewadd ?x ?y) ?w)" => "(ewadd (smul ?x ?w)  (smul ?y ?w))"),
    rewrite!("operator-commutativity-0"; "(smul (ewmul ?x ?y) ?w)" => "(ewmul ?x  (smul ?y ?w))"),
    rewrite!("matmul-is-associative"; "(matmul ?x (matmul ?y ?z))" => "(matmul (matmul ?x ?y) ?z)"),
    rewrite!("matmul-is-linear-0"; "(smul (matmul ?x ?y) ?w)" => "(matmul ?x  (smul ?y ?w))"),
    rewrite!("matmul-is-linear-1"; "(matmul ?x (ewadd ?y ?z))" => "(ewadd (matmul ?x ?y) (matmul ?x ?z))"),
    rewrite!("-ewadd-associative"; "(ewadd (ewadd ?x ?y) ?z)" =>  "(ewadd ?x (ewadd ?y ?z))"),
    rewrite!("-ewmul-associative"; "(ewmul (ewmul ?x ?y) ?z)" => "(ewmul ?x (ewmul ?y ?z))"),
    rewrite!("-distributivity-0"; "(ewadd (ewmul ?x ?z) (ewmul ?y ?z))" => "(ewmul (ewadd ?x ?y) ?z)"),
    rewrite!("-smul-associative"; "(smul ?x  (smul ?y ?w))" => "(smul (smul ?x ?y) ?w)"),
    rewrite!("-distributivity-1"; "(ewadd (smul ?x ?w)  (smul ?y ?w))" => "(smul (ewadd ?x ?y) ?w)"),
    rewrite!("-operator-commutativity-0"; "(ewmul ?x  (smul ?y ?w))" => "(smul (ewmul ?x ?y) ?w)"),
    rewrite!("-matmul-is-associative"; "(matmul (matmul ?x ?y) ?z)" => "(matmul ?x (matmul ?y ?z))"),
    rewrite!("-matmul-is-linear-0"; "(matmul ?x  (smul ?y ?w))" => "(smul (matmul ?x ?y) ?w)"),
    rewrite!("-matmul-is-linear-1"; "(ewadd (matmul ?x ?y) (matmul ?x ?z))" => "(matmul ?x (ewadd ?y ?z))"),
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

const ADDITION_COST: f32 = 1.0;
const MULTIPLY_COST: f32 = 6.0;

fn get_op_cost(egraph: &EGraph<TnsrLang, TnsrAnalysis>, enode: &TnsrLang) -> f32 {
    let x = |i: &Id| &egraph[*i].data;
    match enode {
        TnsrLang::Num(_)
        | TnsrLang::Var(_)
        | TnsrLang::Input(_)
        | TnsrLang::Noop(_) => 0.0,

        TnsrLang::Ewadd([a, b]) => {
            assert!(x(a).dtype == DataKind::Tnsr);
            assert!(x(b).dtype == DataKind::Tnsr);
            if x(a).constant_foldable && x(b).constant_foldable {
                0.0
            } else {
                let product: i32 = x(a).dims.iter().product::<i32>();
                product as f32 * ADDITION_COST
            }
        }

        TnsrLang::Ewmul([a, b]) => {
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

        TnsrLang::Matmul([a, b]) => {
            assert!(x(a).dtype == DataKind::Tnsr);
            assert!(x(b).dtype == DataKind::Tnsr);
            if x(a).constant_foldable && x(b).constant_foldable {
                0.0
            } else {
                (x(a).dims[0] * x(a).dims[1] * x(b).dims[1]) as f32 * MULTIPLY_COST
            }
        }
    }
}